"""
IR Pipeline
---------------------------------------
Model: ResNet-101 + GeM
Datasets:
  - Pretrain: SimCol3D (synthetic)
  - Fine-tune: C3VD (phantom, real colonoscope)
  - Test: Colon10K (patient benchmark)
Loss: Pose-aware InfoNCE
Metrics: Rank@K, mAP, Pose-aware Recall
"""

import os, glob, math, random
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics.pairwise import cosine_similarity


# =====================
# 1. Config & Paths
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Generalized dataset paths (set DATA_ROOT in your system or defaults to ./data)
DATA_ROOT     = os.getenv("DATA_ROOT", "./data")
C3VD_ROOT     = os.path.join(DATA_ROOT, "C3VD")
SIMCOL3D_ROOT = os.path.join(DATA_ROOT, "SimCol3D")
COLON10K_ROOT = os.path.join(DATA_ROOT, "Colon10K")


# =====================
# 2. Frame & Pose Loaders
# =====================
def collect_frames_c3vd(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, "*_color.png")))

def load_poses_c3vd(seq_dir, n_frames):
    pose_file = os.path.join(seq_dir,"pose.txt")
    if not os.path.isfile(pose_file): return None
    mats=[]
    with open(pose_file,'r') as f:
        for ln in f:
            vals=[float(x) for x in ln.strip().replace(","," ").split() if x]
            if len(vals)>=16:
                mats.append(np.array(vals[:16],dtype=np.float32).reshape(4,4))
    return np.stack(mats[:n_frames]) if mats else None

def collect_frames_simcol(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir,"FrameBuffer_*.png")))

def load_poses_simcol(frames_dir, scene_dir, n_frames):
    tag = os.path.basename(frames_dir).replace("Frames_","")
    pos_fp=os.path.join(scene_dir,f"SavedPosition_{tag}.txt")
    quat_fp=os.path.join(scene_dir,f"SavedRotationQuaternion_{tag}.txt")
    if not (os.path.isfile(pos_fp) and os.path.isfile(quat_fp)): return None
    P=np.loadtxt(pos_fp,dtype=np.float32)[:n_frames]
    Q=np.loadtxt(quat_fp,dtype=np.float32)[:n_frames]
    N=min(len(P),len(Q),n_frames); P,Q=P[:N],Q[:N]
    S=np.diag([1,1,-1]).astype(np.float32)
    R_lh=R.from_quat(Q).as_matrix(); R_rh=S@R_lh@S; t_rh=(S@P.T).T
    T=np.repeat(np.eye(4,dtype=np.float32)[None,...],N,axis=0)
    T[:,:3,:3]=R_rh; T[:,:3,3]=t_rh
    return T


# =====================
# 3. Datasets
# =====================
class PairDataset(Dataset):
    """Training: returns two augmented views + pose"""
    def __init__(self, root, tfm, collect_fn, pose_fn):
        self.tfm=tfm; self.index=[]; self.seq_frames={}; self.seq_poses={}
        seqs=[d for d in glob.glob(os.path.join(root,"*")) if os.path.isdir(d)]
        for seq in seqs:
            frames=collect_fn(seq); poses=pose_fn(seq,len(frames))
            if poses is None: continue
            self.seq_frames[seq]=frames; self.seq_poses[seq]=poses
            self.index+=[(f,seq) for f in frames]

    def __len__(self): return len(self.index)

    def __getitem__(self,i):
        f,seq=self.index[i]; img=Image.open(f).convert("RGB")
        pose=self.seq_poses[seq][self.seq_frames[seq].index(f)]
        img1,img2=self.tfm(img),self.tfm(img)
        return torch.stack([img1,img2],dim=0), torch.tensor(pose,dtype=torch.float32)

class Colon10KDataset(Dataset):
    """Colon10K: returns image + path"""
    def __init__(self, frames, tfm):
        self.frames=frames; self.tfm=tfm
    def __len__(self): return len(self.frames)
    def __getitem__(self,i):
        img=Image.open(self.frames[i]).convert("RGB")
        return self.tfm(img) if self.tfm else img, self.frames[i]


# =====================
# 4. Augmentations
# =====================
tfm_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(3)],p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
tfm_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =====================
# 5. Model (ResNet-101 + GeM)
# =====================
class GeM(nn.Module):
    def __init__(self,p=3.0,eps=1e-6):
        super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self,x):
        x=x.clamp(min=self.eps).pow(self.p)
        x=F.avg_pool2d(x,(x.size(-2),x.size(-1)))
        return x.pow(1/self.p)

class IRModel(nn.Module):
    def __init__(self):
        super().__init__()
        bb=models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.features=nn.Sequential(*list(bb.children())[:-2])
        self.pool=GeM()
    def global_embed(self,x):
        g=self.pool(self.features(x)).view(x.size(0),-1)
        return F.normalize(g,p=2,dim=1)
    def forward(self,x): return self.global_embed(x)


# =====================
# 6. Pose-aware InfoNCE Loss
# =====================
def pose_aware_infonce(desc, poses, temp=0.07, pos_thresh_trans=15.0, pos_thresh_rot=10.0, hard_neg_k=5):
    N=desc.size(0); desc=F.normalize(desc,dim=1)
    sim=torch.matmul(desc,desc.t())/temp
    P=poses[:,:3,3]; dists=torch.norm(P.unsqueeze(1)-P.unsqueeze(0),dim=-1)
    Rm=poses[:,:3,:3]; rel_rot=torch.matmul(Rm.unsqueeze(1),Rm.transpose(1,2).unsqueeze(0))
    tr=rel_rot.diagonal(offset=0,dim1=-2,dim2=-1).sum(-1)
    cos_theta=((tr-1)/2).clamp(-1,1); angles=torch.rad2deg(torch.acos(cos_theta))
    pos_mask=(dists<pos_thresh_trans)&(angles<pos_thresh_rot)
    eye=torch.eye(N,device=desc.device).bool()
    pos_mask=pos_mask&~eye; neg_mask=~pos_mask&~eye
    hard_neg_mask=torch.zeros_like(neg_mask)
    for i in range(N):
        sim_row=sim[i].detach().clone(); sim_row[~neg_mask[i]]=-1e9
        topk=torch.topk(sim_row,k=min(hard_neg_k,neg_mask[i].sum().item()),largest=True)
        hard_neg_mask[i,topk.indices]=True
    exp_sim=torch.exp(sim)
    pos_sum=(exp_sim*pos_mask).sum(dim=1)
    neg_sum=(exp_sim*hard_neg_mask).sum(dim=1)+1e-9
    loss=-torch.log(pos_sum/(pos_sum+neg_sum)+1e-9)
    return loss.mean()


# =====================
# 7. Training Loop
# =====================
def train_model(train_loader, model, opt, epochs=3):
    for ep in range(epochs):
        model.train(); run_loss=0.0
        for imgs,poses in tqdm(train_loader,desc=f"Epoch {ep+1}"):
            B,V,C,H,W=imgs.shape
            imgs=imgs.view(B*V,C,H,W).to(DEVICE)
            poses=poses.unsqueeze(1).repeat(1,V,1,1).view(B*V,4,4).to(DEVICE)
            opt.zero_grad(); desc=model(imgs)
            loss=pose_aware_infonce(desc,poses)
            loss.backward(); opt.step()
            run_loss+=loss.item()
        print(f"Epoch {ep+1} | Loss={run_loss/len(train_loader):.4f}")


# =====================
# 8. Colon10K Evaluation Helpers
# =====================
def parse_colon10k_matching(seq_dir,match_file):
    img_dir=os.path.join(seq_dir,"image")
    queries=[]; positives={}
    with open(match_file,'r') as f:
        for ln in f:
            parts=ln.strip().split()
            if len(parts)<2: continue
            q, pos_list=parts[0],parts[1:]
            q_path=os.path.join(img_dir,q)
            pos_paths=[os.path.join(img_dir,p) for p in pos_list]
            queries.append(q_path); positives[q_path]=pos_paths
    return queries, positives

@torch.no_grad()
def evaluate_colon10k(model,q_loader,g_loader,positives,device):
    model.eval()
    q_desc,q_paths=[],[]
    for imgs,paths in tqdm(q_loader,desc="Queries"):
        d=model.global_embed(imgs.to(device))
        q_desc.append(d.cpu().numpy()); q_paths+=list(paths)
    q_desc=np.vstack(q_desc)

    g_desc,g_paths=[],[]
    for imgs,paths in tqdm(g_loader,desc="Gallery"):
        d=model.global_embed(imgs.to(device))
        g_desc.append(d.cpu().numpy()); g_paths+=list(paths)
    g_desc=np.vstack(g_desc)

    sims=cosine_similarity(q_desc,g_desc)
    rank1,rank3,rank5,ap_list=0,0,0,[]
    for i,q in enumerate(q_paths):
        order=np.argsort(-sims[i])
        rel=[j for j,g in enumerate(g_paths) if g in positives[q]]
        if not rel: continue
        if g_paths[order[0]] in positives[q]: rank1+=1
        if any(g_paths[order[k]] in positives[q] for k in range(min(3,len(order)))): rank3+=1
        if any(g_paths[order[k]] in positives[q] for k in range(min(5,len(order)))): rank5+=1
        hit,precs=0,[]
        for k,j in enumerate(order,1):
            if g_paths[j] in positives[q]:
                hit+=1; precs.append(hit/k)
        if precs: ap_list.append(np.mean(precs))
    N=len(q_paths)
    print(f"[Colon10K] Rank@1={rank1/N:.4f}, Rank@3={rank3/N:.4f}, Rank@5={rank5/N:.4f}, mAP={np.mean(ap_list):.4f}")


# =====================
# 9. Run Pipeline
# =====================
if __name__ == "__main__":
    model=IRModel().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)

    # --- Phase 1: Pretrain on SimCol3D (subset for speed) ---
    print("Pretraining on SimCol3D...")
    scene_dir=os.path.join(SIMCOL3D_ROOT,"SyntheticColon_I")
    train_ds_sim=PairDataset(scene_dir,tfm_train,collect_frames_simcol,
                             lambda seq,n: load_poses_simcol(seq,scene_dir,n))
    train_loader_sim=DataLoader(train_ds_sim,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    train_model(train_loader_sim,model,opt,epochs=2)

    # --- Phase 2: Fine-tune on C3VD ---
    print("Fine-tuning on C3VD...")
    train_ds_c3vd=PairDataset(C3VD_ROOT,tfm_train,collect_frames_c3vd,load_poses_c3vd)
    train_loader_c3vd=DataLoader(train_ds_c3vd,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    train_model(train_loader_c3vd,model,opt,epochs=2)

    # --- Phase 3: Test on Colon10K (example sequence 07) ---
    seq_name="07"; seq_dir=os.path.join(COLON10K_ROOT,seq_name)
    queries,positives=parse_colon10k_matching(seq_dir,os.path.join(seq_dir,f"{seq_name}-matchings.txt"))
    gallery=sorted(glob.glob(os.path.join(seq_dir,"image","*.jpg")))

    q_ds=Colon10KDataset(queries,tfm_eval); g_ds=Colon10KDataset(gallery,tfm_eval)
    q_loader=DataLoader(q_ds,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
    g_loader=DataLoader(g_ds,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)

    evaluate_colon10k(model,q_loader,g_loader,positives,DEVICE)
