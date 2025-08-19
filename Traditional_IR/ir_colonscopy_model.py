import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

# --- 1. 数据加载和三元组生成 (从 dataset.py 整合) ---

class ColonoscopyTripletDataset(Dataset):
    def __init__(self, root_dirs, transform=None, triplet_strategy="sequential_intra_video"): 
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.triplet_strategy = triplet_strategy
        
        self.video_folders = []  # 存储所有实际的视频序列文件夹路径
        self.video_frames = []   # 存储所有帧的 (frame_path, unique_video_id)
        self.video_ids = {}      # unique_video_id -> 内部数字ID
        
        global_video_counter = 0 
        
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                print(f"Warning: Training root directory not found: {root_dir}. Skipping.")
                continue

            top_level_folders = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
            
            candidate_video_folders = []
            for tlf in top_level_folders:
                if os.path.basename(tlf).startswith('SyntheticColon_'):
                    candidate_video_folders.extend(sorted([d for d in glob.glob(os.path.join(tlf, '*')) if os.path.isdir(d)]))
                else:
                    candidate_video_folders.append(tlf)
            
            # --- 核心修复：将找到的视频文件夹添加到 self.video_folders ---
            self.video_folders.extend(candidate_video_folders) 
            # -----------------------------------------------------------
            
            for video_folder_path in candidate_video_folders:
                video_name_from_folder = os.path.basename(video_folder_path) 

                unique_video_id_prefix = os.path.basename(root_dir)
                if os.path.basename(os.path.dirname(video_folder_path)).startswith('SyntheticColon_'):
                    unique_video_id_prefix = f"{unique_video_id_prefix}_{os.path.basename(os.path.dirname(video_folder_path))}"

                unique_video_id = f"{unique_video_id_prefix}_{video_name_from_folder}_{global_video_counter}" 
                self.video_ids[unique_video_id] = global_video_counter

                color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*_color.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, 'FrameBuffer_*.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg'))) 

                if not color_frames: 
                    print(f"Warning: No color images found in folder: {video_folder_path}. Skipping.")
                    continue 
                
                self.video_frames.extend([(frame_path, unique_video_id) for frame_path in color_frames])
                global_video_counter += 1 

        self.frames_by_video_id = {vid: [] for vid in self.video_ids.keys()}
        for f_path, v_id in self.video_frames:
            self.frames_by_video_id[v_id].append(f_path)
            
        print(f"Dataset initialized: Found {len(self.video_folders)} actual video folders in total from {len(self.root_dirs)} roots.")
        print(f"Total unique video sequences (trajectories): {len(self.video_ids)}")
        print(f"Total individual frames loaded: {len(self.video_frames)}")


    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        anchor_path, anchor_video_id = self.video_frames[idx]
        
        # --- Positive 样本选择 ---
        if self.triplet_strategy == "sequential_intra_video":
            current_video_frames = self.frames_by_video_id[anchor_video_id]
            
            # 如果视频序列只有一帧，则 Anchor 就是 Positive (损失会是0)
            if len(current_video_frames) == 1:
                positive_path = anchor_path
            else:
                anchor_idx_in_video = current_video_frames.index(anchor_path)
                if anchor_idx_in_video < len(current_video_frames) - 1:
                    positive_path = current_video_frames[anchor_idx_in_video + 1]
                else: 
                    positive_path = current_video_frames[anchor_idx_in_video - 1]

        elif self.triplet_strategy == "semantic_similar_intra_video":
            raise NotImplementedError("Semantic similarity strategy requires custom implementation for positive mining.")
        else:
            raise ValueError(f"Unknown triplet strategy: {self.triplet_strategy}")

        # --- Negative 样本选择 ---
        negative_path = None
        # 避免无限循环：如果整个数据集只有一个视频序列，无法找到负样本
        if len(self.video_ids) < 2: 
            print("Warning: Only one video sequence found in dataset. Negative sampling not possible. Triplet Loss may not function correctly.")
            negative_path = random.choice(self.frames_by_video_id[anchor_video_id]) # 随机选一个同视频的作为负样本，这可能导致训练无效
            if negative_path == anchor_path: # 避免 A=N
                 negative_path = None # Force re-selection if only one frame too
                 while negative_path is None or negative_path == anchor_path:
                    negative_path = random.choice(self.frames_by_video_id[anchor_video_id])

        while negative_path is None:
            negative_video_id = random.choice(list(self.video_ids.keys()))
            if negative_video_id == anchor_video_id:
                continue 
            
            if self.frames_by_video_id[negative_video_id]:
                negative_path = random.choice(self.frames_by_video_id[negative_video_id])
            
        # --- 图像加载和转换 ---
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# 评估时使用的 Dataset，只加载单张图片 (不变)
class InferenceDataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.img_labels[idx]

# --- 2. 模型定义 (不变) ---

class GeM(nn.Module):
    """
    Generalized Mean Pooling
    Args:
        p (float): initial p value
        eps (float): small value to avoid division by zero
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class IRModel(nn.Module):
    def __init__(self, backbone_name='resnet50', agg_layer='gem'):
        super(IRModel, self).__init__()
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")
        
        self.features = nn.Sequential(*list(backbone.children())[:-2]) 
        
        if agg_layer == 'gem':
            self.agg_layer = GeM()
            self.output_dim = 2048 
        elif agg_layer == 'netvlad':
            raise NotImplementedError("NetVLAD integration requires a separate library or custom implementation.")
        else:
            raise ValueError("Unsupported aggregation layer. Choose 'gem' or 'netvlad'.")

    def forward(self, x):
        x = self.features(x)
        x = self.agg_layer(x)
        x = x.view(x.size(0), -1) 
        x = F.normalize(x, p=2, dim=1) 
        return x

# --- 3. 损失函数 (不变) ---

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# --- 4. 训练脚本 (不变) ---

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_desc = model(anchor)
            positive_desc = model(positive)
            negative_desc = model(negative)

            loss = criterion(anchor_desc, positive_desc, negative_desc)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    print("Training complete.")

    return model

# --- 5. 评估脚本 (不变) ---

def get_descriptors(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    
    descriptors = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting Descriptors"):
            imgs = imgs.to(device)
            desc = model(imgs)
            descriptors.append(desc.cpu().numpy())
            labels.extend(lbls.cpu().numpy().tolist() if isinstance(lbls, torch.Tensor) else lbls)
            
    if not descriptors:
        print("Warning: No descriptors were extracted. DataLoader might be empty.")
        return np.array([]), np.array([])
        
    return np.vstack(descriptors), np.array(labels)

def evaluate_retrieval(query_descriptors, gallery_descriptors, query_labels, gallery_labels,
                       query_paths=None, gallery_paths=None): # 新增 query_paths 和 gallery_paths
    
    if query_descriptors.size == 0 or gallery_descriptors.size == 0:
        print("Warning: Query or gallery descriptors are empty. Cannot perform evaluation.")
        return 0.0, 0.0

    similarities = cosine_similarity(query_descriptors, gallery_descriptors)
    
    rank1_count = 0
    aps = [] 

    for i in range(len(query_labels)):
        query_label = query_labels[i]
        
        current_query_path = query_paths[i] if query_paths is not None else None
        
        relevant_gallery_indices = [j for j, label in enumerate(gallery_labels) if label == query_label]
        
        num_true_relevant = len(relevant_gallery_indices)
        if current_query_path is not None and current_query_path in gallery_paths:
            query_in_gallery_idx = gallery_paths.index(current_query_path)
            if query_in_gallery_idx in relevant_gallery_indices:
                num_true_relevant -= 1 

        if num_true_relevant == 0:
            aps.append(0.0)
            continue

        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # === 计算 RANK-1 ===
        found_rank1 = False
        for k, gallery_idx in enumerate(sorted_indices):
            if current_query_path is not None and gallery_paths is not None:
                if gallery_paths[gallery_idx] == current_query_path:
                    continue 
            
            if gallery_labels[gallery_idx] == query_label: 
                rank1_count += 1
                found_rank1 = True 
                break 

        # === 计算 Average Precision (AP) ===
        num_relevant_found_in_ranking = 0
        sum_precisions = 0
        
        for k, gallery_idx in enumerate(sorted_indices):
            if current_query_path is not None and gallery_paths is not None:
                if gallery_paths[gallery_idx] == current_query_path:
                    continue 
            
            if gallery_labels[gallery_idx] == query_label:
                num_relevant_found_in_ranking += 1
                precision_at_k = num_relevant_found_in_ranking / (k + 1 - (1 if current_query_path is not None and gallery_paths is not None and gallery_paths.index(current_query_path) < (k+1) else 0) ) # 调整分母，如果自身被跳过，排名会提前
                sum_precisions += precision_at_k
            
            if num_relevant_found_in_ranking == num_true_relevant:
                break

        if num_true_relevant > 0:
            ap = sum_precisions / num_true_relevant
            aps.append(ap)
        else:
            aps.append(0.0)

    rank1 = rank1_count / len(query_labels)
    mAP = np.mean(aps) if len(aps) > 0 else 0.0

    return rank1, mAP

# --- 主运行逻辑 (从 main.py 整合) ---

if __name__ == "__main__":
    # --- 0. 设置保存/加载路径 ---
    MODEL_CHECKPOINT_DIR = './checkpoints' 
    DESCRIPTORS_CACHE_DIR = './descriptors_cache' 
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DESCRIPTORS_CACHE_DIR, exist_ok=True)

    # 1. 配置参数
    # C3VD 数据集根目录
    C3VD_ROOT = '/data/horse/ws/zhyu410g-horse_C3VD_data/exported/' 
    # SimCol3D 数据集根目录 (包含 SyntheticColon_I/II/III 的父目录)
    SIMCOL3D_ROOT = '/data/horse/ws/zhyu410g-horse_simcol/' 

    # 组合训练数据集的根目录列表
    TRAIN_ROOT_DIRS = [C3VD_ROOT, SIMCOL3D_ROOT]

    # Colon10K 测试数据集根目录
    TEST_ROOT_DIR = os.path.expanduser('~/Colon10K/10kdata/test')     

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    MARGIN = 1.0 
    BACKBONE_NAME = 'resnet50' 
    AGG_LAYER = 'gem' 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型和描述符保存的文件名 (现在考虑到多个训练数据集)
    # 你可以加入一个 TRAIN_DATASET_IDENTIFIER 来区分不同的训练集组合
    TRAIN_DATASET_IDENTIFIER = "C3VD_SimCol3D_Combined" 
    model_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"ir_model_{BACKBONE_NAME}_{AGG_LAYER}_ep{NUM_EPOCHS}_{TRAIN_DATASET_IDENTIFIER}.pth")
    query_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    query_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_labels_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")
    gallery_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    gallery_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_labels_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")


    # 2. 数据转换 (Data Augmentation)
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) 
        ], p=0.8),
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 准备训练数据
    # 传入 TRAIN_ROOT_DIRS 列表
    train_dataset = ColonoscopyTripletDataset(root_dirs=TRAIN_ROOT_DIRS, 
                                              transform=data_transforms_train,
                                              triplet_strategy="sequential_intra_video")
    if len(train_dataset) == 0:
        print(f"Error: Training dataset is empty. Check TRAIN_ROOT_DIRS: {TRAIN_ROOT_DIRS}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8, 
                              drop_last=True) 

    # 4. 初始化模型、损失函数和优化器
    model = IRModel(backbone_name=BACKBONE_NAME, agg_layer=AGG_LAYER)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练模型 或 加载已训练模型
    if os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("No pre-trained model found. Starting training...")
        trained_model = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Trained model saved to {model_save_path}")
    
    trained_model = model

    # 6. 准备评估数据 (Colon10K)
    test_video_folders = sorted([d for d in glob.glob(os.path.join(TEST_ROOT_DIR, '*')) if os.path.isdir(d)])
    
    all_test_img_paths = []
    all_test_img_labels = [] 

    for folder in test_video_folders:
        video_id = os.path.basename(folder)
        color_frames = sorted(glob.glob(os.path.join(folder, '*.jpg'))) 
        if not color_frames:
            print(f"Warning: No .jpg files found in folder: {folder}. Skipping this folder.")
            continue 
        for frame_path in color_frames:
            all_test_img_paths.append(frame_path)
            all_test_img_labels.append(video_id) 
    
    if not all_test_img_paths:
        print(f"\nCRITICAL ERROR: 'all_test_img_paths' is EMPTY after attempting to load from {TEST_ROOT_DIR}.")
        print("This means no .jpg images were found conforming to the expected structure.")
        print("Please double-check the TEST_ROOT_DIR absolute path and the file structure within it.")
        sys.exit(1)
    else:
        print(f"\nSuccessfully loaded {len(all_test_img_paths)} images for evaluation from {TEST_ROOT_DIR}.")
        print(f"First 5 test image paths: {all_test_img_paths[:min(5, len(all_test_img_paths))]}")
        print(f"First 5 test image labels: {all_test_img_labels[:min(5, len(all_test_img_labels))]}")

    gallery_dataset = InferenceDataset(all_test_img_paths, all_test_img_labels, data_transforms_eval)
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8)

    num_queries = min(100, len(all_test_img_paths))
    if num_queries == 0:
        print("CRITICAL ERROR: num_queries is 0. No queries will be processed for evaluation.")
        sys.exit(1)
        
    query_indices = random.sample(range(len(all_test_img_paths)), num_queries)
    query_paths = [all_test_img_paths[i] for i in query_indices]
    query_labels = [all_test_img_labels[i] for i in query_indices]

    query_dataset = InferenceDataset(query_paths, query_labels, data_transforms_eval)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8)

    # 7. 提取描述符 或 加载已提取的描述符
    if os.path.exists(query_desc_path) and os.path.exists(query_labels_path) and \
       os.path.exists(gallery_desc_path) and os.path.exists(gallery_labels_path):
        print(f"\nLoading cached descriptors from {DESCRIPTORS_CACHE_DIR}...")
        query_descriptors = np.load(query_desc_path)
        gallery_descriptors = np.load(gallery_desc_path)
        with open(query_labels_path, 'rb') as f:
            extracted_query_labels = pickle.load(f)
        with open(gallery_labels_path, 'rb') as f:
            extracted_gallery_labels = pickle.load(f)
        print("Descriptors loaded successfully.")
    else:
        print("\nNo cached descriptors found. Extracting descriptors...")
        query_descriptors, extracted_query_labels = get_descriptors(trained_model, query_loader, DEVICE)
        gallery_descriptors, extracted_gallery_labels = get_descriptors(trained_model, gallery_loader, DEVICE)
        
        np.save(query_desc_path, query_descriptors)
        np.save(gallery_desc_path, gallery_descriptors)
        with open(query_labels_path, 'wb') as f:
            pickle.dump(extracted_query_labels, f)
        with open(gallery_labels_path, 'wb') as f:
            pickle.dump(extracted_gallery_labels, f)
        print("Descriptors extracted and cached successfully.")

    # 8. 评估检索性能
    print("Evaluating retrieval performance...")
    # 传入 query_paths 和 all_test_img_paths 用于排除自匹配
    rank1, mAP = evaluate_retrieval(query_descriptors, gallery_descriptors, extracted_query_labels, extracted_gallery_labels,
                                    query_paths=query_paths, gallery_paths=all_test_img_paths) 

    print(f"\n--- Evaluation Results ({AGG_LAYER.upper()} aggregation) ---")
    print(f"RANK-1 (excluding self-match): {rank1:.4f}") 
    print(f"mAP: {mAP:.4f}")