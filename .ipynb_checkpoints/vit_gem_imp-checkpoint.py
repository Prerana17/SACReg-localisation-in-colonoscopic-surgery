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
import re 
from scipy.spatial.transform import Rotation as R 
import math 
from sklearn.neighbors import KDTree 
from torchvision.models import vit_b_16, ViT_B_16_Weights # Add this import
import sys
# Add the path to your cloned NetVLAD repository so Python can find it
sys.path.append('/home/h8/zhyu410g/Team_project_IR/IR/netvlad_repo') 
from netvlad import NetVLAD # Now import from the cloned directory

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

def calculate_pose_distance(pose1, pose2):
    t_dist = np.linalg.norm(np.array(pose1[:3]) - np.array(pose2[:3]))
    q1 = R.from_quat(pose1[3:]) 
    q2 = R.from_quat(pose2[3:])
    dot_product = np.dot(q1.as_quat(), q2.as_quat())
    angle_rad = 2 * np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0)) 
    r_dist_deg = math.degrees(angle_rad) 
    return t_dist, r_dist_deg

def load_simcol3d_poses(pos_file_path, quat_file_path):
    positions = np.loadtxt(pos_file_path) 
    quaternions = np.loadtxt(quat_file_path) 
    return np.hstack((positions, quaternions)) 

def load_c3vd_poses(pose_file_path):
    poses_raw = np.loadtxt(pose_file_path, delimiter=',') # <--- ADD delimiter=','
    frames_poses = []
    for i in range(poses_raw.shape[0]):
        mat = poses_raw[i].reshape(4, 4)
        t = mat[:3, 3] 
        r_mat = mat[:3, :3] 
        q = R.from_matrix(r_mat).as_quat() 
        frames_poses.append(np.concatenate((t, q)))
    return np.array(frames_poses) 

def build_place_clusters(all_frame_paths, all_frame_pose_data, trans_threshold, rot_threshold):
    num_frames = len(all_frame_paths)
    uf = UnionFind(num_frames)
    
    print(f"Building place clusters for {num_frames} frames with Translation_thresh={trans_threshold}m, Rotation_thresh={rot_threshold} degrees...")
    print(f"Using KD-Tree for proximity search.")
    
    original_indices_with_pose = [i for i, pose in enumerate(all_frame_pose_data) if pose is not None]
    poses_array_for_kd = np.array([all_frame_pose_data[i] for i in original_indices_with_pose])
    
    if poses_array_for_kd.shape[0] < 2:
        print("Warning: Less than 2 valid pose data points. Cannot build KD-Tree or clusters effectively.")
        return np.full(num_frames, -1, dtype=int), {}, {} 

    kd_tree = KDTree(poses_array_for_kd)
    
    approx_radius = max(trans_threshold, rot_threshold / 100.0) * 5 

    for i_valid_idx, i_orig_idx in tqdm(enumerate(original_indices_with_pose), total=len(original_indices_with_pose), desc="Filtering neighbors by precise pose"):
        current_pose = poses_array_for_kd[i_valid_idx]
        
        indices_in_kd_array = kd_tree.query_radius(current_pose[np.newaxis, :], r=approx_radius)[0]
        
        for j_valid_idx in indices_in_kd_array:
            if i_valid_idx == j_valid_idx: 
                continue
            
            j_orig_idx = original_indices_with_pose[j_valid_idx] 
            
            if i_orig_idx < j_orig_idx: 
                other_pose = all_frame_pose_data[j_orig_idx] 
                t_dist, r_dist = calculate_pose_distance(current_pose, other_pose)
                if t_dist <= trans_threshold and r_dist <= rot_threshold:
                    uf.union(i_orig_idx, j_orig_idx) 

    path_to_cluster_id = {}
    cluster_map_id_to_path_indices = {} 
    cluster_labels_arr = np.full(num_frames, -1, dtype=int) 
    
    unique_cluster_roots_map = {} 
    next_compact_cluster_id = 0

    for i in range(num_frames):
        if all_frame_pose_data[i] is not None:
            root = uf.find(i)
            if root not in unique_cluster_roots_map:
                unique_cluster_roots_map[root] = next_compact_cluster_id
                next_compact_cluster_id += 1
                
            cluster_id = unique_cluster_roots_map[root]
            cluster_labels_arr[i] = cluster_id
            
            path_to_cluster_id[all_frame_paths[i]] = cluster_id
            
            if cluster_id not in cluster_map_id_to_path_indices:
                cluster_map_id_to_path_indices[cluster_id] = []
            cluster_map_id_to_path_indices[cluster_id].append(i) 

    print(f"Finished clustering. Found {next_compact_cluster_id} unique place clusters.")
    
    return cluster_labels_arr, cluster_map_id_to_path_indices, path_to_cluster_id

# --- 1. Data Loading and Triplet Generation (ColonoscopyTripletDataset) ---

class ColonoscopyTripletDataset(Dataset):
    def __init__(self, root_dirs, transform=None, triplet_strategy="pose_cluster_based", 
                 place_cluster_trans_threshold=0.01, 
                 place_cluster_rot_threshold=5.0): 
        
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.triplet_strategy = triplet_strategy
        self.place_cluster_trans_threshold = place_cluster_trans_threshold
        self.place_cluster_rot_threshold = place_cluster_rot_threshold
        
        self.all_frame_data = []   
        
        # Data needed for clustering (collected globally)
        all_paths_for_clustering = [] # Stores original frame paths in global index order for clustering
        all_poses_for_clustering = [] # Stores pose data for all frames (None if not available)
        #all_frame_paths_ordered = [] # Stores original frame paths in global index order
        
        temp_frames_by_original_video_id = {} 
        
        global_frame_index = 0 

        print(f"Starting data collection and pose loading from {len(self.root_dirs)} root directories...")
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                print(f"Warning: Training root directory not found: {root_dir}. Skipping: {root_dir}")
                continue

            top_level_folders = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
            
            candidate_video_folders = []
            for tlf in top_level_folders:
                if os.path.basename(tlf).startswith('SyntheticColon_'):
                    candidate_video_folders.extend(sorted([d for d in glob.glob(os.path.join(tlf, '*')) if os.path.isdir(d)]))
                else:
                    candidate_video_folders.append(tlf)
            
            for video_folder_path in candidate_video_folders:
                video_name_from_folder = os.path.basename(video_folder_path) 
                
                original_video_id_prefix = os.path.basename(root_dir)
                if os.path.basename(os.path.dirname(video_folder_path)).startswith('SyntheticColon_'):
                    original_video_id_prefix = f"{original_video_id_prefix}_{os.path.basename(os.path.dirname(video_folder_path))}"

                original_video_id = f"{original_video_id_prefix}_{video_name_from_folder}" 

                # --- Load pose data for the current video (all available poses) ---
                current_video_poses_full = [] # Store all poses loaded from file, will be indexed by frame_idx
                
                is_simcol3d_scene = os.path.basename(os.path.dirname(video_folder_path)).startswith('SyntheticColon_')
                
                if is_simcol3d_scene: 
                    scene_folder = os.path.dirname(video_folder_path) 
                    traj_match = re.match(r'(Frames_[SBO])(\d+)', video_name_from_folder)
                    if traj_match:
                        traj_prefix_str = traj_match.group(1) 
                        traj_num = traj_match.group(2)    
                        
                        pose_file_prefix = traj_prefix_str.split('_')[-1] 

                        try:
                            pos_file = os.path.join(scene_folder, f"SavedPosition_{pose_file_prefix}{traj_num}.txt")
                            quat_file = os.path.join(scene_folder, f"SavedRotationQuaternion_{pose_file_prefix}{traj_num}.txt")
                            current_video_poses_full = load_simcol3d_poses(pos_file, quat_file)
                            print(f"  Info: Loaded {len(current_video_poses_full)} poses from SimCol3D {video_folder_path}.")
                        except FileNotFoundError:
                            print(f"Warning: SimCol3D pose files not found for {video_folder_path} (expected {pos_file}, {quat_file}). Skipping poses for this video.")
                        except Exception as e:
                            print(f"Error loading SimCol3D poses for {video_folder_path}: {e}. Skipping poses.")
                    else:
                        print(f"Warning: SimCol3D video folder name {video_name_from_folder} not in expected format (e.g., Frames_S1). Skipping poses.")

                else: # Defaulting to C3VD-like structure
                    pose_file = os.path.join(video_folder_path, 'pose.txt')
                    try:
                        current_video_poses_full = load_c3vd_poses(pose_file)
                        print(f"  Info: Loaded {len(current_video_poses_full)} poses from C3VD {video_folder_path}.")
                    except FileNotFoundError:
                        print(f"Warning: C3VD pose file not found for {video_folder_path} (expected {pose_file}). Skipping poses for this video.")
                    except Exception as e:
                        print(f"Error loading C3VD poses for {video_folder_path}: {e}. Skipping poses.")
                
                # --- Load color frames (will only include actually existing color image files) ---
                color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*_color.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, 'FrameBuffer_*.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.png'))) 
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg'))) 

                if not color_frames: 
                    print(f"Warning: No color images found in folder: {video_folder_path}. Skipping this folder.")
                    continue 
                
                print(f"  Info: Loaded {len(color_frames)} color frames for {video_folder_path}.")
                
                if len(current_video_poses_full) > 0 and len(current_video_poses_full) != len(color_frames):
                     print(f"Warning: Frame count ({len(color_frames)}) and pose count ({len(current_video_poses_full)}) mismatch for {video_folder_path}. Will attempt to associate poses by frame index for existing images.")

                # Store frame data for clustering and later access
                for i, frame_path in enumerate(color_frames): 
                    frame_idx_str = re.search(r'(\d+)', os.path.basename(frame_path)).group(1) 
                    
                    pose_data = None # Default to None if not found/parsed
                    try:
                        frame_idx = int(frame_idx_str)
                        
                        # Only associate pose if it's available and frame_idx is within its bounds
                        if len(current_video_poses_full) > frame_idx: 
                            pose_data = current_video_poses_full[frame_idx] 
                        else:
                            print(f"Warning: Pose data for frame index {frame_idx} not found in {video_folder_path}'s pose.txt (available: {len(current_video_poses_full)}). Skipping pose association for this frame.")
                            pose_data = None # Pose data index out of bounds
                            
                    except (ValueError, AttributeError): 
                        print(f"Warning: Could not parse frame index from {frame_path}. Skipping pose association for this frame.")
                        pose_data = None # Error parsing index
                        
                    self.all_frame_data.append((frame_path, original_video_id, pose_data, global_frame_index)) 
                    
                    all_paths_for_clustering.append(frame_path)
                    all_poses_for_clustering.append(pose_data)
                    
                    if original_video_id not in temp_frames_by_original_video_id:
                        temp_frames_by_original_video_id[original_video_id] = []
                    temp_frames_by_original_video_id[original_video_id].append((frame_path, pose_data, global_frame_index))
                    
                    global_frame_index += 1
        
        self.original_video_ids = list(temp_frames_by_original_video_id.keys())
        self.frames_by_original_video_id = temp_frames_by_original_video_id

        self.path_to_cluster_id = {} 
        self.cluster_map_id_to_global_indices = {} 
        self.all_cluster_ids = [] 

        # Filter out frames without pose data for clustering (before calling build_place_clusters)
        valid_indices_for_clustering = [i for i, pose in enumerate(all_poses_for_clustering) if pose is not None]
        valid_paths_for_clustering = [all_paths_for_clustering[i] for i in valid_indices_for_clustering]
        valid_poses_for_clustering = [all_poses_for_clustering[i] for i in valid_indices_for_clustering]
        
        if len(valid_paths_for_clustering) < 2: 
            print("Warning: Insufficient valid pose data (less than 2 frames) for clustering. Proceeding without place clusters. Triplet strategy will fall back to original_video_id based logic.")
        else:
            print(f"Attempting to build place clusters for {len(valid_paths_for_clustering)} frames with valid poses...")
            
            # The build_place_clusters function will operate on these filtered valid paths/poses
            # It returns cluster_labels_arr (for filtered set), map of id->indices (filtered set), and path->cluster_id (filtered set)
            _, self.cluster_map_id_to_global_indices, self.path_to_cluster_id = \
                build_place_clusters(valid_paths_for_clustering, valid_poses_for_clustering,
                                     self.place_cluster_trans_threshold, self.place_cluster_rot_threshold)
            
            # Re-create self.cluster_to_frame_paths from path_to_cluster_id
            self.cluster_to_frame_paths = {cid: [] for cid in set(self.path_to_cluster_id.values())}
            for path, cid in self.path_to_cluster_id.items():
                self.cluster_to_frame_paths[cid].append(path)
            
            self.all_cluster_ids = list(self.cluster_to_frame_paths.keys())

            print(f"Total place clusters formed: {len(self.all_cluster_ids)}")
        
        if not self.all_frame_data:
            raise ValueError(f"No frames were loaded from any of the root directories: {root_dirs}. Please check paths and file patterns.")

        print(f"Dataset initialized: Total original video sequences loaded: {len(self.original_video_ids)}")
        print(f"Total individual frames loaded: {len(self.all_frame_data)}")


    def __len__(self):
        return len(self.all_frame_data)

    def __getitem__(self, idx):
        anchor_path, original_video_id, anchor_pose_data, global_anchor_index = self.all_frame_data[idx]
        
        anchor_cluster_id = self.path_to_cluster_id.get(anchor_path) 

        positive_path = None
        
        if anchor_cluster_id is None or self.triplet_strategy == "sequential_fallback": 
            current_video_frames_data = self.frames_by_original_video_id[original_video_id]
            current_video_frames_paths_only = [item[0] for item in current_video_frames_data]
            
            if len(current_video_frames_paths_only) == 1:
                positive_path = anchor_path
            else:
                anchor_idx_in_video = current_video_frames_paths_only.index(anchor_path)
                if anchor_idx_in_video < len(current_video_frames_paths_only) - 1:
                    positive_path = current_video_frames_paths_only[anchor_idx_in_video + 1]
                else: 
                    positive_path = current_video_frames_paths_only[anchor_idx_in_video - 1]
        else: 
            candidate_positive_paths = self.cluster_to_frame_paths[anchor_cluster_id]
            
            candidate_positive_paths = [p for p in candidate_positive_paths if p != anchor_path]
            
            if not candidate_positive_paths: 
                positive_path = anchor_path 
            else:
                positive_path = random.choice(candidate_positive_paths)

        negative_path = None
        
        if anchor_cluster_id is None or len(self.all_cluster_ids) < 2: 
            negative_path = None
            while negative_path is None:
                random_original_video_id = random.choice(list(self.frames_by_original_video_id.keys()))
                if random_original_video_id == original_video_id: 
                    continue
                
                candidate_negative_frames_data = self.frames_by_original_video_id[random_original_video_id]
                if candidate_negative_frames_data:
                    negative_path = random.choice([item[0] for item in candidate_negative_frames_data])
        else: 
            other_cluster_ids = [cid for cid in self.all_cluster_ids if cid != anchor_cluster_id]
            if not other_cluster_ids: 
                 print(f"Warning: Only one effective place cluster. Fallback to original_video_id based negative for {anchor_path}.")
                 negative_path = None
                 while negative_path is None:
                    random_original_video_id = random.choice(list(self.frames_by_original_video_id.keys()))
                    if random_original_video_id == original_video_id:
                        continue
                    
                    candidate_negative_frames_data = self.frames_by_original_video_id[random_original_video_id]
                    if candidate_negative_frames_data:
                        negative_path = random.choice([item[0] for item in candidate_negative_frames_data])
            else:
                random_negative_cluster_id = random.choice(other_cluster_ids)
                negative_path = random.choice(self.cluster_to_frame_paths[random_negative_cluster_id])


        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

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

class GeM(nn.Module):
    """
    Generalized Mean Pooling
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
        self.backbone_name = backbone_name 
        self.agg_layer_name = agg_layer 

        # --- 1. Backbone ---
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
            self.features = nn.Sequential(*list(backbone.children())[:-2]) 
            self.backbone_output_channels = 2048 
            self.backbone_global_descriptor_dim = 2048 
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.backbone_output_channels = 2048
            self.backbone_global_descriptor_dim = 2048
        elif backbone_name == 'vit_b_16': 
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT) 
            self.backbone_global_descriptor_dim = self.backbone.hidden_dim 
            print(f"Info: ViT backbone chosen. Ensure input image size is compatible (e.g., 224x224 for ViT-B/16).")
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50', 'resnet101', or 'vit_b_16'.")
        
        # --- 2. Aggregation Layer ---
        # Initialize aggregation layer based on chosen backbone and aggregation strategy
        if self.backbone_name.startswith('vit'):
            # For ViT (CLS Token strategy), no explicit aggregation layer is needed, as CLS token is the descriptor
            # We still keep 'agg_layer' in signature for consistency, but it acts as a no-op or is bypassed.
            if self.agg_layer_name == 'gem': # When agg_layer is 'gem' with ViT, we use CLS token directly
                print("Info: For ViT backbone, 'gem' aggregation is interpreted as using ViT's [CLS] token directly.")
                self.agg_layer = nn.Identity() # No-op layer
                self.output_dim = self.backbone_global_descriptor_dim
            elif self.agg_layer_name == 'netvlad':
                # This path is for future complex integration of NetVLAD with ViT dense features
                raise NotImplementedError("ViT with NetVLAD requires specific implementation for dense patch embeddings output. Not yet implemented.")
            else:
                raise ValueError(f"Unsupported aggregation '{self.agg_layer_name}' for ViT backbone.")
        elif self.backbone_name.startswith('resnet'): # For ResNet backbones, apply chosen aggregation
            if self.agg_layer_name == 'gem':
                self.agg_layer = GeM()
                self.output_dim = self.backbone_global_descriptor_dim
            elif self.agg_layer_name == 'netvlad':
                num_clusters = 64 
                vlad_in_dim = self.backbone_output_channels 
                self.agg_layer = NetVLAD(num_clusters=num_clusters, dim=vlad_in_dim, normalize_input=True) 
                self.output_dim = num_clusters * vlad_in_dim 
            else:
                raise ValueError(f"Unsupported aggregation '{self.agg_layer_name}' for ResNet backbone.")
        else:
            raise NotImplementedError("Aggregation layer initialization not implemented for selected backbone type.")


    def forward(self, x):
        print(f"DEBUG: Input to IRModel.forward shape: {x.shape}") # DEBUG PRINT

        if self.backbone_name.startswith('resnet'):
            x = self.features(x)  
            x = self.agg_layer(x) 
            x = x.view(x.size(0), -1) 
        elif self.backbone_name.startswith('vit'):
            x = self.backbone(x) 
            print(f"DEBUG: After self.backbone(x) shape: {x.shape}") # DEBUG PRINT
            
            # Ensure x has at least 2 dimensions before slicing
            if x.dim() < 2:
                raise ValueError(f"ViT backbone output has unexpected low dimension: {x.dim()}. Expected at least 2 for (num_tokens, embedding_dim). Shape: {x.shape}")

            # If ViT output is (batch_size, num_tokens, embedding_dim), then x[:, 0] is correct
            # If ViT output is already (batch_size, embedding_dim) (e.g., if it only returns CLS token), then x[:,0] will reduce it to 1D.
            # We need to ensure we correctly get the (batch_size, embedding_dim)
            if x.dim() == 3: # Standard ViT output (batch_size, num_tokens, embedding_dim)
                x = x[:, 0] # Extract the [CLS] token (N, D)
            elif x.dim() == 2: # Already (batch_size, embedding_dim), so no slicing needed. (Less common for raw backbone output)
                pass # x is already the desired shape
            else:
                raise ValueError(f"Unexpected ViT backbone output dimension for CLS token extraction. Shape: {x.shape}")

            print(f"DEBUG: After [CLS] token extraction shape: {x.shape}") # DEBUG PRINT
            
        else:
            raise NotImplementedError("Forward pass not implemented for the selected backbone.")

        # Ensure x is at least 2D before normalization
        if x.dim() < 2:
            raise ValueError(f"Descriptor before normalization has unexpected low dimension: {x.dim()}. Shape: {x.shape}")

        x = F.normalize(x, p=2, dim=1) 
        print(f"DEBUG: After normalization shape: {x.shape}") # DEBUG PRINT
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

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
            negative_desc = model(negative) # Corrected: ensures negative_desc comes from negative sample

            loss = criterion(anchor_desc, positive_desc, negative_desc)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    print("Training complete.")

    return model

def get_descriptors(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    
    descriptors = []
    # labels = [] # We don't need to store labels from the DataLoader here, as we use cached/parsed paths
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting Descriptors"): # Use _ to ignore labels
            imgs = imgs.to(device)
            desc = model(imgs)
            descriptors.append(desc.cpu().numpy())
            # No need to extend labels here, as we get paths/labels from cache or parsing
            
    if not descriptors:
        print("Warning: No descriptors were extracted. DataLoader might be empty.")
        return np.array([]), np.array([])
        
    # Return dummy labels if needed, or remove labels from the return signature if they are truly unused
    return np.vstack(descriptors), np.array([]) # Return empty array for labels as they are not used

def _evaluate_retrieval_core(query_descriptors, gallery_descriptors, query_labels_effective, gallery_labels_effective,
                            query_paths_for_eval, gallery_paths_for_eval):
    
    if len(query_labels_effective) == 0 or len(gallery_labels_effective) == 0:
        print("Warning: Effective query or gallery labels are empty. Cannot perform core evaluation.")
        return 0.0, 0.0

    similarities = cosine_similarity(query_descriptors, gallery_descriptors)
    
    rank1_count = 0
    aps = [] 

    for i in range(len(query_labels_effective)):
        query_label = query_labels_effective[i]
        
        current_query_path = query_paths_for_eval[i]
        
        relevant_gallery_indices = [j for j, label in enumerate(gallery_labels_effective) if label == query_label]
        
        num_true_relevant = len(relevant_gallery_indices)
        if current_query_path in gallery_paths_for_eval: 
            query_in_gallery_idx = gallery_paths_for_eval.index(current_query_path)
            if query_in_gallery_idx in relevant_gallery_indices:
                num_true_relevant -= 1 

        if num_true_relevant == 0: 
            aps.append(0.0)
            continue

        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # === Calculate RANK-1 ===
        for k, gallery_idx in enumerate(sorted_indices):
            if current_query_path == gallery_paths_for_eval[gallery_idx]:
                continue 
            
            if gallery_labels_effective[gallery_idx] == query_label: 
                rank1_count += 1
                break 
        
        # === Calculate Average Precision (AP) ===
        num_relevant_found_in_ranking = 0
        sum_precisions = 0
        
        for k, gallery_idx in enumerate(sorted_indices):
            if current_query_path == gallery_paths_for_eval[gallery_idx]:
                continue 
            
            if gallery_labels_effective[gallery_idx] == query_label:
                num_relevant_found_in_ranking += 1
                effective_rank_k_plus_1 = k + 1 
                if current_query_path in gallery_paths_for_eval:
                    self_idx_in_ranking = sorted_indices.tolist().index(gallery_paths_for_eval.index(current_query_path))
                    if self_idx_in_ranking < (k+1): 
                        effective_rank_k_plus_1 -= 1
                
                if effective_rank_k_plus_1 > 0: 
                    precision_at_k = num_relevant_found_in_ranking / effective_rank_k_plus_1
                    sum_precisions += precision_at_k
            
            if num_relevant_found_in_ranking == num_true_relevant:
                break

        if num_true_relevant > 0:
            ap = sum_precisions / num_true_relevant
            aps.append(ap)
        else:
            aps.append(0.0)

    rank1 = rank1_count / len(query_labels_effective)
    mAP = np.mean(aps) if len(aps) > 0 else 0.0

    return rank1, mAP


def evaluate_retrieval(query_descriptors, gallery_descriptors, query_paths, gallery_paths,
                       test_path_to_ground_truth_positives_map=None): # Changed from test_path_to_cluster_id
    
    if query_descriptors.size == 0 or gallery_descriptors.size == 0:
        print("Warning: Query or gallery descriptors are empty. Cannot perform evaluation.")
        return 0.0, 0.0

    if test_path_to_ground_truth_positives_map and len(test_path_to_ground_truth_positives_map) > 0:
        print("Evaluating with Colon10K's manual 'same place' annotations as effective relevance labels.")
        
        rank1_count = 0
        aps = []
        num_valid_queries = 0

        for i in range(len(query_paths)):
            query_path = query_paths[i]
            
            # Get the set of ground truth positive frame paths for this query
            ground_truth_positives_paths = test_path_to_ground_truth_positives_map.get(query_path)
            
            if not ground_truth_positives_paths:
                # print(f"Warning: No ground truth positives found for query {query_path}. Skipping this query.")
                continue # Skip queries without defined positives

            num_valid_queries += 1
            
            # --- For this query, identify which gallery items are truly relevant ---
            # Create a boolean mask or list of indices for gallery items that are true positives
            is_relevant_in_gallery = np.array([1 if p in ground_truth_positives_paths else 0 for p in gallery_paths])
            num_true_relevant_total = np.sum(is_relevant_in_gallery)
            
            # Ensure query itself is not counted as a positive if it's in the list
            # The Colon10K annotation style `query_id | positive intervals` implies query_id itself is included
            # So, if query_path is in ground_truth_positives_paths, num_true_relevant_total already counts it.
            # We still need to exclude it from ranking check.

            similarities_for_query = cosine_similarity(query_descriptors[i:i+1], gallery_descriptors)[0]
            sorted_indices = np.argsort(similarities_for_query)[::-1]
            
            # === Calculate RANK-1 ===
            for k, gallery_idx in enumerate(sorted_indices):
                # Exclude self-match for Rank-1 if the query itself is in the gallery and is not a "true relevant" item
                # Or simply exclude it if it's the exact query path
                if gallery_paths[gallery_idx] == query_path:
                    continue # Skip query itself in ranking
                
                # Check if the current top-ranked (non-self) item is a ground truth positive
                if gallery_paths[gallery_idx] in ground_truth_positives_paths:
                    rank1_count += 1
                    break 
            
            # === Calculate Average Precision (AP) ===
            num_relevant_found_in_ranking = 0
            sum_precisions = 0
            
            for k, gallery_idx in enumerate(sorted_indices):
                if gallery_paths[gallery_idx] == query_path:
                    continue # Skip query itself
                
                if gallery_paths[gallery_idx] in ground_truth_positives_paths:
                    num_relevant_found_in_ranking += 1
                    # Position in the effective ranking (excluding self) is k+1 - (1 if self was before this rank)
                    effective_rank_k_plus_1 = k + 1 
                    if query_path in gallery_paths: # Only adjust if query was in gallery
                        self_idx_in_ranking = sorted_indices.tolist().index(gallery_paths.index(query_path))
                        if self_idx_in_ranking < (k+1): 
                            effective_rank_k_plus_1 -= 1
                    
                    if effective_rank_k_plus_1 > 0: 
                        precision_at_k = num_relevant_found_in_ranking / effective_rank_k_plus_1
                        sum_precisions += precision_at_k
                
                if num_relevant_found_in_ranking == num_true_relevant_total:
                    break

            if num_true_relevant_total > 0:
                ap = sum_precisions / num_true_relevant_total
                aps.append(ap)
            else: # Should not happen if ground_truth_positives_paths is not empty
                aps.append(0.0)

        rank1 = rank1_count / num_valid_queries if num_valid_queries > 0 else 0.0
        mAP = np.mean(aps) if len(aps) > 0 else 0.0

        return rank1, mAP
    else:
        print("Error: Colon10K ground truth matchings map not provided or is empty. Cannot perform evaluation.")
        return 0.0, 0.0


# --- Helper Function: Parse Colon10K Matchings File ---
def parse_colon10k_matchings(matchings_file_path, image_folder_path):
    """
    Parses a Colon10K matchings.txt file to create a map from query frame path to its positive frame paths.
    Format: <query_id> | positive_interval1,positive_interval2,...
    e.g., 0 | 0-15
    e.g., 90 | 72-110,152,200-203,234-236

    Args:
        matchings_file_path (str): Path to the matchings.txt file.
        image_folder_path (str): Path to the corresponding image folder (e.g., /.../image/01/).
    Returns:
        dict: A map where keys are query frame paths and values are lists of positive frame paths.
    """
    query_path_to_positives_map = {}
    
    if not os.path.exists(matchings_file_path):
        print(f"Error: Colon10K matchings file not found: {matchings_file_path}")
        return {}

    with open(matchings_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line in matchings file: {line}")
                continue
            
            query_frame_idx_str = parts[0].strip()
            positive_intervals_str = parts[1].strip()

            try:
                query_frame_idx = int(query_frame_idx_str)
            except ValueError:
                print(f"Warning: Skipping line with invalid query_id: {line}")
                continue
            
            # Colon10K frame names are 'frameXXXXX.jpg'
            query_frame_filename = f"{query_frame_idx:05d}.jpg" # Assumes 5-digit padding
            query_frame_path = os.path.join(image_folder_path, query_frame_filename)

            positive_paths_for_query = set() # Use a set to avoid duplicates

            intervals = positive_intervals_str.split(',')
            for interval in intervals:
                interval = interval.strip()
                if '-' in interval:
                    start_idx_str, end_idx_str = interval.split('-')
                    try:
                        start_idx = int(start_idx_str)
                        end_idx = int(end_idx_str)
                        for idx in range(start_idx, end_idx + 1):
                            positive_frame_filename = f"{idx:05d}.jpg"
                            positive_paths_for_query.add(os.path.join(image_folder_path, positive_frame_filename))
                    except ValueError:
                        print(f"Warning: Skipping invalid interval: {interval} in line: {line}")
                else: # Single frame index
                    try:
                        idx = int(interval)
                        positive_frame_filename = f"{idx:05d}.jpg"
                        positive_paths_for_query.add(os.path.join(image_folder_path, positive_frame_filename))
                    except ValueError:
                        print(f"Warning: Skipping invalid single index: {interval} in line: {line}")
            
            query_path_to_positives_map[query_frame_path] = list(positive_paths_for_query)
            
    print(f"Parsed {len(query_path_to_positives_map)} queries from {matchings_file_path}")
    return query_path_to_positives_map

# --- Main Execution Logic ---

if __name__ == "__main__":
    # --- 0. Setup Save/Load Paths ---
    MODEL_CHECKPOINT_DIR = './checkpoints' 
    DESCRIPTORS_CACHE_DIR = './descriptors_cache' 
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DESCRIPTORS_CACHE_DIR, exist_ok=True)

    # 1. Configuration Parameters
    C3VD_ROOT = '/data/horse/ws/zhyu410g-horse_C3VD_data/exported/' 
    SIMCOL3D_ROOT = '/data/horse/ws/zhyu410g-horse_simcol/' 
    
    TRAIN_ROOT_DIRS = [C3VD_ROOT, SIMCOL3D_ROOT]

    # --- Colon10K Test Dataset (Actual Colon10K, not Oblique) ---
    COLON10K_ROOT = '/data/horse/ws/zhyu410g-horse_colon10k/real_10kdata' # e.g. /home/zhyu410g/Colon10K_Official/
    # Within this root, expect image/ folder and <CASE-ID>-matchings.txt files
    # From the document, it has 20 sequences (01-20)
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    MARGIN = 1.0 
    BACKBONE_NAME = 'vit_b_16' # <--- Change this to 'vit_b_16'
    AGG_LAYER = 'gem' # <--- Keep this as 'gem' to imply 'ViT (CLS Token)'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Place Clustering Thresholds (Used for C3VD/SimCol3D training)
    PLACE_CLUSTER_TRANS_THRESHOLD_SIMCOL = 0.02 
    PLACE_CLUSTER_ROT_THRESHOLD = 5.0 

    # Define filenames for saving model and descriptors
    TRAIN_DATASET_IDENTIFIER = "C3VD_SimCol3D_PoseClusters_T0.02_R5.0_ViT_CLS" 
    model_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"ir_model_{BACKBONE_NAME}_{AGG_LAYER}_ep{NUM_EPOCHS}_{TRAIN_DATASET_IDENTIFIER}.pth")
    query_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    query_paths_cache_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_paths_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")
    gallery_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    gallery_paths_cache_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_paths_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")


    # 2. Data Transformation (Data Augmentation)
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

    # 3. Prepare Training Data (Builds place clusters for C3VD/SimCol3D)
    train_dataset = ColonoscopyTripletDataset(root_dirs=TRAIN_ROOT_DIRS, 
                                              transform=data_transforms_train,
                                              triplet_strategy="pose_cluster_based", 
                                              place_cluster_trans_threshold=PLACE_CLUSTER_TRANS_THRESHOLD_SIMCOL, 
                                              place_cluster_rot_threshold=PLACE_CLUSTER_ROT_THRESHOLD) 

    if len(train_dataset) == 0:
        print(f"Error: Training dataset is empty. Check TRAIN_ROOT_DIRS: {TRAIN_ROOT_DIRS}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8, 
                              drop_last=True) 

    # 4. Initialize Model, Loss Function, and Optimizer
    model = IRModel(backbone_name=BACKBONE_NAME, agg_layer=AGG_LAYER)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Train Model or Load Pre-trained Model
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

    # --- 6. Prepare Evaluation Data (Colon10K Dataset with manual annotations) ---
    
    all_colon10k_image_paths = [] # All images in Colon10K
    colon10k_query_paths = []     # Paths of actual query frames (every 30th)
    colon10k_query_positives_map = {} # query_path -> list of positive frame paths

    print(f"\nPreparing Colon10K Dataset for evaluation from {COLON10K_ROOT}...")
    
    # Iterate through all 20 sequences (01 to 20)
    for case_id_num in range(1, 21):
        case_id = f"{case_id_num:02d}" # "01", "02", ... "20"
        case_image_folder = os.path.join(COLON10K_ROOT, case_id, 'image')
        case_matchings_file = os.path.join(COLON10K_ROOT, case_id, f"{case_id}-matchings.txt")
        
        if not os.path.isdir(case_image_folder):
            print(f"Warning: Colon10K image folder not found for case {case_id}: {case_image_folder}. Skipping.")
            continue
        if not os.path.exists(case_matchings_file):
            print(f"Warning: Colon10K matchings file not found for case {case_id}: {case_matchings_file}. Skipping.")
            continue

        # Load all images for this case (these form part of the gallery)
        case_frames = sorted(glob.glob(os.path.join(case_image_folder, '*.jpg')))
        if not case_frames:
            print(f"Warning: No frame*.jpg images found in {case_image_folder}. Skipping.")
            continue

        all_colon10k_image_paths.extend(case_frames)
        
        # Parse matchings file for queries and positives for this case
        case_query_positives_map = parse_colon10k_matchings(case_matchings_file, case_image_folder)
        
        # Add query paths to overall list and update the master map
        for q_path, pos_paths in case_query_positives_map.items():
            if q_path not in colon10k_query_paths: # Ensure unique query paths
                colon10k_query_paths.append(q_path)
            colon10k_query_positives_map[q_path] = pos_paths # Update/set positive paths for this query


    if not all_colon10k_image_paths:
        print(f"\nCRITICAL ERROR: No images loaded for Colon10K evaluation from {COLON10K_ROOT}. Please check paths and structure.")
        sys.exit(1)
    if not colon10k_query_paths:
        print(f"\nCRITICAL ERROR: No queries loaded for Colon10K evaluation. Please check {COLON10K_ROOT} and matchings files.")
        sys.exit(1)
    
    print(f"Successfully loaded {len(all_colon10k_image_paths)} gallery images for Colon10K evaluation.")
    print(f"Successfully loaded {len(colon10k_query_paths)} query images for Colon10K evaluation.")
    print(f"First 5 Colon10K gallery paths: {all_colon10k_image_paths[:min(5, len(all_colon10k_image_paths))]}")
    print(f"First 5 Colon10K query paths: {colon10k_query_paths[:min(5, len(colon10k_query_paths))]}")


    # Gallery dataset (all Colon10K images)
    gallery_dataset = InferenceDataset(all_colon10k_image_paths, 
                                       img_labels=np.arange(len(all_colon10k_image_paths)), # Use sequential indices as placeholder labels
                                       transform=data_transforms_eval)
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8)

    # Query dataset (specific query frames from Colon10K)
    query_dataset = InferenceDataset(colon10k_query_paths, 
                                     img_labels=np.arange(len(colon10k_query_paths)), # Use sequential indices as placeholder labels
                                     transform=data_transforms_eval)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8)


    # 7. Extract Descriptors or Load Cached Descriptors
    # The cache identifier needs to reflect that we are now using true Colon10K matching labels.
    desc_cache_identifier = TRAIN_DATASET_IDENTIFIER + "_Colon10K_GT_Eval" 
    query_desc_path_eff = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_desc_{BACKBONE_NAME}_{AGG_LAYER}_{desc_cache_identifier}.npy")
    query_paths_cache_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_paths_{BACKBONE_NAME}_{AGG_LAYER}_{desc_cache_identifier}.pkl")
    gallery_desc_path_eff = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_desc_{BACKBONE_NAME}_{AGG_LAYER}_{desc_cache_identifier}.npy")
    gallery_paths_cache_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_paths_{BACKBONE_NAME}_{AGG_LAYER}_{desc_cache_identifier}.pkl")

    if os.path.exists(query_desc_path_eff) and os.path.exists(query_paths_cache_path) and \
       os.path.exists(gallery_desc_path_eff) and os.path.exists(gallery_paths_cache_path):
        print(f"\nLoading cached descriptors and paths from {DESCRIPTORS_CACHE_DIR}...")
        query_descriptors = np.load(query_desc_path_eff)
        gallery_descriptors = np.load(gallery_desc_path_eff)
        with open(query_paths_cache_path, 'rb') as f:
            query_paths_for_eval = pickle.load(f) 
        with open(gallery_paths_cache_path, 'rb') as f:
            gallery_paths_for_eval = pickle.load(f) 
        print("Descriptors and paths loaded successfully.")
    else:
        print("\nNo cached descriptors found. Extracting descriptors...")
        query_descriptors, _ = get_descriptors(trained_model, query_loader, DEVICE) 
        gallery_descriptors, _ = get_descriptors(trained_model, gallery_loader, DEVICE)
        
        np.save(query_desc_path_eff, query_descriptors)
        with open(query_paths_cache_path, 'wb') as f:
            pickle.dump(colon10k_query_paths, f) # Save the actual Colon10K query paths
        
        np.save(gallery_desc_path_eff, gallery_descriptors)
        with open(gallery_paths_cache_path, 'wb') as f:
            pickle.dump(all_colon10k_image_paths, f) # Save all Colon10K gallery paths
        print("Descriptors extracted and cached successfully.")
        
        query_paths_for_eval = colon10k_query_paths
        gallery_paths_for_eval = all_colon10k_image_paths


    # 8. Evaluate Retrieval Performance
    print("Evaluating retrieval performance...")

    # Pass the Colon10K specific ground truth map to evaluate_retrieval
    # Note: query_labels_original_video_id and gallery_labels_original_video_id are no longer used for Colon10K eval
    rank1, mAP = evaluate_retrieval(query_descriptors, gallery_descriptors, 
                                    query_paths=query_paths_for_eval, 
                                    gallery_paths=gallery_paths_for_eval,
                                    test_path_to_ground_truth_positives_map=colon10k_query_positives_map) 

    print(f"\n--- Evaluation Results ({AGG_LAYER.upper()} aggregation) ---")
    print(f"RANK-1 (Colon10K GT 'same place' eval): {rank1:.4f}") 
    print(f"mAP (Colon10K GT 'same place' eval): {mAP:.4f}")

    # TODO: If you want to compare GeM vs. NetVLAD:
    # 1. Implement NetVLAD module or find and integrate an existing PyTorch NetVLAD library.
    # 2. Repeat the training and evaluation process here for NetVLAD.
