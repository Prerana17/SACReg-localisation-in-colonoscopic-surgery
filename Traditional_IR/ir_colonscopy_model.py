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

# --- 1. Data Loading and Triplet Generation (Integrated from dataset.py) ---

class ColonoscopyTripletDataset(Dataset):
    def __init__(self, root_dirs, transform=None, triplet_strategy="sequential_intra_video"): 
        # Ensure root_dirs is a list
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.triplet_strategy = triplet_strategy
        
        self.video_folders = []  # Stores paths to all actual video sequence folders
        self.video_frames = []   # Stores (frame_path, unique_video_id) for all frames
        self.video_ids = {}      # Maps unique_video_id to an internal numeric ID
        
        global_video_counter = 0 # Used to generate a globally unique ID for each video sequence
        
        for root_dir in self.root_dirs:
            # Check if each root directory exists
            if not os.path.isdir(root_dir):
                print(f"Warning: Training root directory not found: {root_dir}. Skipping.")
                continue

            # Collect all top-level folders within the current root_dir (e.g., cecum_t1_a or SyntheticColon_I/II/III)
            top_level_folders = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
            
            candidate_video_folders = []
            for tlf in top_level_folders:
                # If it's a SimCol3D scene folder (e.g., SyntheticColon_I/II/III)
                if os.path.basename(tlf).startswith('SyntheticColon_'):
                    # Video folders are one level deeper (e.g., Frames_SXX, Frames_BXX, Frames_OXX)
                    candidate_video_folders.extend(sorted([d for d in glob.glob(os.path.join(tlf, '*')) if os.path.isdir(d)]))
                else:
                    # Otherwise, it's likely a C3VD video folder (e.g., cecum_t1_a, sigmoid_t2_a)
                    candidate_video_folders.append(tlf)
            
            # --- Core Fix: Add found video folders to self.video_folders ---
            self.video_folders.extend(candidate_video_folders) 
            # -----------------------------------------------------------
            
            # Now, iterate through the candidate_video_folders to get frames
            for video_folder_path in candidate_video_folders:
                video_name_from_folder = os.path.basename(video_folder_path) 

                # Create a globally unique video ID by combining root directory basename,
                # parent folder (if SimCol3D scene), and original folder name with a counter.
                unique_video_id_prefix = os.path.basename(root_dir)
                if os.path.basename(os.path.dirname(video_folder_path)).startswith('SyntheticColon_'):
                    unique_video_id_prefix = f"{unique_video_id_prefix}_{os.path.basename(os.path.dirname(video_folder_path))}"

                unique_video_id = f"{unique_video_id_prefix}_{video_name_from_folder}_{global_video_counter}" 
                self.video_ids[unique_video_id] = global_video_counter # Map to an internal numeric ID

                # Search for color images (prioritize _color.png (C3VD), then FrameBuffer_*.png (SimCol3D), 
                # then general *.png, finally *.jpg)
                color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*_color.png'))) # For C3VD
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, 'FrameBuffer_*.png'))) # For SimCol3D
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.png'))) # General .png
                if not color_frames: 
                    color_frames = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg'))) # General .jpg (e.g., for Colon10K test or other formats)

                if not color_frames: 
                    print(f"Warning: No color images found in folder: {video_folder_path}. Skipping.")
                    continue 
                
                self.video_frames.extend([(frame_path, unique_video_id) for frame_path in color_frames])
                global_video_counter += 1 

        # Index for faster negative sample lookup (now using unique_video_id as key)
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
        
        # --- Positive Sample Selection ---
        if self.triplet_strategy == "sequential_intra_video":
            current_video_frames = self.frames_by_video_id[anchor_video_id]
            
            # If video sequence has only one frame, Anchor is Positive (loss will be 0)
            if len(current_video_frames) == 1:
                positive_path = anchor_path
            else:
                anchor_idx_in_video = current_video_frames.index(anchor_path)
                if anchor_idx_in_video < len(current_video_frames) - 1:
                    positive_path = current_video_frames[anchor_idx_in_video + 1]
                else: # If Anchor is the last frame of the video, choose the previous frame
                    positive_path = current_video_frames[anchor_idx_in_video - 1]

        elif self.triplet_strategy == "semantic_similar_intra_video":
            raise NotImplementedError("Semantic similarity strategy requires custom implementation for positive mining.")
        else:
            raise ValueError(f"Unknown triplet strategy: {self.triplet_strategy}")

        # --- Negative Sample Selection ---
        negative_path = None
        # Avoid infinite loop: if the entire dataset has only one video sequence, cannot find a negative sample
        if len(self.video_ids) < 2: 
            print("Warning: Only one video sequence found in dataset. Negative sampling not possible. Triplet Loss may not function correctly.")
            # Fallback: choose a random frame from the same video as negative (will lead to non-informative triplets)
            negative_path = random.choice(self.frames_by_video_id[anchor_video_id]) 
            if negative_path == anchor_path: # Avoid A=N if there's only one frame in the sequence
                 negative_path = None 
                 while negative_path is None or negative_path == anchor_path:
                    negative_path = random.choice(self.frames_by_video_id[anchor_video_id])

        while negative_path is None:
            negative_video_id = random.choice(list(self.video_ids.keys()))
            if negative_video_id == anchor_video_id:
                continue 
            
            if self.frames_by_video_id[negative_video_id]:
                negative_path = random.choice(self.frames_by_video_id[negative_video_id])
            
        # --- Image Loading and Transformation ---
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# Inference Dataset (for evaluation, loads single images) (Unchanged)
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

# --- 2. Model Definition (Unchanged) ---

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

# --- 3. Loss Function (Unchanged) ---

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# --- 4. Training Script (Unchanged) ---

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

# --- 5. Evaluation Script (Unchanged) ---

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
                       query_paths=None, gallery_paths=None): 
    
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
        # If query itself is in gallery, exclude it from count of true relevant items
        if current_query_path is not None and current_query_path in gallery_paths:
            query_in_gallery_idx = gallery_paths.index(current_query_path)
            if query_in_gallery_idx in relevant_gallery_indices:
                num_true_relevant -= 1 

        if num_true_relevant == 0: # If no relevant items other than query itself, AP is 0
            aps.append(0.0)
            continue

        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # === Calculate RANK-1 ===
        found_rank1 = False
        for k, gallery_idx in enumerate(sorted_indices):
            # Exclude self-match: if current gallery item is the query itself, skip
            if current_query_path is not None and gallery_paths is not None:
                if gallery_paths[gallery_idx] == current_query_path:
                    continue 
            
            # If the current top-ranked (non-self) item has the same label as the query, it's a Rank-1 hit
            if gallery_labels[gallery_idx] == query_label: 
                rank1_count += 1
                found_rank1 = True 
                break # Found the first relevant item, stop searching for Rank-1
        
        # === Calculate Average Precision (AP) ===
        num_relevant_found_in_ranking = 0 # Count of relevant items found so far in the ranking
        sum_precisions = 0
        
        for k, gallery_idx in enumerate(sorted_indices):
            # Exclude self-match for AP calculation as well
            if current_query_path is not None and gallery_paths is not None:
                if gallery_paths[gallery_idx] == current_query_path:
                    continue 
            
            if gallery_labels[gallery_idx] == query_label:
                num_relevant_found_in_ranking += 1
                # Precision at this rank (number of relevant items found so far / total items retrieved so far)
                # Denominator (k+1) needs adjustment if self-match was skipped earlier in the rank
                precision_at_k = num_relevant_found_in_ranking / (k + 1 - (1 if current_query_path is not None and gallery_paths is not None and gallery_paths.index(current_query_path) < (k+1) else 0) ) 
                sum_precisions += precision_at_k
            
            # If all true relevant items have been found, can break early for optimization
            if num_relevant_found_in_ranking == num_true_relevant:
                break

        if num_true_relevant > 0:
            ap = sum_precisions / num_true_relevant # AP is the average of precisions at each relevant item's rank
            aps.append(ap)
        else: # If a query has no relevant items (other than itself), its AP is 0
            aps.append(0.0)

    rank1 = rank1_count / len(query_labels) # Rank-1 accuracy for all queries
    mAP = np.mean(aps) if len(aps) > 0 else 0.0 # Mean Average Precision across all queries

    return rank1, mAP

# --- Main Execution Logic (Integrated from main.py) ---

if __name__ == "__main__":
    # --- 0. Setup Save/Load Paths ---
    MODEL_CHECKPOINT_DIR = './checkpoints' # Directory to save trained models
    DESCRIPTORS_CACHE_DIR = './descriptors_cache' # Directory to cache extracted descriptors
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True) # Create directories if they don't exist
    os.makedirs(DESCRIPTORS_CACHE_DIR, exist_ok=True)

    # 1. Configuration Parameters
    # C3VD Dataset root directory
    C3VD_ROOT = '/data/horse/ws/zhyu410g-horse_C3VD_data/exported/' 
    # SimCol3D Dataset root directory (parent of SyntheticColon_I/II/III folders)
    SIMCOL3D_ROOT = '/data/horse/ws/zhyu410g-horse_simcol/' 

    # List of root directories for training datasets (C3VD and SimCol3D)
    TRAIN_ROOT_DIRS = [C3VD_ROOT, SIMCOL3D_ROOT]

    # Colon10K Test Dataset root directory
    TEST_ROOT_DIR = os.path.expanduser('~/Colon10K/10kdata/test')     

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    MARGIN = 1.0 # Margin for Triplet Loss
    BACKBONE_NAME = 'resnet50' # or 'resnet101'
    AGG_LAYER = 'gem' # or 'netvlad' (requires additional implementation)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define filenames for saving model and descriptors (now considering multiple training datasets)
    # A TRAIN_DATASET_IDENTIFIER is used to differentiate models trained with different dataset combinations.
    TRAIN_DATASET_IDENTIFIER = "C3VD_SimCol3D_Combined" 
    model_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"ir_model_{BACKBONE_NAME}_{AGG_LAYER}_ep{NUM_EPOCHS}_{TRAIN_DATASET_IDENTIFIER}.pth")
    query_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    query_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_labels_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")
    gallery_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_desc_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.npy")
    gallery_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_labels_{BACKBONE_NAME}_{AGG_LAYER}_{TRAIN_DATASET_IDENTIFIER}.pkl")


    # 2. Data Transformation (Data Augmentation)
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) # 'hue' parameter removed to prevent OverflowError
        ], p=0.8),
        transforms.RandomRotation(degrees=5), # Small rotations
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std for normalization
    ])

    data_transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Prepare Training Data
    # Pass TRAIN_ROOT_DIRS list to the dataset
    train_dataset = ColonoscopyTripletDataset(root_dirs=TRAIN_ROOT_DIRS, 
                                              transform=data_transforms_train,
                                              triplet_strategy="sequential_intra_video")
    if len(train_dataset) == 0:
        print(f"Error: Training dataset is empty. Check TRAIN_ROOT_DIRS: {TRAIN_ROOT_DIRS}")
        sys.exit(1) # Exit if training dataset is empty

    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8, # Recommended value to suppress warnings and optimize performance
                              drop_last=True) # Drop the last incomplete batch for Triplet Loss

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
    
    # Ensure 'trained_model' always references the loaded or newly trained model
    trained_model = model

    # 6. Prepare Evaluation Data (Colon10K)
    # Colon10K path and file type correction
    # Ensure TEST_ROOT_DIR points to /home/h8/zhyu410g/Colon10K/10kdata/test/
    test_video_folders = sorted([d for d in glob.glob(os.path.join(TEST_ROOT_DIR, '*')) if os.path.isdir(d)])
    
    all_test_img_paths = []
    all_test_img_labels = [] 

    for folder in test_video_folders:
        video_id = os.path.basename(folder) # Folder name serves as the "relevance label"
        color_frames = sorted(glob.glob(os.path.join(folder, '*.jpg'))) # Search for .jpg files
        if not color_frames:
            print(f"Warning: No .jpg files found in folder: {folder}. Skipping this folder.")
            continue 
        for frame_path in color_frames:
            all_test_img_paths.append(frame_path)
            all_test_img_labels.append(video_id) 
    
    # Critical checkpoint: Check if evaluation dataset is empty
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
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8) # Recommended value

    num_queries = min(100, len(all_test_img_paths))
    if num_queries == 0:
        print("CRITICAL ERROR: num_queries is 0. No queries will be processed for evaluation.")
        sys.exit(1)
        
    query_indices = random.sample(range(len(all_test_img_paths)), num_queries)
    query_paths = [all_test_img_paths[i] for i in query_indices]
    query_labels = [all_test_img_labels[i] for i in query_indices]

    query_dataset = InferenceDataset(query_paths, query_labels, data_transforms_eval)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8) # Recommended value

    # 7. Extract Descriptors or Load Cached Descriptors
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

    # 8. Evaluate Retrieval Performance
    print("Evaluating retrieval performance...")
    # Pass query_paths and all_test_img_paths (as gallery_paths) to evaluate_retrieval for self-match exclusion
    rank1, mAP = evaluate_retrieval(query_descriptors, gallery_descriptors, extracted_query_labels, extracted_gallery_labels,
                                    query_paths=query_paths, gallery_paths=all_test_img_paths) 

    print(f"\n--- Evaluation Results ({AGG_LAYER.upper()} aggregation) ---")
    print(f"RANK-1 (excluding self-match): {rank1:.4f}") 
    print(f"mAP: {mAP:.4f}")

    # TODO: If you want to compare GeM vs. NetVLAD:
    # 1. Implement NetVLAD module or find and integrate an existing PyTorch NetVLAD library.
    # 2. Repeat the training and evaluation process here for NetVLAD.