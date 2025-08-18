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
import pickle # 新增: 用于保存/加载Python对象，如列表、字典等
import sys # 新增: 用于更早退出

# from tifffile import imread # 如果你需要读取 .tiff 文件 (深度图, 法线图等) 来实现更复杂的 Triplet Mining 策略，请取消注释并安装 tifffile 库

# --- 1. 数据加载和三元组生成 (从 dataset.py 整合) ---

class ColonoscopyTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, triplet_strategy="sequential_intra_video"):
        self.root_dir = root_dir
        self.transform = transform
        self.triplet_strategy = triplet_strategy
        
        self.video_folders = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        
        self.video_frames = [] 
        self.video_ids = {} 
        
        for i, video_folder in enumerate(self.video_folders):
            video_id = os.path.basename(video_folder)
            self.video_ids[video_id] = i
            
            color_frames = sorted(glob.glob(os.path.join(video_folder, '*_color.png')))
            if not color_frames: # 检查是否有 .png 文件，C3VD通常是png
                color_frames = sorted(glob.glob(os.path.join(video_folder, '*.png'))) # 尝试匹配所有.png
            if not color_frames: # 如果还没找到，可能是其他格式或文件夹为空
                print(f"Warning: No _color.png or other .png files found in folder: {video_folder}. Skipping.")
                continue # 跳过空文件夹
            self.video_frames.extend([(frame_path, video_id) for frame_path in color_frames])

        self.frames_by_video_id = {vid: [] for vid in self.video_ids.keys()}
        for f_path, v_id in self.video_frames:
            self.frames_by_video_id[v_id].append(f_path)


    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        anchor_path, anchor_video_id = self.video_frames[idx]
        
        # --- Positive 样本选择 ---
        if self.triplet_strategy == "sequential_intra_video":
            current_video_frames = self.frames_by_video_id[anchor_video_id]
            
            anchor_idx_in_video = current_video_frames.index(anchor_path)
            
            positive_path = None
            if len(current_video_frames) > 1:
                if anchor_idx_in_video < len(current_video_frames) - 1:
                    positive_path = current_video_frames[anchor_idx_in_video + 1]
                else:
                    positive_path = current_video_frames[anchor_idx_in_video - 1]
            else: 
                positive_path = anchor_path

        elif self.triplet_strategy == "semantic_similar_intra_video":
            raise NotImplementedError("Semantic similarity strategy requires custom implementation for positive mining.")
        else:
            raise ValueError(f"Unknown triplet strategy: {self.triplet_strategy}")

        # --- Negative 样本选择 ---
        negative_path = None
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

# 评估时使用的 Dataset，只加载单张图片 (从 dataset.py 整合)
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

# --- 2. 模型定义 (从 model.py 整合) ---

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
        
        # 1. Backbone
        if backbone_name == 'resnet50':
            # 推荐使用 weights=models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")
        
        self.features = nn.Sequential(*list(backbone.children())[:-2]) 
        
        # 2. Aggregation Layer
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

# --- 3. 损失函数 (从 losses.py 整合) ---

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# --- 4. 训练脚本 (从 train.py 整合) ---

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

# --- 5. 评估脚本 (从 evaluate.py 整合) ---

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
            # Convert labels to numpy array directly if they are tensors
            labels.extend(lbls.cpu().numpy().tolist() if isinstance(lbls, torch.Tensor) else lbls)
            
    if not descriptors: # 添加检查，避免vstack空列表
        print("Warning: No descriptors were extracted. DataLoader might be empty.")
        return np.array([]), np.array([]) # 返回空数组
        
    return np.vstack(descriptors), np.array(labels)

def evaluate_retrieval(query_descriptors, gallery_descriptors, query_labels, gallery_labels):
    if query_descriptors.size == 0 or gallery_descriptors.size == 0:
        print("Warning: Query or gallery descriptors are empty. Cannot perform evaluation.")
        return 0.0, 0.0

    similarities = cosine_similarity(query_descriptors, gallery_descriptors)
    
    rank1_count = 0
    aps = [] 

    for i in range(len(query_labels)):
        query_label = query_labels[i]
        
        relevant_gallery_indices = [j for j, label in enumerate(gallery_labels) if label == query_label]
        
        if not relevant_gallery_indices: # 如果当前查询没有相关项，则跳过
            aps.append(0.0)
            continue

        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # === 计算 RANK-1 ===
        # 找到第一个相关项的索引 (在排序后的列表中)
        for k, gallery_idx in enumerate(sorted_indices):
            if gallery_labels[gallery_idx] == query_label: 
                rank1_count += 1
                break 

        # === 计算 Average Precision (AP) ===
        num_relevant_found = 0
        sum_precisions = 0
        
        for k, gallery_idx in enumerate(sorted_indices):
            if gallery_labels[gallery_idx] == query_label:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / (k + 1)
                sum_precisions += precision_at_k
            
            # 如果所有相关项都已找到，可以提前退出循环以优化
            if num_relevant_found == len(relevant_gallery_indices):
                break

        ap = sum_precisions / len(relevant_gallery_indices)
        aps.append(ap)

    rank1 = rank1_count / len(query_labels)
    mAP = np.mean(aps) if len(aps) > 0 else 0.0

    return rank1, mAP

# --- 主运行逻辑 (从 main.py 整合) ---

if __name__ == "__main__":
    # --- 0. 设置保存/加载路径 ---
    MODEL_CHECKPOINT_DIR = './checkpoints' # 保存模型的目录
    DESCRIPTORS_CACHE_DIR = './descriptors_cache' # 保存描述符的目录
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DESCRIPTORS_CACHE_DIR, exist_ok=True)

    # 1. 配置参数
    TRAIN_ROOT_DIR = '/data/horse/ws/zhyu410g-horse_C3VD_data/exported/' # 你的 C3VD 和 SimCol3D 数据集根目录
    TEST_ROOT_DIR = os.path.expanduser('~/Colon10K/10kdata/test')    # 你的 Colon10K 数据集根目录

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    MARGIN = 1.0 # Triplet Loss 的 margin
    BACKBONE_NAME = 'resnet50' # 或 'resnet101'
    AGG_LAYER = 'gem' # 或 'netvlad' (需要额外实现)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型和描述符保存的文件名
    model_save_path = os.path.join(MODEL_CHECKPOINT_DIR, f"ir_model_{BACKBONE_NAME}_{AGG_LAYER}_ep{NUM_EPOCHS}.pth")
    query_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_desc_{BACKBONE_NAME}_{AGG_LAYER}.npy")
    query_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"query_labels_{BACKBONE_NAME}_{AGG_LAYER}.pkl")
    gallery_desc_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_desc_{BACKBONE_NAME}_{AGG_LAYER}.npy")
    gallery_labels_path = os.path.join(DESCRIPTORS_CACHE_DIR, f"gallery_labels_{BACKBONE_NAME}_{AGG_LAYER}.pkl")


    # 2. 数据转换 (Data Augmentation)
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) # 移除了 hue 参数
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
    train_dataset = ColonoscopyTripletDataset(root_dir=TRAIN_ROOT_DIR, 
                                              transform=data_transforms_train,
                                              triplet_strategy="sequential_intra_video")
    if len(train_dataset) == 0:
        print(f"Error: Training dataset is empty. Check TRAIN_ROOT_DIR: {TRAIN_ROOT_DIR}")
        sys.exit(1) # 如果训练集为空，则退出

    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=8, # 建议值，以消除警告
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
    
    # 确保无论是否加载，模型的引用都是 model
    trained_model = model

    # 6. 准备评估数据 (Colon10K)
    # 你的 Colon10K 路径和文件类型修正
    # 确保 TEST_ROOT_DIR 指向 /home/h8/zhyu410g/Colon10K/10kdata/test/
    test_video_folders = sorted([d for d in glob.glob(os.path.join(TEST_ROOT_DIR, '*')) if os.path.isdir(d)])
    
    all_test_img_paths = []
    all_test_img_labels = [] 

    for folder in test_video_folders:
        video_id = os.path.basename(folder)
        color_frames = sorted(glob.glob(os.path.join(folder, '*.jpg'))) # 查找 .jpg 文件
        if not color_frames:
            print(f"Warning: No .jpg files found in folder: {folder}. Skipping this folder.")
            continue 
        for frame_path in color_frames:
            all_test_img_paths.append(frame_path)
            all_test_img_labels.append(video_id) 
    
    # 关键检查点：评估数据集是否为空
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
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8) # 建议值

    num_queries = min(100, len(all_test_img_paths))
    if num_queries == 0:
        print("CRITICAL ERROR: num_queries is 0. No queries will be processed for evaluation.")
        sys.exit(1)
        
    query_indices = random.sample(range(len(all_test_img_paths)), num_queries)
    query_paths = [all_test_img_paths[i] for i in query_indices]
    query_labels = [all_test_img_labels[i] for i in query_indices]

    query_dataset = InferenceDataset(query_paths, query_labels, data_transforms_eval)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=8) # 建议值

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
        
        # 确保目录存在才能保存
        np.save(query_desc_path, query_descriptors)
        np.save(gallery_desc_path, gallery_descriptors)
        with open(query_labels_path, 'wb') as f:
            pickle.dump(extracted_query_labels, f)
        with open(gallery_labels_path, 'wb') as f:
            pickle.dump(extracted_gallery_labels, f)
        print("Descriptors extracted and cached successfully.")

    # 8. 评估检索性能
    print("Evaluating retrieval performance...")
    rank1, mAP = evaluate_retrieval(query_descriptors, gallery_descriptors, extracted_query_labels, extracted_gallery_labels)

    print(f"\n--- Evaluation Results ({AGG_LAYER.upper()} aggregation) ---")
    print(f"RANK-1: {rank1:.4f}")
    print(f"mAP: {mAP:.4f}")

    # TODO: 如果你想比较 GeM 和 NetVLAD，你需要：
    # 1. 实现 NetVLAD 模块或找到并集成一个现有的 PyTorch NetVLAD 库
    # 2. 在这里为 NetVLAD 重复训练和评估流程。