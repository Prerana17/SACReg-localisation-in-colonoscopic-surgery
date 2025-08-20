import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import random
import numpy as np

# 如果需要读取 .tiff 文件，可能需要
# from tifffile import imread 

class ColonoscopyTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, triplet_strategy="sequential_intra_video"):
        self.root_dir = root_dir
        self.transform = transform
        self.triplet_strategy = triplet_strategy
        
        # 收集所有视频（例如 cecum_t1_a, sigmoid_t2_a 等）
        # 假设每个子文件夹是一个“视频”
        self.video_folders = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        
        self.video_frames = [] # 存储每个视频的所有 (frame_path, video_id)
        self.video_ids = {} # video_folder_name -> unique_id
        
        for i, video_folder in enumerate(self.video_folders):
            video_id = os.path.basename(video_folder)
            self.video_ids[video_id] = i
            
            # 找到当前视频文件夹下的所有颜色图像
            color_frames = sorted(glob.glob(os.path.join(video_folder, '*_color.png')))
            self.video_frames.extend([(frame_path, video_id) for frame_path in color_frames])

        # 用于加速负样本查找的索引
        self.frames_by_video_id = {vid: [f_path for f_path, _ in frames] 
                                   for vid, frames in zip(self.video_ids.keys(), 
                                                           [[] for _ in range(len(self.video_folders))])}
        for f_path, v_id in self.video_frames:
            self.frames_by_video_id[v_id].append(f_path)


    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        anchor_path, anchor_video_id = self.video_frames[idx]
        
        # --- Positive 样本选择 ---
        # 策略1: sequential_intra_video (同一视频内相邻帧)
        if self.triplet_strategy == "sequential_intra_video":
            # 找到同一视频中的所有帧
            current_video_frames = self.frames_by_video_id[anchor_video_id]
            
            # 找到 Anchor 在当前视频中的索引
            anchor_idx_in_video = current_video_frames.index(anchor_path)
            
            # 尝试选择紧随 Anchor 的帧作为 Positive
            positive_path = None
            if anchor_idx_in_video < len(current_video_frames) - 1:
                positive_path = current_video_frames[anchor_idx_in_video + 1]
            else: # 如果 Anchor 是视频的最后一帧，则选择前一帧
                positive_path = current_video_frames[anchor_idx_in_video - 1]
            
            # 如果视频只有一帧 (极不可能), 则 P = A (这会使 loss=0)
            if positive_path is None or positive_path == anchor_path:
                 # Fallback: choose a random frame from the same video
                if len(current_video_frames) > 1:
                    positive_path = random.choice([f for f in current_video_frames if f != anchor_path])
                else:
                    positive_path = anchor_path # Only one frame, P=A

        # 策略2: semantic_similar_intra_video (同一视频内，基于地面实况的相似帧)
        # 这是一个更高级的策略，需要读取深度/法线数据并计算相似度。
        # 示例：你可以加载 anchor_path 对应的深度图，然后遍历 current_video_frames
        # 寻找与 anchor 深度图相似度最高的帧作为 positive。
        # 这里只是一个占位符，实际实现会更复杂。
        elif self.triplet_strategy == "semantic_similar_intra_video":
            # 示例: 加载 Anchor 的深度图
            # anchor_depth_path = anchor_path.replace('_color.png', '_depth.tiff')
            # anchor_depth = imread(anchor_depth_path) # 用 tifffile 读取
            
            # ... 循环 current_video_frames, 计算深度相似度并选择 Positive
            # 为了简洁，这里暂时退回到 sequential
            return self.__getitem__(idx) # 重新尝试获取 (A,P,N)
            # 或者直接报错让用户知道需要实现
            # raise NotImplementedError("Semantic similarity strategy requires custom implementation for positive mining.")
        else:
            raise ValueError(f"Unknown triplet strategy: {self.triplet_strategy}")

        # --- Negative 样本选择 ---
        # 目标: 选择来自不同视频的帧，或者同一视频中语义/视觉上差异很大的帧
        # 这里实现一个简单的：随机选择一个来自不同视频的帧
        
        negative_path = None
        while negative_path is None:
            # 随机选择一个不同的视频 ID
            negative_video_id = random.choice(list(self.video_ids.keys()))
            if negative_video_id == anchor_video_id:
                continue # 确保是不同的视频
            
            # 从这个不同的视频中随机选择一帧
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