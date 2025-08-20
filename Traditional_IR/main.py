import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import glob
import random
import numpy as np
from PIL import Image # For InferenceDataset, although it's used within a function
from sklearn.metrics.pairwise import cosine_similarity # For evaluation

# 导入你自己的模块
from dataset import ColonoscopyTripletDataset, InferenceDataset # 从 dataset.py 导入这两个类
from model import IRModel, GeM # 从 model.py 导入 IRModel 和 GeM (如果GeM是外部类)
from losses import TripletLoss # 从 losses.py 导入 TripletLoss

# 导入你定义的训练和评估函数
from train import train_model
from evaluate import get_descriptors, evaluate_retrieval


if __name__ == "__main__":
    # 1. 配置参数
    TRAIN_ROOT_DIR = '/data/horse/ws/zhyu410g-horse_C3VD_data/exported/' # 你的 C3VD 和 SimCol3D 数据集根目录
    TEST_ROOT_DIR = '~/Colon10K/10kdata'     # 你的 Colon10K 数据集根目录

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    MARGIN = 1.0 # Triplet Loss 的 margin
    BACKBONE_NAME = 'resnet50' # 或 'resnet101'
    AGG_LAYER = 'gem' # 或 'netvlad' (需要额外实现)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 数据转换 (Data Augmentation)
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 输入尺寸
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=0.8), # 随机亮度、对比度等
        transforms.RandomRotation(degrees=5), # 小范围旋转
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 均值和标准差
    ])

    data_transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 准备训练数据
    # Triplet strategy 可以根据需求调整
    # 例如：'sequential_intra_video', 'semantic_similar_intra_video' (需要你自己实现)
    train_dataset = ColonoscopyTripletDataset(root_dir=TRAIN_ROOT_DIR, 
                                              transform=data_transforms_train,
                                              triplet_strategy="sequential_intra_video")
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=os.cpu_count() // 2, # 通常设置为CPU核心数的一半
                              drop_last=True) # Triplet Loss 建议 drop_last

    # 4. 初始化模型、损失函数和优化器
    model = IRModel(backbone_name=BACKBONE_NAME, agg_layer=AGG_LAYER)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练模型
    print("Starting model training...")
    trained_model = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)

    # 保存模型
    model_save_path = f"ir_model_{BACKBONE_NAME}_{AGG_LAYER}.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # 6. 准备评估数据 (Colon10K)
    # 对于 Colon10K，你需要确定其地面实况标签。
    # 最常见的是：如果查询是某个病例的帧，那么该病例的所有其他帧都是相关的。
    # 你需要编写代码遍历 Colon10K 目录，提取所有图像路径和它们的“相关性ID”（例如，从文件夹名或文件名解析出病例ID/视频ID）。
    
    # 示例：假设 Colon10K 结构类似 C3VD，每个子文件夹是一个病例/视频
    test_video_folders = sorted([d for d in glob.glob(os.path.join(TEST_ROOT_DIR, '*')) if os.path.isdir(d)])
    
    all_test_img_paths = []
    all_test_img_labels = [] # 例如，存储视频ID/病例ID

    for folder in test_video_folders:
        video_id = os.path.basename(folder)
        color_frames = sorted(glob.glob(os.path.join(folder, '*_color.png'))) # 假设 Colon10K 也有 _color.png
        for frame_path in color_frames:
            all_test_img_paths.append(frame_path)
            all_test_img_labels.append(video_id) # 使用视频ID作为相关性标签
    
    # 将整个测试集作为画廊 (gallery)
    gallery_dataset = InferenceDataset(all_test_img_paths, all_test_img_labels, data_transforms_eval)
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=os.cpu_count() // 2)

    # 选择一部分帧作为查询 (queries)
    # 通常查询集是画廊集的子集，或来自不同源但预期能检索出画廊中的相关项
    # 这里为了简化，我们从画廊中随机选择100帧作为查询
    num_queries = min(100, len(all_test_img_paths))
    query_indices = random.sample(range(len(all_test_img_paths)), num_queries)
    query_paths = [all_test_img_paths[i] for i in query_indices]
    query_labels = [all_test_img_labels[i] for i in query_indices]

    query_dataset = InferenceDataset(query_paths, query_labels, data_transforms_eval)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=os.cpu_count() // 2)


    # 7. 提取描述符
    print("Extracting query descriptors...")
    query_descriptors, extracted_query_labels = get_descriptors(trained_model, query_loader, DEVICE)
    print("Extracting gallery descriptors...")
    gallery_descriptors, extracted_gallery_labels = get_descriptors(trained_model, gallery_loader, DEVICE)

    # 8. 评估检索性能
    print("Evaluating retrieval performance...")
    rank1, mAP = evaluate_retrieval(query_descriptors, gallery_descriptors, extracted_query_labels, extracted_gallery_labels)

    print(f"\n--- Evaluation Results ({AGG_LAYER.upper()} aggregation) ---")
    print(f"RANK-1: {rank1:.4f}")
    print(f"mAP: {mAP:.4f}")