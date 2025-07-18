import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download
import os

# 1. 图像预处理函数
def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228], dataset_std=[0.2520, 0.2128, 0.2093]):
    """
    处理单张图像，进行大小调整、转换为Tensor和归一化。
    Args:
        image_path (str or Path): 图像文件的路径。
        input_size (int): 模型期望的输入图像大小（例如 224）。
        dataset_mean (list): 用于归一化的数据集均值。
        dataset_std (list): 用于归一化的数据集标准差。
    Returns:
        torch.Tensor: 处理后的图像Tensor。
    """
    # 定义图像转换
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    # 应用转换
    processed_image = transform(image)
    return processed_image

# 2. 从 Hugging Face 加载模型函数
def load_model_from_huggingface(repo_id, model_filename):
    """
    从 Hugging Face Hub 下载并加载 EndoViT 模型权重。
    Args:
        repo_id (str): Hugging Face 仓库ID (例如 "egeozsoy/EndoViT")。
        model_filename (str): 模型权重文件的名称 (例如 "pytorch_model.bin")。
    Returns:
        tuple: (加载后的模型实例, 加载信息)。
    """
    # 下载模型文件到本地缓存
    # revision="main" 表示下载主分支
    model_path = snapshot_download(repo_id=repo_id, revision="main")
    model_weights_path = Path(model_path) / model_filename
    
    # 确保模型权重文件存在
    if not model_weights_path.exists():
        raise FileNotFoundError(f"模型权重文件未找到: {model_weights_path}. 请检查 repo_id 和 model_filename。")

    # 加载模型权重
    # 'model' 键是 EndoViT 权重文件中的实际模型状态字典
    model_weights = torch.load(model_weights_path)['model']
    
    # 定义模型架构
    # 这必须与 EndoViT 的 ViT-Base 架构相匹配
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).eval() # 设置为评估模式
    
    # 加载权重到模型中，strict=False 表示如果模型和权重文件中的键不完全匹配，也允许加载
    loading = model.load_state_dict(model_weights, strict=False)
    return model, loading

if __name__ == '__main__':
    # 示例用法
    
    # 设定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设定数据类型，通常在推理时使用 float16 可以加速
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Hugging Face 仓库ID和模型文件名
    repo_id = "egeozsoy/EndoViT"
    model_filename = "pytorch_model.bin"

    # 加载 EndoViT 模型
    model, loading_info = load_model_from_huggingface(repo_id, model_filename)
    model = model.to(device, dtype) # 将模型移动到指定设备和数据类型
    print("模型加载信息:", loading_info)

    # 假设您的图像文件在 'demo_images' 文件夹中
    # 请根据您的实际图像路径进行调整
    # 例如，如果您想处理 cecum_t1_a 文件夹中的所有 _color.png 图像：
    # image_dir = Path('/data/horse/ws/zhyu410g-horse_C3VD_data/cecum_t1_a/')
    # image_paths = sorted(image_dir.glob('*_color.png'))
    
    # 这里使用一个占位符路径，您需要替换为实际的图像路径
    # 确保 'demo_images' 文件夹存在并包含一些 .png 图像用于测试
    demo_image_dir = Path('demo_images') 
    if not demo_image_dir.exists():
        demo_image_dir.mkdir(parents=True, exist_ok=True)
        # 创建一些虚拟图像用于测试，如果实际图像尚未准备好
        print(f"创建虚拟图像用于测试: {demo_image_dir}")
        dummy_image = Image.new('RGB', (224, 224), color = 'red')
        dummy_image.save(demo_image_dir / 'dummy_000.png')
        dummy_image.save(demo_image_dir / 'dummy_001.png')
        
    image_paths = sorted(demo_image_dir.glob('*.png')) 
    if not image_paths:
        print(f"警告: 在 {demo_image_dir} 中没有找到任何图像。请确保路径正确或创建测试图像。")
        exit()

    # 批量处理图像
    images_tensor = torch.stack([process_single_image(image_path) for image_path in image_paths])
    print(f"处理后的图像 Tensor 形状: {images_tensor.shape}")

    # 提取特征
    with torch.no_grad(): # 在推理时关闭梯度计算，节省内存和计算
        # EndoViT (基于 ViT) 的 forward_features 方法通常返回 [CLS] token 的特征
        output_features = model.forward_features(images_tensor.to(device, dtype))
    
    print(f"提取的特征形状: {output_features.shape}")
    # output_features 的形状通常是 (num_images, embedding_dim)
    # 对于 ViT，第一个 token (CLS token) 的特征通常用于表示整个图像
    # 如果模型输出是 (batch_size, num_patches + 1, embedding_dim)，则需要取第一个 token
    # 例如：output_features = output_features[:, 0] 
    # EndoViT 的 forward_features 通常已经处理了这一点，直接返回图像特征

    print("\n特征提取完成！您可以将这些特征用于图像检索任务。")
