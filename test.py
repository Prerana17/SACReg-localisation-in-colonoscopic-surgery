# # #!/usr/bin/env python3
# # """
# # 测试脚本：导入SCRegNet网络并加载权重
# # Test script: Import SCRegNet network and load weights
# # """

# # import torch
# # import sys
# # from pathlib import Path

# # # 添加项目根目录到Python路径
# # project_root = Path(__file__).parent
# # sys.path.insert(0, str(project_root))

# # from screg.model import SCRegNet

# # def test_network_import_and_loading():
# #     """测试网络导入和权重加载"""
# #     print("=== SCRegNet 网络导入和权重加载测试 ===")
    
# #     # 可用的权重文件
# #     weight_dir = project_root / "screg" / "weight"
# #     available_weights = {
# #         "dpt_512": weight_dir / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
# #         "linear_512": weight_dir / "DUSt3R_ViTLarge_BaseDecoder_512_linear.pth", 
# #         "linear_224": weight_dir / "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
# #     }
    
# #     print(f"发现权重文件:")
# #     for name, path in available_weights.items():
# #         exists = "✓" if path.exists() else "✗"
# #         print(f"  {exists} {name}: {path.name}")
    
# #     # 测试不同配置的网络加载
# #     test_configs = [
# #         {"name": "DPT Head (512)", "head_type": "dpt", "weight_key": "dpt_512"},
# #         {"name": "Linear Head (512)", "head_type": "linear", "weight_key": "linear_512"},
# #         {"name": "Linear Head (224)", "head_type": "linear", "weight_key": "linear_224"}
# #     ]
    
# #     for config in test_configs:
# #         print(f"\n--- 测试 {config['name']} ---")
# #         weight_path = available_weights[config["weight_key"]]
        
# #         if not weight_path.exists():
# #             print(f"权重文件不存在: {weight_path}")
# #             continue
            
# #         try:
# #             # 使用类方法加载预训练模型
# #             model = SCRegNet.load_from_dust3r_weight(
# #                 ckpt_path=str(weight_path),
# #                 head_type=config["head_type"],
# #                 output_mode="pts3d",
# #                 map_location="cpu",
# #                 strict=False,
# #                 show_mismatch=True
# #             )
            
# #             print(f"✓ 成功加载模型")
# #             print(f"  - 头部类型: {config['head_type']}")
# #             print(f"  - 输出模式: pts3d")
# #             print(f"  - 参数总数: {sum(p.numel() for p in model.parameters()):,}")
# #             print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
# #             # 测试模型结构
# #             model.eval()
# #             print(f"  - 模型设置为评估模式")
            
# #         except Exception as e:
# #             print(f"✗ 加载失败: {e}")

# # def test_manual_loading():
# #     """测试手动创建模型并加载权重"""
# #     print(f"\n=== 手动创建模型并加载权重测试 ===")
    
# #     try:
# #         # 手动创建模型
# #         model = SCRegNet(
# #             head_type="dpt",
# #             output_mode="pts3d", 
# #             has_conf=True,
# #             img_size=(512, 512)
# #         )
# #         print("✓ 成功创建空模型")
        
# #         # 加载权重
# #         weight_path = project_root / "screg" / "weight" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# #         if weight_path.exists():
# #             missing, unexpected = model.load_weights(str(weight_path), strict=False)
# #             print(f"✓ 权重加载完成")
# #             print(f"  - 缺失参数: {len(missing)}")
# #             print(f"  - 意外参数: {len(unexpected)}")
# #         else:
# #             print(f"✗ 权重文件不存在: {weight_path}")
            
# #     except Exception as e:
# #         print(f"✗ 手动加载失败: {e}")

# # def test_model_inference_shape():
# #     """测试模型推理形状"""
# #     print(f"\n=== 模型推理形状测试 ===")
    
# #     try:
# #         # 加载一个模型进行形状测试
# #         weight_path = project_root / "screg" / "weight" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# #         if not weight_path.exists():
# #             print("跳过推理测试 - 权重文件不存在")
# #             return
            
# #         model = SCRegNet.load_from_dust3r_weight(
# #             ckpt_path=str(weight_path),
# #             head_type="dpt",
# #             map_location="cpu",
# #             strict=False,
# #             show_mismatch=False
# #         )
# #         model.eval()
        
# #         # 创建测试输入
# #         B, H, W = 1, 512, 512
# #         query_img = torch.randn(B, 3, H, W)
# #         retrieved_img = torch.randn(B, 3, H, W) 
# #         correspondences = torch.randn(B, 100, 5)  # 100个对应点 (u,v,x,y,z)
        
# #         print(f"输入形状:")
# #         print(f"  - query_img: {query_img.shape}")
# #         print(f"  - retrieved_img: {retrieved_img.shape}")
# #         print(f"  - correspondences: {correspondences.shape}")
        
# #         # 前向推理
# #         with torch.no_grad():
# #             output = model(query_img, retrieved_img, correspondences)
            
# #         print(f"✓ 推理成功")
# #         print(f"输出形状:")
# #         for key, value in output.items():
# #             if isinstance(value, torch.Tensor):
# #                 print(f"  - {key}: {value.shape}")
# #             else:
# #                 print(f"  - {key}: {type(value)}")
                
# #     except Exception as e:
# #         print(f"✗ 推理测试失败: {e}")

# # if __name__ == "__main__":
# #     test_network_import_and_loading()
# #     test_manual_loading() 
# #     test_model_inference_shape()
# #     print(f"\n=== 测试完成 ===")

# # from screg.model import SCRegNet

# # model = SCRegNet.load_from_dust3r_weight(
# #     ckpt_path="/home/tim/screg/screg/weight/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
# #     head_type="dpt",
# #     map_location="cuda",
# #     strict=False,
# #     show_mismatch=True
# # )
# # model.eval()

# # print(model)

# # if __name__ == "__main__":
# #     from pathlib import Path
# #     import imageio.v2 as imageio
# #     import numpy as np

# #     processed_root = Path(__file__).resolve().parent / "data" / "processed" / "SimCol3D"
# #     depth_files = list(processed_root.rglob("*.depth.png"))
# #     if not depth_files:
# #         print("No depth files found under processed directory.")
# #     else:
# #         sample = depth_files[0]
# #         img = imageio.imread(sample)
# #         print(f"Sample depth file: {sample}")
# #         print(f"dtype: {img.dtype}, raw min: {np.min(img)}, raw max: {np.max(img)}")

# #         # Convert to metres (0–0.20 m) as in preprocessing
# #         if img.dtype == np.uint16:
# #             depth_norm = img.astype(np.float32) / 65535.0
# #         else:
# #             depth_norm = img.astype(np.float32) / 255.0
# #         depth_m = depth_norm * 0.20
# #         print(f"depth_m range: min {depth_m.min():.6f} m, max {depth_m.max():.6f} m")

# #         # ---------- copy of _depth_to_cam logic from preprocess_SimCol3D.py ----------
# #         h, w = depth_m.shape
# #         ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
# #         zs = depth_m
# #         # 由于这里只测试范围，摄像机内参用像素中心近似即可。
# #         fx = fy = 1.0
# #         cx = cy = 0.0
# #         xs_cam = (xs - cx) / fx * zs
# #         ys_cam = (ys - cy) / fy * zs
# #         xyz_cam = np.stack([xs_cam, ys_cam, zs], axis=2)
# #         print(f"xyz_cam z-range: {np.nanmin(zs):.6f}–{np.nanmax(zs):.6f} m")

# #         # Save depth_m as npy for inspection
# #         depth_npy_path = sample.with_suffix('.depth_m.npy')
# #         np.save(depth_npy_path, depth_m)
# #         print(f"Saved depth_m to {depth_npy_path}")


# from dreamsim import dreamsim
# from PIL import Image

# device = "cuda"
# dreamsim_dino_model, preprocess = dreamsim(pretrained=True, dreamsim_type="dino_vitb16")

# # img1 = preprocess(Image.open("img1_path")).to(device)
# # img2 = preprocess(Image.open("img2_path")).to(device)
# # distance = dreamsim_dino_model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224

# img1 = preprocess(Image.open("/home/tim/screg/data/processed/SimCol3D/SyntheticColon_I/database/Frames_S1/0000.rgb.png")).to("cuda")
# embedding = dreamsim_dino_model.embed(img1)
# print(embedding.shape)


# -----------------------------------------------------------------------------
# SCRegNet + SimCol3D dataloader quick shape test
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader
#     from screg.datasets import SimCol3DDataset, simcol3d_collate_fn
#     from screg.model import SCRegNet

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Small dataset sample for quick test
#     ds = SimCol3DDataset(top_k=50, n_samples=32)
#     loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=simcol3d_collate_fn)

#     # model = SCRegNet(head_type="dpt", output_mode="pts3d", has_conf=True, img_size=(512, 512), dec_depth=12)
#     # model.to(device)
#     # model.eval()
#     model = SCRegNet.load_from_dust3r_weight(
#         ckpt_path="/home/tim/screg/screg/weight/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
#         head_type="dpt",
#         output_mode="pts3d",
#         # has_conf=True,
#         show_mismatch=True)
#     model.to(device)
#     model.eval()
#     batch = next(iter(loader))
#     q_img = batch["query_img"].to(device)
#     db_img = batch["db_img"].to(device)
#     uvxyz = batch["uvxyz"].to(device)

#     with torch.no_grad():
#         out = model(q_img, db_img, uvxyz)
#     for k, v in out.items():
#         if torch.is_tensor(v):
#             print(f"{k}: {tuple(v.shape)}")
#         else:
#             print(f"{k}: {type(v)}")



# import torch, pprint, os, json, sys
# ckpt_path = '/home/tim/screg/screg/weight/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
# if not os.path.exists(ckpt_path):
#     print('Checkpoint not found:', ckpt_path)
#     sys.exit(1)
# ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
# print('Checkpoint keys:', ckpt.keys())
# print('\nargs:')
# pprint.pprint(ckpt.get('args'))


# import numpy as np, pathlib, sys
# p=pathlib.Path('data/processed/SimCol3D/SyntheticColon_I/database/Frames_S10/0000.xyz.npy')
# xyz=np.load(p)
# print('shape', xyz.shape, 'dtype', xyz.dtype)

import numpy as np, glob, os, random, sys
file=glob.glob('data/processed/SimCol3D/SyntheticColon_I/database/Frames_S10/*.xyz.npy')[0]
xyz=np.load(file, mmap_mode='r')
pts=xyz.reshape(-1,3)
pts=pts[~np.isnan(pts).any(axis=1)]
print('min', pts.min(axis=0))
print('max', pts.max(axis=0))