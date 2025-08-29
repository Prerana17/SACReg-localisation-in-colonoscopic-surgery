"""统一的工具函数
用于确保φ38编码在不同模块中的一致性
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Union


def create_normalized_uv_grid_torch(H: int, W: int, 
                                   device: torch.device | str | None = None, 
                                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """创建归一化的UV网格 (PyTorch版本)
    
    Parameters
    ----------
    H : int
        图像高度
    W : int  
        图像宽度
    device : torch.device | str | None
        设备
    dtype : torch.dtype
        数据类型
        
    Returns
    -------
    torch.Tensor
        形状为 (2, H, W) 的UV网格，范围 [-1, 1]
        第0通道是u坐标，第1通道是v坐标
    """
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    
    # 归一化到 [-1, 1] 范围
    if W > 1:
        u_norm = 2.0 * u / (W - 1) - 1.0
    else:
        u_norm = torch.zeros_like(u)
        
    if H > 1:
        v_norm = 2.0 * v / (H - 1) - 1.0  
    else:
        v_norm = torch.zeros_like(v)
    
    # 创建网格 (注意indexing="ij"确保v是行，u是列)
    v_grid, u_grid = torch.meshgrid(v_norm, u_norm, indexing="ij")
    
    # 堆叠为 (2, H, W): [u_grid, v_grid]
    return torch.stack([u_grid, v_grid], dim=0)


def create_normalized_uv_grid_numpy(H: int, W: int) -> np.ndarray:
    """创建归一化的UV网格 (NumPy版本)
    
    Parameters
    ----------
    H : int
        图像高度
    W : int
        图像宽度
        
    Returns
    -------
    np.ndarray
        形状为 (2, H, W) 的UV网格，范围 [-1, 1]  
        第0通道是u坐标，第1通道是v坐标
    """
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    
    # 归一化到 [-1, 1] 范围
    if W > 1:
        u_norm = 2.0 * u / (W - 1) - 1.0
    else:
        u_norm = np.zeros_like(u)
        
    if H > 1:
        v_norm = 2.0 * v / (H - 1) - 1.0
    else:
        v_norm = np.zeros_like(v)
    
    # 创建网格
    u_grid, v_grid = np.meshgrid(u_norm, v_norm, indexing='xy')
    
    # 堆叠为 (2, H, W): [u_grid, v_grid]
    return np.stack([u_grid, v_grid], axis=0).astype(np.float32)


def compute_phi_frequencies(f1: float = 31.4159, 
                           gamma: float = 2.884031503126606, 
                           F: int = 6) -> np.ndarray:
    """计算φ编码的谐波频率
    
    Parameters
    ----------
    f1 : float
        基础频率
    gamma : float
        频率增长因子
    F : int
        频率数量
        
    Returns
    -------
    np.ndarray
        形状为 (F,) 的频率数组
    """
    return np.array([f1 * (gamma ** i) for i in range(F)], dtype=np.float32)


def phi_encode_xyz_torch(xyz: torch.Tensor, 
                        frequencies: torch.Tensor) -> torch.Tensor:
    """对XYZ坐标进行φ编码 (PyTorch版本)
    
    Parameters
    ----------
    xyz : torch.Tensor
        形状任意，最后一维是3 (xyz坐标)
    frequencies : torch.Tensor
        形状为 (F,) 的频率张量
        
    Returns
    -------
    torch.Tensor
        φ编码结果，形状为 xyz.shape[:-1] + (6*F,)
    """
    original_shape = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)  # (N, 3)
    
    # 展开为 (N, 3, F)
    xyz_expanded = xyz_flat.unsqueeze(-1)  # (N, 3, 1)
    freq_expanded = frequencies.view(1, 1, -1)  # (1, 1, F)
    
    # 计算角度
    angles = xyz_expanded * freq_expanded  # (N, 3, F)
    
    # 计算cos和sin
    cos_part = torch.cos(angles)  # (N, 3, F)
    sin_part = torch.sin(angles)  # (N, 3, F)
    
    # 拼接并重塑
    encoded = torch.cat([cos_part, sin_part], dim=-1)  # (N, 3, 2F)
    encoded = encoded.reshape(-1, 6 * len(frequencies))  # (N, 6F)
    
    # 恢复原始形状
    target_shape = original_shape[:-1] + (6 * len(frequencies),)
    return encoded.reshape(target_shape)


def phi_encode_xyz_numpy(xyz: np.ndarray, 
                        frequencies: np.ndarray) -> np.ndarray:
    """对XYZ坐标进行φ编码 (NumPy版本)
    
    Parameters
    ----------
    xyz : np.ndarray
        形状任意，最后一维是3 (xyz坐标)
    frequencies : np.ndarray
        形状为 (F,) 的频率数组
        
    Returns
    -------
    np.ndarray
        φ编码结果，形状为 xyz.shape[:-1] + (6*F,)
    """
    original_shape = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)  # (N, 3)
    
    # 展开为 (N, 3, F)
    xyz_expanded = xyz_flat[:, :, None]  # (N, 3, 1)
    freq_expanded = frequencies[None, None, :]  # (1, 1, F)
    
    # 计算角度
    angles = xyz_expanded * freq_expanded  # (N, 3, F)
    
    # 计算cos和sin
    cos_part = np.cos(angles)  # (N, 3, F)
    sin_part = np.sin(angles)  # (N, 3, F)
    
    # 拼接并重塑
    encoded = np.concatenate([cos_part, sin_part], axis=-1)  # (N, 3, 2F)
    encoded = encoded.reshape(-1, 6 * len(frequencies))  # (N, 6F)
    
    # 恢复原始形状
    target_shape = original_shape[:-1] + (6 * len(frequencies),)
    return encoded.reshape(target_shape).astype(np.float32)
