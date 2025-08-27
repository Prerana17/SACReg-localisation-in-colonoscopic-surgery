#!/usr/bin/env python3
"""
三维交互式可视化工具
用于显示相机视锥和点云数据
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
import argparse


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: [w, x, y, z]
    """
    w, x, y, z = q
    
    # 归一化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def create_camera_frustum(position, rotation_matrix, scale=0.05, color='red'):
    """
    创建相机视锥
    position: 相机位置 [x, y, z] (米)
    rotation_matrix: 3x3旋转矩阵
    scale: 视锥大小缩放因子
    color: 视锥颜色
    """
    # 定义相机坐标系下的视锥顶点（简化的金字塔形状）
    # 相机朝向-Z方向
    frustum_points = np.array([
        [0, 0, 0],           # 相机中心
        [-scale, -scale, -2*scale],  # 左下
        [scale, -scale, -2*scale],   # 右下
        [scale, scale, -2*scale],    # 右上
        [-scale, scale, -2*scale],   # 左上
    ])
    
    # 将视锥点转换到世界坐标系
    world_points = []
    for point in frustum_points:
        world_point = rotation_matrix @ point + position
        world_points.append(world_point)
    
    world_points = np.array(world_points)
    
    # 定义视锥的边
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从相机中心到四个角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 四个角之间的连线
    ]
    
    # 创建线条
    lines_x, lines_y, lines_z = [], [], []
    for edge in edges:
        start, end = edge
        lines_x.extend([world_points[start][0], world_points[end][0], None])
        lines_y.extend([world_points[start][1], world_points[end][1], None])
        lines_z.extend([world_points[start][2], world_points[end][2], None])
    
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color=color, width=3),
        name=f'Camera Frustum',
        showlegend=True
    )


def load_pose_data(pose_file):
    """
    加载pose数据
    返回位置(米)和旋转矩阵
    """
    with open(pose_file, 'r') as f:
        data = f.read().strip().split()
    
    # 位置：厘米转米
    position = np.array([float(data[0]), float(data[1]), float(data[2])]) / 100.0
    
    # 四元数 [w, x, y, z]
    quaternion = np.array([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    
    # 转换为旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    
    return position, rotation_matrix


def load_pointcloud_data(xyz_file, downsample_factor=10):
    """
    加载点云数据
    downsample_factor: 下采样因子，减少点云密度以提高性能
    """
    xyz = np.load(xyz_file)
    
    # 下采样
    if downsample_factor > 1:
        xyz = xyz[::downsample_factor, ::downsample_factor, :]
    
    # 重塑为N×3的点云
    points = xyz.reshape(-1, 3)
    
    # 过滤掉无效点（全零或异常值）
    valid_mask = ~np.all(points == 0, axis=1)
    points = points[valid_mask]
    
    return points


def create_visualization(data_dir, frame_indices, downsample_factor=10):
    """
    创建3D可视化
    """
    fig = go.Figure()
    
    # 为每个帧分配不同颜色
    colors = px.colors.qualitative.Set1[:len(frame_indices)]
    
    for i, frame_idx in enumerate(frame_indices):
        frame_str = f"{frame_idx:04d}"
        pose_file = os.path.join(data_dir, f"{frame_str}.pose.txt")
        xyz_file = os.path.join(data_dir, f"{frame_str}.xyz.npy")
        
        if not os.path.exists(pose_file) or not os.path.exists(xyz_file):
            print(f"Warning: Files for frame {frame_str} not found")
            continue
        
        color = colors[i % len(colors)]
        
        # 加载相机pose
        position, rotation_matrix = load_pose_data(pose_file)
        
        # 创建相机视锥
        frustum = create_camera_frustum(position, rotation_matrix, scale=0.02, color=color)
        frustum.name = f'Camera {frame_str}'
        fig.add_trace(frustum)
        
        # 加载点云
        points = load_pointcloud_data(xyz_file, downsample_factor)
        
        if len(points) > 0:
            # 添加点云
            pointcloud = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1], 
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=color,
                    opacity=0.3  # 透明度
                ),
                name=f'PointCloud {frame_str}',
                showlegend=True
            )
            fig.add_trace(pointcloud)
    
    # 设置布局
    fig.update_layout(
        title="相机视锥和点云可视化",
        scene=dict(
            xaxis_title="X (米)",
            yaxis_title="Y (米)",
            zaxis_title="Z (米)",
            aspectmode='data',  # 保持真实比例
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        showlegend=True
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='3D Camera and PointCloud Visualization')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/tim/screg/data/processed/SimCol3D/SyntheticColon_I/database/Frames_S5',
                       help='Directory containing pose.txt and xyz.npy files')
    parser.add_argument('--frames', type=str, default='0-10',
                       help='Frame range to visualize (e.g., "0-10" or "0,5,10")')
    parser.add_argument('--downsample', type=int, default=10,
                       help='Downsample factor for point clouds (default: 10)')
    parser.add_argument('--output', type=str, default='visualization.html',
                       help='Output HTML file name')
    
    args = parser.parse_args()
    
    # 解析帧索引
    if '-' in args.frames:
        start, end = map(int, args.frames.split('-'))
        frame_indices = list(range(start, end + 1))
    else:
        frame_indices = [int(x) for x in args.frames.split(',')]
    
    print(f"Visualizing frames: {frame_indices}")
    print(f"Data directory: {args.data_dir}")
    print(f"Downsample factor: {args.downsample}")
    
    # 创建可视化
    fig = create_visualization(args.data_dir, frame_indices, args.downsample)
    
    # 保存为HTML文件
    fig.write_html(args.output)
    print(f"Visualization saved to: {args.output}")
    
    # 显示
    fig.show()


if __name__ == "__main__":
    main()
