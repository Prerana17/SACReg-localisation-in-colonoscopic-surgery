#!/bin/bash
#------------------------------------------------------------------
# SLURM Directives
#------------------------------------------------------------------
#SBATCH --job-name=Vit_GeM      # 作业名称：更新为信息检索模型训练
#SBATCH --partition=capella              # 根据sinfo输出，使用主分区 'capella'
#SBATCH --nodes=1                        # 请求1个节点
#SBATCH --ntasks=1                       # 在该节点上运行1个任务
#SBATCH --cpus-per-task=8                # 为该任务请求8个CPU核心 (与Python脚本中的num_workers保持一致或略多)
#SBATCH --mem=32G                        # 请求32GB内存 (ResNet模型和数据可能需要较多，根据实际监控调整)
#SBATCH --gpus=h100:1                    # 请求1块H100 GPU (确保'h100'是该分区可用的GRES名称)
#SBATCH --time=2-00:00:00                # 请求2天的运行时限 (2天0小时0分0秒)
#SBATCH --mail-user=zhifan.yu@mailbox.tu-dresden.de # 你的邮箱
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80 # 作业状态邮件通知

# 输出和错误文件的路径
# 确保 /data/horse/ws/zhyu410g-horse_C3VD_data/slurm_logs/ 目录存在
# 你可以先手动创建它: mkdir -p /data/horse/ws/zhyu410g-horse_C3VD_data/slurm_logs/
# 注意：这里已将路径更新为你的C3VD工作区
#SBATCH --output=/data/horse/ws/zhyu410g-horse_C3VD_data/slurm_logs/Vit_GeM_training_%j.out  # 标准输出
#SBATCH --error=/data/horse/ws/zhyu410g-horse_C3VD_data/slurm_logs/Vit_GeM_training_%j.err   # 标准错误

#------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------
echo "========================================================"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB" # 或者根据您的集群配置使用 SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK
if [ -n "$SLURM_GPUS" ]; then
    echo "GPUs allocated: $SLURM_GPUS"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "========================================================"
echo ""

# 1. 激活您的 Python 虚拟环境 (venv)
# 请将此路径替换为您实际的虚拟环境路径
VENV_PATH="/home/h8/zhyu410g/Team_project_IR/IR" # 你的虚拟环境路径
echo "Attempting to activate virtual environment at: ${VENV_PATH}/bin/activate"

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Virtual environment activated."
    echo "Python executable: $(which python)"
    echo "pip list (first few lines):"
    pip list --format=freeze | head -n 10
else
    echo "ERROR: Virtual environment activation script not found at ${VENV_PATH}/bin/activate."
    echo "Please ensure the virtual environment path is correct and it has been created."
    exit 1 # Stop the job if environment activation failed
fi
echo "--------------------------------------------------------"
echo ""

# 2. 设置 Hugging Face 缓存目录到您的工作区
# 确保这个目录存在。通常用于下载预训练模型权重。
export HF_HOME="/data/horse/ws/zhyu410g-horse_C3VD_data/huggingface_cache"
mkdir -p "$HF_HOME"
echo "Hugging Face cache directory set to: $HF_HOME"
echo "--------------------------------------------------------"
echo ""

#------------------------------------------------------------------
# Execute the Python Script
#------------------------------------------------------------------
# 切换到您的Python脚本所在的目录
# 假设您的Python脚本 (ir_colonscopy_model.py) 放在 /data/horse/ws/zhyu410g-horse_C3VD_data/script/ 目录下
cd ~/Team_project_IR/IR/Vit_Gem/ 

echo "Current working directory for script execution: $(pwd)"
echo "Executing Python script: vit_gem_imp.py" # ***脚本名称已更新***
echo "Start time: $(date)"
echo "--------------------------------------------------------"
echo ""

# 运行 Python 脚本
python vit_gem_imp.py

# 捕获 Python 脚本的退出状态
EXIT_STATUS=$?
echo ""
echo "--------------------------------------------------------"
echo "Python script finished with exit status: $EXIT_STATUS"
echo "End time: $(date)"

if [ $EXIT_STATUS -ne 0 ]; then
    echo "ERROR: Python script failed with exit status $EXIT_STATUS"
fi

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"

exit $EXIT_STATUS