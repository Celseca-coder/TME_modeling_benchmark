#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=space_gm
BASE_ENV=p3

echo "=== 1. 克隆 ${BASE_ENV} -> ${ENV_NAME} ==="
conda create -y -n "${ENV_NAME}" --clone "${BASE_ENV}"

echo "=== 2. 激活新环境 ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== 3. 探测 CUDA 版本 ==="
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
else
    CUDA_VER=""
fi
echo "检测到 CUDA: ${CUDA_VER:-未检测到 / 无GPU}"

# 根据CUDA版本选择torch的安装源;没有GPU则装CPU版
if [[ "${CUDA_VER}" == "12."* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "${CUDA_VER}" == "11."* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
echo "使用 torch 安装源: ${TORCH_INDEX}"

echo "=== 4. 安装 torch ==="
pip install 'torch>=2.2.2' --index-url "${TORCH_INDEX}"

echo "=== 5. 安装 torch_geometric（及其可选加速依赖）==="
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CU_TAG=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
echo "torch=${TORCH_VER}, tag=${CU_TAG}"

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CU_TAG}.html" || \
    echo "可选加速包安装失败可忽略,torch_geometric 核心功能不依赖它们"

echo "=== 6. 验证 ==="
python -c "
import torch, torch_geometric
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('torch_geometric:', torch_geometric.__version__)
"

echo "=== 完成。使用方式: conda activate ${ENV_NAME} ==="