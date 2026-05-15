FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# 1. 安装 PyTorch CPU 版（2.4.1 兼容 directml）
RUN pip install --no-cache-dir torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 DirectML 后端（调用 AMD 显卡）
RUN pip install --no-cache-dir torch-directml==0.2.5.dev240914

# 3. 复制 requirements.txt 并安装其余依赖（不含 torch）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制修改过的 pokerenv 源码
COPY pokerenv_pkg/ /usr/local/lib/python3.11/site-packages/pokerenv/

EXPOSE 6006

CMD ["python", "train.py"]
