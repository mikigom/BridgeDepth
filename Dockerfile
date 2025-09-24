# Base image with CUDA 12.8 and cuDNN, on Ubuntu 22.04
FROM nvcr.io/nvidia/tensorrt:25.03-py3

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python 3
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libyaml-cpp-dev \
    libopencv-dev \
    python3-opencv \
    cmake \
    wget \
    gnupg \
    ca-certificates \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the repository contents
COPY . .

# Install Python dependencies
# Install PyTorch and torchvision compatible with CUDA 12.8
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# RUN pip3 install --no-deps -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/cu128
# RUN pip install -U packaging && pip install -U flash-attn==2.8.1 --no-build-isolation

# Install dependencies from requirements.txt and others mentioned in get_started.md
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir timm==0.5.4

# Set environment variable for pretrained models
ENV TORCH_HOME="/app/pretrained_models"
RUN mkdir -p $TORCH_HOME

# Set the default command to bash
CMD ["/bin/bash"]