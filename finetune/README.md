This folder contains the code and instructions for fine-tuning a chatbot model using PyTorch and GPU acceleration. It provides multiple recipes for fine-tuning. Follow the steps below to set up your environment and install the necessary dependencies.

## Setup Instructions

1. Install Miniconda:

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   sudo sh Miniconda3-latest-Linux-x86_64.sh
   ```
2. Create a Conda environment:
   ```bash
   conda create --name chatbot python=3.10
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate chatbot
   ```
3. Install GPU drivers and CUDA compiler:
   ```bash
   curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output    install_gpu_driver.py
   sudo python3 install_gpu_driver.py
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```
5. Install PyTorch with GPU support and other dependencies:
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```
6. Modify Jupyter config file to enable running it from your browser:
   ```bash
   cp jupyter_notebook_config.py ~/.jupyter/
   ```
