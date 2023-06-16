wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sudo sh Miniconda3-latest-Linux-x86_64.sh

conda create --name chatbot python=3.10

source ~/miniconda3/etc/profile.d/conda.sh

conda activate chatbot

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py


wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

sudo sh cuda_11.8.0_520.61.05_linux.run

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
