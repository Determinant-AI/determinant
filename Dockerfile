# Use the official Python base image
FROM rayproject/ray:7ee908-py39-cpu

# Switch to root user
USER root

# Update package list and install required dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg

# Add Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update package list to include Google Cloud SDK repository
RUN apt-get update && apt-get install google-cloud-cli

USER ray

# Copy the requirements.txt file into the container
COPY requirements_.txt .

# Install the package in editable mode
cd python
pip install -e .

# Install the dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements_.txt
