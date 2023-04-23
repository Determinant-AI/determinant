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

# Assuming that setup.py is in the root of your source code,
# copy the entire source code directory into the container
COPY . /app

# Change the working directory
WORKDIR /app


# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the package 
RUN pip install --upgrade pip

# Update pip and setuptools to the latest version
RUN pip install --upgrade pip setuptools
RUN python setup.py install

# Install the dependencies
RUN pip install -r requirements.txt

USER ray
