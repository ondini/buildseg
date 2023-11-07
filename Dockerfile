# Base image 
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Copy the files from the local directory to the container
# COPY . /app/ # commented out, as it is easier to use codes from shared folder due to githubing etc.

# Set working directory
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app/codes/utils/fo_utils"
ENV FIFTYONE_MODULE_PATH custom_embedded_files
ENV DEBIAN_FRONTEND noninteractive 

# install dependencies
RUN apt-get update \
    && apt-get install wget ffmpeg libsm6 libxext6 libcurl4 -y

# install fiftyone libcrypto dependency
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb && rm libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb

# install needed python libraries
RUN pip install numpy matplotlib scikit-image scikit-learn scipy \
    opencv-python albumentations tensorboard fiftyone ipykernel kornia

EXPOSE 6006
EXPOSE 5151