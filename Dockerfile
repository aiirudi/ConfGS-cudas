FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates unzip git \
        build-essential ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Build the conda env and compile the three CUDA extensions from this checkout.
# environment.yml's pip section references these relative source paths.
WORKDIR /build
COPY environment.yml /build/environment.yml
COPY submodules/ /build/submodules/
ARG TORCH_CUDA_ARCH_LIST="8.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN conda env create -f /build/environment.yml -n rfgs \
    && conda clean -afy \
    && rm -rf /build

# Make the rfgs env the default Python on PATH
ENV PATH=$CONDA_DIR/envs/rfgs/bin:$PATH
ENV CONDA_DEFAULT_ENV=rfgs

# Project source is bind-mounted here at runtime
WORKDIR /workspace/RFGS

CMD ["python", "test.py"]
