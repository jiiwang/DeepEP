ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG DEEPEP_COMMIT=dbe63eaed372d7c2da8072085940e7338fcb8cf9
ENV NVSHMEM_HOME=/workspace/nvshmem/install
ENV MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${NVSHMEM_HOME}/lib"

# Required for NVSHMEM build
ENV CUDA_ARCH="90;100"
ARG NVSHMEM_VERSION=3.4.5
ARG GDRCOPY_VERSION=2.5.1
# Required for DeepEP build
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"
ENV NVSHMEM_DIR=${NVSHMEM_HOME}

# Ensure tools for adding GPG keys are available
RUN apt update && apt install -y wget gnupg ca-certificates

# Install dependencies
RUN apt update && apt install -y software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt update
RUN apt install -y --no-install-recommends python-is-python3 python3.12-venv python3.12-dev
RUN apt install -y --no-install-recommends git-lfs wget curl build-essential cmake
RUN apt install -y --no-install-recommends openmpi-bin libopenmpi-dev libfabric-dev
RUN apt install -y --no-install-recommends devscripts pkg-config debhelper dkms pip fakeroot

# Install GDRCopy
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
 && git clone https://github.com/NVIDIA/gdrcopy.git -b v${GDRCOPY_VERSION} \
 && cd gdrcopy/packages \
 && CUDA=/usr/local/cuda ./build-deb-packages.sh \
 && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
 && cd / && rm -rf /tmp/gdrcopy

# Install NVSHMEM
WORKDIR /workspace
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && tar -xf nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && mv nvshmem_src nvshmem \
    && rm -f /sgl-workspace/nvshmem_src_cuda12-all-all-${NVSHMEM_VERSION}.tar.gz \
    && mkdir nvshmem/build
WORKDIR /workspace/nvshmem
ENV CPATH=${MPI_HOME}/include:${CPATH}
ENV CMAKE_BUILD_PARALLEL_LEVEL=64
RUN export CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx && \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_LIBFABRIC_SUPPORT=0 \
    NVSHMEM_IBDEVX_SUPPORT=0 \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=0 \
    NVSHMEM_BUILD_EXAMPLES=0 \
    NVSHMEM_DISABLE_CUDA_VMM=1 \
    NVSHMEM_MPI_SUPPORT=0 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=${NVSHMEM_HOME} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} &&\
    cmake --build build  -j${CMAKE_BUILD_PARALLEL_LEVEL} 
RUN mkdir -p ~/.config/pip; echo "[global] \n\
break-system-packages = true" > ~/.config/pip/pip.conf
RUN cmake --build build --target install && \
    # install python wheels that are built along with nvshmem
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d'.' -f1) && \
    pip install build/dist/nvshmem4py_cu${CUDA_MAJOR}*-many*.whl

# install torch and ninja
RUN CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d'.' -f1) && CUDA_MINOR=$(echo ${CUDA_VERSION} | cut -d'.' -f2) && \
    pip install torch --no-build-isolation --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR} --break-system-packages
RUN pip install numpy --break-system-packages
RUN apt install ninja-build -y

# Uninstsall nvdia-nvshmem-cu12 so that it use the one built above is used.
# Otherwise, IBGDA will not work with nvshmem installed as part of torch package.
RUN CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d'.' -f1) && pip uninstall nvidia-nvshmem-cu${CUDA_MAJOR} -y 

# Install DeepEP
WORKDIR /workspace
RUN git clone https://github.com/jiiwang/DeepEP.git && cd DeepEP && git checkout ${DEEPEP_COMMIT}
WORKDIR /workspace/DeepEP
# Modify DeepEP to enable NIC PE mapping for NVSHMEM (Deprecate when DeepEP fixes in upstream))
RUN    sed -i "/'NVSHMEM_IBGDA_NUM_RC_PER_PE'/a \\            os.environ['NVSHMEM_ENABLE_NIC_PE_MAPPING'] = '1'\n            local_rank = os.environ['LOCAL_RANK'] if 'LOCAL_RANK' in os.environ else self.rank % 8\n            os.environ['NVSHMEM_HCA_LIST'] = f'mlx5_{local_rank}:1'\n            print(f\"Setting {self.rank} to use {local_rank}\", flush=True)" deep_ep/buffer.py 
RUN pip install . --no-build-isolation --break-system-packages

# Set workspace directory
WORKDIR /workspace
