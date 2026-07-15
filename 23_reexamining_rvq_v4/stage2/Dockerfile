FROM vastai/pytorch:cuda-13.0.2-auto

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# 2. Setup workspace and requirements
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --ignore-requires-python
RUN pip install tensorboardX

# 3. Perform Sparse Clone
RUN git clone --filter=blob:none --sparse https://github.com/effusiveperiscope/raraai.git && \
    cd raraai && \
    git sparse-checkout set 23_reexamining_rvq_v4/ && \
    git checkout sem-codec

# 4. Set final working directory to the code location
WORKDIR /workspace/raraai/23_reexamining_rvq_v4/stage2/