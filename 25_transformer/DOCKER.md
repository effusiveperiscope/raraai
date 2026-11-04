# Running 25_transformer on vast.ai

Everything needed is in `Dockerfile` + `requirements.txt` + `docker/prefetch_models.py`.
The image is built **on top of `vastai/pytorch`, so keep the base layer intact**:
vast.ai hosts already have those layers cached, which is what makes instance
startup a pull of a couple of GB instead of ~10 GB.

---

## 1. Build and push

From this directory (`25_transformer/`, which is the build context —
`.dockerignore` keeps `data/`, `checkpoints/`, `logs/`, `pretrain/` and audio out):

```bash
docker build -t <dockerhub-user>/raraai-25-transformer:latest .

# optional: also bake the Qwen3-ASR + forced-aligner weights (+~2.5 GB).
# Only worth it if the instance will run 1_preprocess.py / 2_preprocess.py
# (i.e. align audio) -- stage-2 training on existing code tensors never uses
# those weights.
docker build --build-arg PREFETCH_QWEN=1 -t <dockerhub-user>/raraai-25-transformer:latest .

docker push <dockerhub-user>/raraai-25-transformer:latest
```

Build arguments:

| ARG | Default | Meaning |
| --- | --- | --- |
| `VAST_BASE` | `vastai/pytorch:cuda-13.0.3-auto` | Base image. Switch to e.g. `vastai/pytorch:cuda-12.9.2-auto` if your hosts run a driver older than 580 (CUDA 13 needs ≥ 580). `-auto` = latest torch published for that CUDA version, so torch + torchaudio stay consistent. |
| `PREFETCH_QWEN` | `0` | `1` bakes `Qwen/Qwen3-ASR-0.6B` + `Qwen/Qwen3-ForcedAligner-0.6B` into the image. |

## 2. Launch on vast.ai

Pick the image in **Edit Image & Config → Image Path**, and:

* **Disk**: ≥ 100 GB. The image is ~11 GB to pull and ~30 GB unpacked (most of
  it the `vastai/pytorch` base, which vast.ai hosts already have cached), plus
  room for datasets and checkpoints.
* **Ports**: `-p 22:22 -p 8080:8080 -p 6006:6006` if you want SSH, Jupyter and
  TensorBoard exposed (the vast.ai portal normally opens them for you).
* **Launch mode**: SSH or Jupyter — the base image sets both up, no `onstart`
  script required.

The image ships torch (cu130 build, ≥ 2.9 because `2_training.py` uses
`torch.optim.Muon`) and every Python dependency in `/venv/main`.

## 3. Where things live

| Path | Contents |
| --- | --- |
| `/workspace/25_transformer` | This project. Copied there from `/opt/workspace-internal/25_transformer` by the base image **on first boot** (only when that path does not exist yet), so it survives a `/workspace` volume mount. |
| `/venv/main` | Python 3.12 venv (activate with `source /venv/main/bin/activate`, usually already active). |
| `/opt/hf-cache` | `HF_HOME`. Contains RMVPE (`rvc_rmvpe.pt`) and the `therealvul/g2pen_tokenizer_clean` tokenizer; add the Qwen weights with `PREFETCH_QWEN=1`. |
| `/opt/nltk_data` | `NLTK_DATA` — the POS tagger `horsephonemizer` needs. |
| `./checkpoints`, `./logs`, `./data` | Written by the scripts **relative to the project dir**, so put the project on a volume-backed path if you care about keeping them. |

Everything must be run **from the project directory** — paths inside the scripts
(`configs/…`, `data/…`, `checkpoints/…`, `new_horsewords.clean`) are relative to
the cwd:

```bash
cd /workspace/25_transformer
```

## 4. Typical session

```bash
cd /workspace/25_transformer && source /venv/main/bin/activate

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# stage-2 training (the usual reason to rent a GPU here)
python 2_training.py --config configs/transformer.yaml

# stage-1 training
python 1_training.py --config configs/base.yaml

# stage-2 tensor extraction from a stage-1 checkpoint
python 2_preprocess.py \
    --filelist filelists_2/25_stage2_base1.txt \
    --output_dir data/base2 \
    --checkpoint checkpoints/<exp>/last.ckpt \
    --config configs/base.yaml --speaker_map speakers.json

# ad-hoc smoke tests (no dataset needed)
python modeling/transformer.py
python test_stage2_conditioning.py --checkpoint checkpoints/<exp>/last.ckpt --device cuda
```

TensorBoard is already running in the base image against
`TENSORBOARD_LOG_DIR=/workspace/25_transformer/logs`, so Lightning runs appear
in the portal's TensorBoard tab without extra setup.

## 5. Keeping data across instances

The image contains **code only**. Datasets (`data/…`), checkpoints and logs are
created at runtime. To persist them, attach a vast.ai volume and either mount it
at `/workspace` (the code is copied in from `/opt/workspace-internal` on first
boot) or point the scripts at it, e.g. symlink:

```bash
ln -s /workspace/volume/checkpoints /workspace/25_transformer/checkpoints
ln -s /workspace/volume/logs        /workspace/25_transformer/logs
```

The first-boot copy is skipped when `/workspace/25_transformer` already exists
(that is how it avoids clobbering edits), so a volume that already holds an old
copy keeps it. Refresh by hand in that case:

```bash
rsync -a --exclude data --exclude checkpoints --exclude logs \
    /opt/workspace-internal/25_transformer/ /workspace/25_transformer/
```

A late boot stage (`/etc/vast_boot.d/90-25-transformer-perms.sh`) re-applies
`a+rwX` to the copied tree, so the non-root `user` account can write
`checkpoints/`, `logs/` and `data/`.

`HF_HOME=/opt/hf-cache` is inside the image, so any *extra* model downloaded at
runtime is lost when the instance is destroyed. For a long-lived cache, set
`HF_HOME=/workspace/hf-cache` in the instance's environment variables instead.

## 6. Notes / troubleshooting

* **Driver too old**: `nvidia-smi` failing / torch reporting no CUDA device on a
  CUDA 13 image means the host driver is < 580. Rebuild with
  `--build-arg VAST_BASE=vastai/pytorch:cuda-12.9.2-auto`, or pick a host with a
  newer driver.
* **`import ultimate_xc` fails**: it needs `PyQt5` (already installed). It is
  imported for its exception-hook side effect in both training scripts —
  do not remove the import.
* **`FileNotFoundError: new_horsewords.clean`**: you are not in the project
  directory.
* **NLTK/`nltk.download` errors at import**: harmless if the data was baked in
  (it is), fatal if the image was built without network access — rebuild the
  prefetch layer.
* **Rebuilding after a code change**: only the `COPY .` layer and below are
  invalidated; the dependency and model-prefetch layers are reused.
