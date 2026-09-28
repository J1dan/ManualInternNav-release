# Legacy ImagiNav usage guide

The current ImagiNav reproduction instructions are in [ImagiNav/README.md](../../ImagiNav/README.md).
This older guide is retained for historical reference and may contain outdated
commands.

This document covers the main ImagiNav workflow: environment setup, running inference with pretrained checkpoints, preparing datasets, training custom models, and evaluating model quality (both in simulation and via offline metrics).

---

## Table of Contents
- **[1. Environment Setup](#1-environment-setup)**: Install dependencies, configure the `uv` environment, and set up GPU hardware requirements.
- **[2. Inference](#2-inference)**: Download pretrained ImagiNav finetunes and run zero-shot egocentric video generation from reference frames.
- **[3. Training](#3-training)**: Prepare or extract custom motion clips, precompute LTX features, and run LoRA or full finetuning.
- **[4. Evaluation](#4-evaluation)**: Execute end-to-end evaluation inside the Isaac Sim docker container and compute offline video quality metrics (FVD, LPIPS).

---

## 1. Environment Setup

Prerequisites for the full ImagiNav workflow are Linux, Git, Git LFS, `ffmpeg`,
an NVIDIA GPU with a compatible driver, and enough disk space for the selected
models and datasets. Simulation additionally requires Docker with the NVIDIA
Container Toolkit. Dataset and model repositories may require accepting their
terms and authenticating with Hugging Face.

Clone the repository with all pinned submodules:

```bash
git clone --recurse-submodules https://github.com/J1dan/ManualInternNav.git
cd ManualInternNav
```

If the repository was cloned without `--recurse-submodules`, initialize it now.
The status command should print a commit for every submodule without a leading
`-` (uninitialized) marker:

```bash
git submodule update --init --recursive
git submodule status --recursive
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Most non-simulation steps run from the local uv environment in `ImagiNav/LTX-Video-Trainer`.

```bash
cd ImagiNav/LTX-Video-Trainer
uv sync --no-workspace --frozen --python 3.10
source .venv/bin/activate
```

Install the pinned ImagiNav pipeline extras that are not part of the base LTX
trainer lockfile, followed by the two local evaluation packages:

```bash
uv pip install --python .venv/bin/python -r ../requirements.txt -i https://pypi.org/simple
uv pip install --python .venv/bin/python --no-deps \
    -e ../vggt \
    -e ../video_quality_eval/content-debiased-fvd
```

On Blackwell GPUs, the default locked PyTorch build may not support `sm_120`.
The following hardware-specific override intentionally replaces the locked
PyTorch build with CUDA 13 nightly packages:

```bash
uv pip install --python .venv/bin/python --upgrade --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install --python .venv/bin/python --upgrade bitsandbytes -i https://pypi.org/simple
```

If bitsandbytes or CUDA libraries fail to load, add the CUDA runtime libraries from the uv environment to `LD_LIBRARY_PATH` before running generation/evaluation:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

Verify the environment:

```bash
cd ../..
ImagiNav/LTX-Video-Trainer/.venv/bin/python - <<'PY'
import cv2, torch, google.genai, vggt, cdfvd, torchmetrics, lpips, skimage
from ImagiNav.vggt.imaginav_scripts.infer_move import VGGT
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
```

Install `ffmpeg` on the host if extracted clips need broad player compatibility:

```bash
sudo apt-get install ffmpeg
```

---

## 2. Inference

Run inference using the pre-trained checkpoints to quickly generate imagined navigation video clips.

### 2.1 Checkpoint Downloads

The following video-quality metrics were measured on the full evaluation set:

| Model Variant | FVD ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ | Mot. Fid. ↑ | RPE-T (m) ↓ | RPE-R (deg) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base LTX-2B (zero-shot) | 391.14 | 0.520 | 12.61 | 0.37 | 0.40 | 0.036 | 1.39 |
| Real-finetuned 2B w/o AC-MoE | 73.08 | 0.486 | 13.36 | 0.40 | 0.60 | 0.028 | 1.31 |
| Sim-finetuned 2B | 72.72 | 0.514 | 13.03 | 0.39 | 0.67 | 0.051 | 1.73 |
| Real-finetuned 2B AC-MoE | 65.39 | 0.481 | 13.27 | 0.40 | 0.72 | 0.024 | 1.18 |
| Unified 2B full-finetuned | 56.15 | 0.475 | 13.33 | 0.40 | 0.72 | 0.027 | 1.29 |
| Unified 13B LoRA-finetuned | 58.88 | 0.474 | 13.70 | 0.41 | 0.75 | 0.021 | 1.31 |

Base pretrained checkpoints are downloaded automatically on first use through Hugging Face.

ImagiNav finetuned checkpoints are not auto-downloaded by the configs. Download the provided weights, then point the configs at the local `.safetensors` files:

```bash
cd ImagiNav
mkdir -p checkpoints
```

AC-MoE LoRA checkpoints:
```bash
huggingface-cli download J1dan/imaginav --include "LTX-2B-AC-MoE-LoRA/*" --local-dir checkpoints/
```

Unified 13B LoRA checkpoint:
```bash
huggingface-cli download J1dan/imaginav LTX-13B-LoRA/lora-weights-13b-85epoch.safetensors --local-dir checkpoints/
```

Unified 2B full-finetuned checkpoint:
```bash
huggingface-cli download J1dan/imaginav LTX-2B-Full/model-weights-85epoch.safetensors --local-dir checkpoints/
```

### 2.2 Quick Demo Inference

Create a demo reference image from any reference video or dataset, and run generation using a downloaded checkpoint:

```bash
cd ImagiNav/LTX-Video-Trainer
source .venv/bin/activate

mkdir -p ../samples/reference_images
# If dataset is downloaded, you can extract a frame from a real clip:
DEMO_VIDEO=$(find datasets/real-clips-121/videos -name "*.mp4" | head -n 1)
# Alternatively, prepare any PNG reference image at ImagiNav/samples/reference_images/demo_first_frame.png
ffmpeg -y -i "$DEMO_VIDEO" -frames:v 1 ../samples/reference_images/demo_first_frame.png

python inference.py \
    --image ../samples/reference_images/demo_first_frame.png \
    --prompt "POV, Pan left and dolly forward" \
    --model_path ../checkpoints/LTX-2B-AC-MoE-LoRA/lora-weights-2b-85epoch-left.safetensors
```

Generated videos are saved under `ImagiNav/samples/generated_videos/`. For best results, use POV-style prompts (e.g., `POV, Dolly forward`, `POV, Pan left and dolly forward`, or `POV, Dolly forward and pan right at the opening`).

---

## 3. Training

Finetune LTX-Video on egocentric navigation clips.

### 3.1 Dataset Preparation

You can either start from the provided ImagiNav dataset or build your own dataset from raw egocentric videos.

#### Option A: Use the Provided Dataset

The released dataset already contains the processed clips and precomputed LTX training features (including the filtered splits `dataset-left.json` and `dataset-right.json`). Use this path if you want to reproduce training without rerunning motion extraction, labeling, or feature extraction.

Run the training-data download from the LTX trainer directory. This downloads only the real and simulation training folders into the paths expected by the training configs:

```bash
cd ImagiNav/LTX-Video-Trainer
mkdir -p datasets
huggingface-cli download J1dan/imaginav-dataset --repo-type dataset --include "real-clips-121/**" "simulation-clips-121/**" --local-dir datasets
```

#### Option B: Prepare Your Own Dataset

##### 1. Extract Motion Clips
The extraction scripts run VGGT on raw videos, estimate camera motion, and cut 121-frame clips with motion primitive metadata.

For real-world videos:
```bash
cd ImagiNav
python vggt/imaginav_scripts/extract_motion_clips.py \
    --input_folders /path/to/raw_videos \
    --output_folder /path/to/extracted_outputs \
    --window_len 121 \
    --stride 24 \
    --analysis_subsample_step 6 \
    --gpus 0
```

For simulation videos, use the simulation-tuned thresholds:
```bash
cd ImagiNav
python vggt/imaginav_scripts/extract_motion_clips_simulation.py \
    --input_folders /path/to/simulation_videos \
    --output_folder /path/to/extracted_outputs \
    --window_len 121 \
    --stride 24 \
    --analysis_subsample_step 2 \
    --gpus 0
```

##### 2. Label Extracted Clips
Use Gemini to convert extracted motion clips into navigation-style captions. Set `GOOGLE_API_KEY` or pass `--api_key`:
```bash
export GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

cd ImagiNav
python vggt/imaginav_scripts/label_clips.py \
    --input_folder /path/to/extracted_outputs \
    --metadata /path/to/extracted_outputs/sum_metadata.json \
    --output /path/to/extracted_outputs/dataset.json
```

##### 3. (Optional) Aggregate Multiple Label Files
If labels were produced in multiple runs, merge them into a single trainer dataset:
```bash
cd ImagiNav
python vggt/imaginav_scripts/extract_aggregate_dataset_json.py \
    --folder labeled_results \
    --output /path/to/extracted_outputs/dataset.json
```

##### 4. Precompute LTX Training Features
Convert the labeled clip dataset into cached video latents and text embeddings:
```bash
cd ImagiNav/LTX-Video-Trainer
python scripts/preprocess_dataset.py /path/to/extracted_outputs/dataset.json \
    --resolution-buckets "480x256x121" \
    --caption-column "caption" \
    --video-column "media_path" \
    --model-source "LTXV_2B_0.9.6_DEV" \
    --id-token "POV," \
    --device cuda:0
```

### 3.2 Model Training

Finetunes the LTX-Video model using LoRA (Low-Rank Adaptation). The repository supports training either action-specific experts (for the AC-MoE model) or a single unified model for all navigation motions.

The configuration YAML files are pre-configured by default to point to the correct paths in `datasets/real-clips-121/.precomputed/` so no manual dataset path modifications are needed unless you are training on your own custom dataset.

1.  **Select and Modify Config (Optional):**
    Choose a pre-configured training file and optionally customize settings inside the YAML file based on your GPU or custom dataset:
    *   **Action-Specific Experts (2B LoRA Left/Right):**
        *   `configs/ltxv_2b_lora-imaginav-left.yaml`
        *   `configs/ltxv_2b_lora-imaginav-right.yaml`
    *   **Unified Model (2B Full Fine-Tune):**
        *   `configs/ltxv_2b_full-imaginav.yaml`
    *   **Unified Model (13B LoRA Fine-Tune):**
        *   `configs/ltxv_13b_lora-imaginav.yaml`

2.  **Execute Training:**
    ```bash
    cd ImagiNav/LTX-Video-Trainer
    source .venv/bin/activate
    
    # To train 2B LoRA experts:
    python scripts/train.py configs/ltxv_2b_lora-imaginav-left.yaml
    
    # To train 2B Full fine-tune:
    python scripts/train.py configs/ltxv_2b_full-imaginav.yaml
    
    # To train 13B LoRA fine-tune:
    python scripts/train.py configs/ltxv_13b_lora-imaginav.yaml
    ```
    *   **Output**: Finetuned weights saved in the output directory specified in the config.

---

## 4. Evaluation

Evaluate model execution in the interactive simulator and compute offline quality metrics.

### 4.1 Simulation in Docker

Run the evaluation loop within the Isaac Sim simulation container.

1.  **Prepare Data:**
    Ensure the required datasets are downloaded. The following commands will automatically create the `interiornav_data/` and `data/` directories if they do not exist:
    ```bash
    ImagiNav/LTX-Video-Trainer/.venv/bin/huggingface-cli download \
        spatialverse/InteriorAgent --repo-type dataset --local-dir interiornav_data/scene_data
    ImagiNav/LTX-Video-Trainer/.venv/bin/huggingface-cli download \
        spatialverse/InteriorAgent_Nav --repo-type dataset --local-dir data/vln_pe/raw_data/interiornav
    ImagiNav/LTX-Video-Trainer/.venv/bin/huggingface-cli download \
        InternRobotics/Embodiments --repo-type dataset --local-dir data/Embodiments
    ```
    *(Note: `InternRobotics/Embodiments` is gated. Authenticate with `huggingface-cli login` before running.)*

2.  **Prepare Checkpoints and Configs:**
    Put trained or downloaded ImagiNav checkpoints under `ImagiNav/checkpoints/`, then update `ImagiNav/configs/default_config.yaml` and `scripts/eval/configs/imaginav_eval_cfg.py` as needed.

3.  **Launch Docker Container:**
    Start the simulation environment using Docker, mounting necessary directories:
    ```bash
    docker run --rm -it \
        --gpus all \
        --ipc=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v $(pwd):/root/InternNav \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        -v ./data:/root/InternNav/data \
        -v ./interiornav_data:/root/InternNav/interiornav_data \
        -v ./data/scene_data/mp3d_pe:/isaac-sim/Matterport3D/data/v1/scans:ro \
        xerneaschen/internnav:v1.2
    ```

4.  **Execute Evaluation:**
    Set the required Google API key (used for Gemini model integration) and start the evaluation script:
    ```bash
    export GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    ./scripts/eval/bash/start_imaginav_eval.sh
    ```
    *   **Output**: Evaluation logs and generated video results. Videos can be found in `logs/{task_name}/ImagiNav_logs/`.

### 4.2 Video Model Quality Evaluation

Evaluate quality metrics (LPIPS, PSNR, SSIM, FVD, motion fidelity) offline.
Run all commands in this section from the repository root:

```bash
mkdir -p ImagiNav/video_quality_eval/datasets
ImagiNav/LTX-Video-Trainer/.venv/bin/huggingface-cli download J1dan/imaginav-dataset --repo-type dataset --include "eval_clips/**" --local-dir ImagiNav/video_quality_eval/datasets
ln -sfn eval_clips ImagiNav/video_quality_eval/datasets/reference_videos
```

Run evaluation using the local LTX virtual environment:

```bash
PY=ImagiNav/LTX-Video-Trainer/.venv/bin/python
CONFIG=ImagiNav/video_quality_eval/configs/video_quality_eval.yaml

# 1. Extract first frames from reference videos
"$PY" -m ImagiNav.video_quality_eval.scripts.extract_first_frames --config "$CONFIG"

# 2. Generate evaluation videos
"$PY" -m ImagiNav.video_quality_eval.eval_video_gen --config "$CONFIG"

# 3. Calculate metrics (FVD, LPIPS, PSNR, SSIM, Motion Fidelity)
"$PY" -m ImagiNav.video_quality_eval.evaluator \
    --config "$CONFIG" \
    --dataset ImagiNav/video_quality_eval/datasets/reference_videos/dataset-20260131_with_firstframes_full.json \
    --out ImagiNav/video_quality_eval/results/full
```

The evaluator writes per-sample metrics to `results/results.json` and aggregate metrics to `results/aggregate_metrics.json`. FVD downloads the VideoMAE checkpoint automatically.
