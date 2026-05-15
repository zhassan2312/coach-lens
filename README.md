---
title: Coach's Tactical Lens
emoji: 🏀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
hardware: t4-small
---

# Coach's Tactical Lens

This app detects basketball players using RF-DETR, segments and tracks them with SAM2, clusters teams using jersey appearance, and renders short-clip tactical views for coaches.

The expensive pass runs once per uploaded clip. The app stores resized frames, masks, track IDs, and team labels in Gradio session state, so changing the team selector or tactical effects re-renders the video without rerunning RF-DETR or SAM2.

## Model Files

The app expects these local paths:

- `checkpoints/checkpoint_best_regular.pth`
- `checkpoints/sam2.1_hiera_tiny.pt`
- `configs/sam2.1_hiera_t.yaml`

For Hugging Face Spaces, the recommended setup is a separate model repo containing those files. By default the app looks for `zohaib/coach-lens-checkpoints`; override it with:

- `CHECKPOINT_REPO_ID=your-username/coach-lens-checkpoints`

Set this Space secret before running:

- `HF_TOKEN=your_huggingface_token`

## Limits

Uploads are limited to MP4 files under 100 MB and under 10 seconds. Frames are resized internally to 720 px width before inference.

## Local Run

Use an isolated environment. Python 3.10 or 3.11 is recommended for this stack.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The SAM2 real-time fork builds a CUDA extension during installation. On Windows, the NVIDIA driver alone is not enough; you also need the CUDA Toolkit installed, `CUDA_HOME` set to the toolkit root, and `nvcc` available on `PATH`. If you do not want to build SAM2 locally, deploy on a GPU Hugging Face Space or use a Docker Space where CUDA build tooling is available.

Training stays offline in the notebook. Deployment only loads the trained RF-DETR checkpoint and runs inference.
