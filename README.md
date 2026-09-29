# Real-Time LocateAnything — NVIDIA LocateAnything-3B

Real-time visual grounding and object localization using **NVIDIA LocateAnything-3B** with webcam and image support.

This project is designed for practical inference on NVIDIA GPUs, including lower-VRAM GPUs. It automatically selects an appropriate model-loading strategy based on available GPU memory and includes several stability fixes for CUDA, PyTorch, Transformers, and OpenCV environments.

## ✨ Features

* Real-time webcam visual grounding
* Single-object grounding
* Multi-object grounding
* Object detection from text queries
* Text detection
* GUI element localization
* Point-based object localization
* Image-file inference
* Bounding-box and point visualization
* Automatic GPU VRAM detection
* Three-tier model loading
* CPU/GPU memory offloading
* 4-bit NF4 quantization for low-VRAM GPUs
* CUDA out-of-memory recovery
* Frame skipping for lower GPU usage
* Automatic input resizing
* Matplotlib-based display to avoid OpenCV GUI/OpenGL conflicts
* Automatic patching for known NVIDIA/Transformers compatibility issues
* Fault-handler logging for low-level crashes

## 🧠 Model

The project uses:

**Model:** `nvidia/LocateAnything-3B`

The model is loaded using Hugging Face Transformers with `trust_remote_code=True`.

The implementation supports three GPU memory modes:

| Mode      | Approx. GPU Memory | Description                                    |
| --------- | -----------------: | ---------------------------------------------- |
| `full`    |           ~6.5 GB+ | Full bfloat16 model on GPU                     |
| `offload` |             ~3 GB+ | Automatically splits model between GPU and CPU |
| `4bit`    |           ~2.5 GB+ | NF4 4-bit quantized model                      |

The default `auto` mode selects the strategy according to available VRAM.

### Automatic VRAM Selection

```text
Available VRAM

       │
       ▼
   ≥ 6.5 GB
       │
       ├──► FULL bfloat16
       │
       ▼
   ≥ 3.5 GB
       │
       ├──► CPU OFFLOAD
       │
       ▼
    < 3.5 GB
       │
       └──► 4-BIT NF4
```

## 🎯 Supported Tasks

### 1. Object Detection

Detect multiple categories from a comma-separated query.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task detect \
    --query "person, cup, laptop"
```

### 2. Single-Object Grounding

Locate one specific object.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task ground_single \
    --query "red cup"
```

### 3. Multi-Object Grounding

Locate all objects matching a description.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task ground_multi \
    --query "people wearing black shirts"
```

### 4. Text Detection

Detect text regions in the scene.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task text
```

### 5. GUI Grounding

Locate graphical interface elements.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task gui \
    --query "search button"
```

For point output:

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task gui \
    --gui_type point \
    --query "close button"
```

### 6. Point Localization

Return a point corresponding to the requested object.

```bash
python3 locate_anything_realtime.py \
    --mode webcam \
    --task point \
    --query "person's face"
```

## 🖼️ Image Mode

You can run inference on a single image instead of a webcam.

```bash
python3 locate_anything_realtime.py \
    --mode image \
    --image photo.jpg \
    --query "red car"
```

The program prints the raw model output and saves the annotated result as:

```text
photo_located.jpg
```

The output image contains the detected bounding boxes or points.

## 📷 Webcam Mode

Run the default real-time webcam application:

```bash
python3 locate_anything_realtime.py
```

The default configuration uses:

```text
Camera       : 0
Resolution   : 1280 × 720
Task         : detect
Query        : person, cup, laptop
Load mode    : auto
Skip frames  : 5
Inference size: 640
```

The webcam implementation processes only selected frames rather than performing inference on every camera frame. This reduces GPU memory pressure and computational load.

## ⌨️ Controls

| Key     | Action         |
| ------- | -------------- |
| `Q`     | Quit           |
| `SPACE` | Pause / Resume |

The display also shows:

* Current model/loading mode
* Current task
* Query
* FPS
* Inference time
* Number of detections

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Install a PyTorch build compatible with your NVIDIA CUDA environment.

For the CUDA 12.1 wheel:

```bash
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python3 locate_anything_realtime.py
```

On the first run, Hugging Face will download the required model files.

## 📦 Requirements

The project requires Python, PyTorch, NumPy, OpenCV, Pillow, Transformers, Decord, PEFT, LMDB, Accelerate, BitsAndBytes, Safetensors, and Matplotlib. The source specifically requires Transformers `4.57.1` and NumPy `>=2.2`.

See [`requirements.txt`](requirements.txt) for the complete dependency list.

## 🖥️ Hardware

A CUDA-capable NVIDIA GPU is recommended.

Approximate GPU memory requirements:

```text
6.5 GB+ VRAM  → Full bfloat16
3.5 GB+ VRAM  → CPU offload
2.5 GB+ VRAM  → 4-bit NF4
```

Lower-VRAM GPUs should use `4bit` or `offload`.

For example:

```bash
python3 locate_anything_realtime.py --load_mode 4bit
```

## 🚀 Low-VRAM Optimization

If you experience CUDA out-of-memory errors, try:

### Use 4-bit quantization

```bash
python3 locate_anything_realtime.py --load_mode 4bit
```

### Use CPU offloading

```bash
python3 locate_anything_realtime.py --load_mode offload
```

### Reduce inference resolution

```bash
python3 locate_anything_realtime.py \
    --infer_size 480
```

### Process fewer frames

```bash
python3 locate_anything_realtime.py \
    --skip_frames 10
```

You can combine these options:

```bash
python3 locate_anything_realtime.py \
    --load_mode 4bit \
    --infer_size 480 \
    --skip_frames 10
```

The program also automatically retries generation with a smaller token budget when a CUDA OOM occurs during inference.

## 🔧 Command-Line Arguments

| Argument        | Options / Type                                                    | Default               | Description                       |
| --------------- | ----------------------------------------------------------------- | --------------------- | --------------------------------- |
| `--mode`        | `webcam`, `image`                                                 | `webcam`              | Inference source                  |
| `--task`        | `detect`, `ground_single`, `ground_multi`, `text`, `gui`, `point` | `detect`              | Vision task                       |
| `--query`       | String                                                            | `person, cup, laptop` | Object/text query                 |
| `--image`       | Path                                                              | Empty                 | Input image for image mode        |
| `--load_mode`   | `auto`, `full`, `offload`, `4bit`                                 | `auto`                | Model loading strategy            |
| `--camera`      | Integer                                                           | `0`                   | Camera index                      |
| `--width`       | Integer                                                           | `1280`                | Webcam width                      |
| `--height`      | Integer                                                           | `720`                 | Webcam height                     |
| `--flip`        | Flag                                                              | Off                   | Mirror webcam image               |
| `--skip_frames` | Integer                                                           | `5`                   | Run inference every N frames      |
| `--infer_size`  | Integer                                                           | `640`                 | Maximum inference image dimension |
| `--gui_type`    | `box`, `point`                                                    | `box`                 | GUI localization output           |
| `--model`       | Model ID/path                                                     | LocateAnything-3B     | Hugging Face model                |

These options are defined directly by the project's CLI parser.

## 🧪 Example Commands

### Default webcam

```bash
python3 locate_anything_realtime.py
```

### Webcam with mirrored output

```bash
python3 locate_anything_realtime.py \
    --flip
```

### Detect objects

```bash
python3 locate_anything_realtime.py \
    --task detect \
    --query "person, phone, laptop"
```

### Ground a specific object

```bash
python3 locate_anything_realtime.py \
    --task ground_single \
    --query "blue bottle"
```

### Ground multiple objects

```bash
python3 locate_anything_realtime.py \
    --task ground_multi \
    --query "chairs"
```

### Detect text

```bash
python3 locate_anything_realtime.py \
    --task text
```

### Image inference

```bash
python3 locate_anything_realtime.py \
    --mode image \
    --image test.jpg \
    --task detect \
    --query "person, car"
```

### Force 4-bit mode

```bash
python3 locate_anything_realtime.py \
    --load_mode 4bit
```

### Force CPU offload

```bash
python3 locate_anything_realtime.py \
    --load_mode offload
```

## 🛠️ Stability Fixes

This implementation contains several compatibility and stability workarounds.

### Transformers `rope_theta`

A compatibility patch injects the expected `rope_theta` value into `Qwen2Config`:

```text
rope_theta = 1,000,000
```

This addresses compatibility with the Qwen2 configuration used by LocateAnything.

### NVIDIA cache patch

The program scans the cached NVIDIA model source and replaces problematic `torch.tensor(existing_tensor, ...)` patterns with safer tensor cloning/conversion operations.

It also removes stale `.pyc` files and can update already-loaded functions.

### OpenCV OpenCL

OpenCV OpenCL is disabled to reduce GPU-context conflicts with PyTorch:

```python
OPENCV_OPENCL_DEVICE=disabled
```

### CUDA memory allocator

The project enables expandable CUDA memory segments:

```python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Matplotlib display

Instead of `cv2.imshow()`, the project uses Matplotlib with the `TkAgg` backend. This is intended to avoid OpenCV GUI/OpenGL conflicts that can occur with NVIDIA drivers and certain Linux OpenCV environments.

## 🐛 Troubleshooting

### CUDA Out of Memory

Try:

```bash
python3 locate_anything_realtime.py \
    --load_mode 4bit \
    --infer_size 480 \
    --skip_frames 10
```

### Camera unavailable

Try another camera index:

```bash
python3 locate_anything_realtime.py --camera 1
```

### Black/unstable display

The application uses Matplotlib/TkAgg rather than `cv2.imshow()`. Make sure Tk is installed on your Linux system.

For Ubuntu:

```bash
sudo apt install python3-tk
```

### Transformers cache issues

The original setup recommends removing the cached NVIDIA Transformers module before rerunning:

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/
```

The program itself also performs runtime patching of cached NVIDIA model code.

### NumPy / Python 3.13 issue

For Python 3.13, use:

```bash
pip install "numpy>=2.2"
```

The script explicitly checks the NumPy version when running under Python 3.13.

### Fault/segmentation errors

A fault-handler log is created automatically:

```text
/tmp/locate_anything_fault.log
```

This can help identify low-level CUDA or C/C++ crashes.

## 📁 Suggested Project Structure

```text
locate-anything-realtime/
│
├── locate_anything_realtime.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
│   └── test.jpg
│
└── outputs/
    └── test_located.jpg
```

## 🔒 GitHub Recommendations

Do not commit:

```text
.venv/
__pycache__/
*.pyc
*.log
*.pt
*.pth
*.bin
*.safetensors
outputs/
```

Model weights should normally be downloaded through Hugging Face rather than committed directly to the repository.

## 📜 License

This project contains an application built around the NVIDIA `LocateAnything-3B` model. 

## 🙏 Acknowledgements

* NVIDIA / NVLabs — LocateAnything
* Hugging Face Transformers
* PyTorch
* OpenCV
* Pillow
* Matplotlib
* BitsAndBytes
* Accelerate

## 📌 Project Summary

This project demonstrates how a large vision-language grounding model can be adapted for **real-time visual localization on constrained NVIDIA GPUs**.

The main engineering focus is not only model inference, but also practical deployment:

```text
Camera / Image
      ↓
Frame preprocessing
      ↓
VRAM-aware model loading
      ↓
LocateAnything-3B
      ↓
Text-based visual query
      ↓
Bounding boxes / points
      ↓
Real-time visualization
```

The implementation is particularly useful for experimenting with **Physical AI, robotics perception, visual grounding, human-robot interaction, and real-time object localization**.
