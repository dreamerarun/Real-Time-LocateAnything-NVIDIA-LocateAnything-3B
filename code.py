#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Real-Time LocateAnything — NVIDIA Eagle / NVlabs            ║
║  Model : nvidia/LocateAnything-3B                            ║
║  Fixes : rope_theta | dtype= | numpy 3.13 | OOM / 6GB GPU   ║
╚══════════════════════════════════════════════════════════════╝

This version adds three-tier GPU memory management so it works
on GPUs with as little as 4 GB VRAM:

  Tier 1 – Full bfloat16 on GPU        (~6.5 GB)  fastest
  Tier 2 – device_map="auto" offload   (~3 GB GPU + RAM)
  Tier 3 – 4-bit NF4 quantization      (~2.5 GB GPU)  slowest

The tier is chosen automatically, or forced with --load_mode.

Install:
  pip install "numpy>=2.2" "Pillow>=10" opencv-python \\
      "transformers==4.57.1" decord2 peft lmdb accelerate \\
      bitsandbytes safetensors
  rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/

Usage:
  python3 locate_anything_realtime.py
  python3 locate_anything_realtime.py --load_mode 4bit
  python3 locate_anything_realtime.py --load_mode offload
  python3 locate_anything_realtime.py --mode image --image photo.jpg --query "red car"
"""

import re, sys, time, argparse, textwrap, warnings
from pathlib import Path
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable OpenCV's OpenCL to prevent GPU context conflicts with PyTorch
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
os.environ["OPENCV_OPENCL_RUNTIME"] = ""

# faulthandler prints a C-level traceback on segfault — tells us the exact
# CUDA kernel / C++ function that crashes, not just "Segmentation fault".
import faulthandler as _fh, tempfile as _tf, atexit as _at
_fh_log = open(os.path.join(_tf.gettempdir(), "locate_anything_fault.log"), "w")
_fh.enable(file=_fh_log)
_at.register(_fh_log.close)
print(f"[INFO] faulthandler log → {_fh_log.name}")

# ══════════════════════════════════════════════════════════════════════════════
#  EARLY ON-DISK PATCH: generate_utils.py segfault
#  The HuggingFace cache already exists on disk after the first model download.
#  Patch the file NOW, before any transformers import loads it into memory.
# ══════════════════════════════════════════════════════════════════════════════
# ── Substitution table: all known-bad torch.tensor(tensor,...) calls ────────
# torch.tensor() called on an existing tensor causes undefined behaviour in
# PyTorch >= 2.1 + CUDA 12 and reliably segfaults on RTX 40-series laptops.
# The safe replacement is always:  var.detach().clone().to(dtype=..., device=...)
_BAD_PATTERNS = [
    # generate_utils.py  (confirmed in earlier run)
    ("torch.tensor(out_ref, dtype=x0.dtype, device=x0.device)",
     "out_ref.detach().clone().to(dtype=x0.dtype, device=x0.device)"),
    # modeling_locateanything.py — common variants seen in NVlabs VLMs
    ("torch.tensor(x, dtype=",
     "x.detach().clone().to(dtype="),
    ("torch.tensor(hidden_states,",
     "hidden_states.detach().clone().to("),
    ("torch.tensor(attention_mask,",
     "attention_mask.detach().clone().to("),
    ("torch.tensor(position_ids,",
     "position_ids.detach().clone().to("),
    ("torch.tensor(residual,",
     "residual.detach().clone().to("),
    ("torch.tensor(query_states,",
     "query_states.detach().clone().to("),
    ("torch.tensor(key_states,",
     "key_states.detach().clone().to("),
    ("torch.tensor(value_states,",
     "value_states.detach().clone().to("),
]

def _patch_nvidia_cache(report_unchanged=False):
    """
    Stage 1 — disk: scan EVERY .py in the nvidia model cache, apply all
    substitutions from _BAD_PATTERNS, and delete ALL sibling .pyc files.

    Stage 2 — bytecode: for every already-loaded module whose source file
    is inside the nvidia cache, recompile the (now-patched) source and
    transplant new code objects into the live function objects.

    Safe to call multiple times; idempotent.
    """
    import glob, pathlib, importlib.util, types, inspect

    hf_root = pathlib.Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    all_py   = glob.glob(str(hf_root / "**" / "*.py"), recursive=True)

    # ── Stage 1: disk ────────────────────────────────────────────────────
    for fp in all_py:
        p    = pathlib.Path(fp)
        orig = p.read_text(encoding="utf-8", errors="replace")
        text = orig
        for bad, good in _BAD_PATTERNS:
            text = text.replace(bad, good)
        if text != orig:
            p.write_text(text, encoding="utf-8")
            print(f"[PATCH] {p.name} source patched ✓")
        # Nuke every .pyc so none can shadow the patched source
        pycache = p.parent / "__pycache__"
        for pyc in pycache.glob(f"{p.stem}*.pyc"):
            try: pyc.unlink()
            except OSError: pass
        try:
            canonical = pathlib.Path(importlib.util.cache_from_source(fp))
            if canonical.exists(): canonical.unlink()
        except Exception: pass

    # ── Stage 2: in-memory bytecode ──────────────────────────────────────
    # Build a set of canonical source paths we just patched for quick lookup
    patched_paths = {str(pathlib.Path(fp).resolve()) for fp in all_py}

    total_fns = 0
    for key, mod in list(sys.modules.items()):
        try:
            src_file = inspect.getfile(mod)
        except (TypeError, OSError):
            continue
        if str(pathlib.Path(src_file).resolve()) not in patched_paths:
            continue
        p = pathlib.Path(src_file)
        if not p.exists():
            continue
        good_src = p.read_text(encoding="utf-8", errors="replace")
        try:
            new_code = compile(good_src, src_file, "exec")
        except SyntaxError:
            continue

        def _collect(co):
            out = {co.co_name: co}
            for c in co.co_consts:
                if isinstance(c, types.CodeType):
                    out.update(_collect(c))
            return out

        new_codes = _collect(new_code)
        mod_fns = 0
        for attr in list(vars(mod)):
            obj = getattr(mod, attr, None)
            if isinstance(obj, types.FunctionType):
                nm = obj.__code__.co_name
                if nm in new_codes:
                    obj.__code__ = new_codes[nm]
                    mod_fns += 1
        if mod_fns:
            print(f"[PATCH] {p.name} in-memory: {mod_fns} function(s) ✓")
            total_fns += mod_fns

    if total_fns:
        print(f"[PATCH] Total in-memory functions patched: {total_fns}")

# Alias used at both call sites
_patch_generate_utils = _patch_nvidia_cache

_patch_generate_utils()

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def _die(msg): print(f"\n[ERROR] {msg}"); sys.exit(1)

try:
    import torch
except ImportError:
    _die("PyTorch not found.\n  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

try:
    import numpy as np
    maj, mn = (int(x) for x in np.__version__.split(".")[:2])
    if sys.version_info >= (3, 13) and (maj < 2 or (maj == 2 and mn < 1)):
        _die(f"numpy {np.__version__} doesn't support Python 3.13.\n  pip install 'numpy>=2.2'")
except ImportError:
    _die("numpy not found.  pip install 'numpy>=2.2'")

try:
    import cv2
except ImportError:
    _die("OpenCV not found.  pip install opencv-python")

try:
    from PIL import Image
except ImportError:
    _die("Pillow not found.  pip install 'Pillow>=10'")

try:
    import transformers
    from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
    from transformers import BitsAndBytesConfig
except ImportError as e:
    _die(f"transformers error: {e}\n  pip install 'transformers==4.57.1'")

# decord2 is the Python-3.13-compatible fork of decord
try:
    import decord                              # original
except ImportError:
    try:
        import decord2 as decord               # Py-3.13 fork
    except ImportError:
        _die("decord not found.  pip install decord2")

print(f"[INFO] Python {sys.version.split()[0]} | "
      f"transformers {transformers.__version__} | "
      f"numpy {np.__version__} | torch {torch.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
#  GPU MEMORY PROBE
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def _free_vram_gb() -> float:
    if DEVICE != "cuda":
        return 0.0
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    return free / 1024**3

def _total_vram_gb() -> float:
    if DEVICE != "cuda":
        return 0.0
    _, total = torch.cuda.mem_get_info(0)
    return total / 1024**3

def _pick_load_mode(requested: str) -> str:
    """
    Auto-pick loading strategy based on available VRAM.
      full    → bfloat16 entirely on GPU  (needs ≥6.5 GB free)
      offload → device_map="auto", CPU offload for overflow
      4bit    → 4-bit NF4 quant via bitsandbytes (~2.5 GB)
    """
    if requested != "auto":
        return requested
    free = _free_vram_gb()
    total = _total_vram_gb()
    print(f"[INFO] GPU VRAM: {total:.1f} GB total, {free:.1f} GB free")
    if free >= 6.5:
        return "full"
    elif free >= 3.5:
        return "offload"
    else:
        return "4bit"

MODEL_ID = "nvidia/LocateAnything-3B"

# Drawing constants
PALETTE = [
    (0, 230, 118), (0, 176, 255), (255,  82,  82), (255, 196,   0),
    (171,  71, 188),(  0, 191, 165),(255, 109,   0),(233,  30,  99),
    ( 63,  81, 181),(  0, 150, 136),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FSCALE = 0.55
BOX_TH = 2
PAD    = 4

# ══════════════════════════════════════════════════════════════════════════════
#  FIX: rope_theta monkey-patch
#  modeling_qwen2.py line 242:  self.rope_theta = config.rope_theta
#  → raises AttributeError in newer transformers because Qwen2Config
#    doesn't declare rope_theta as a class-level default.
# ══════════════════════════════════════════════════════════════════════════════
_ROPE_DEFAULT = 1_000_000.0   # Qwen2.5-3B actual default

def _patch_rope_theta():
    """Inject rope_theta into every Qwen2Config found in sys.modules."""
    for key, mod in list(sys.modules.items()):
        if "nvidia" not in key and "LocateAnything" not in key.replace("-","_"):
            continue
        cls = getattr(mod, "Qwen2Config", None)
        if cls is None or getattr(cls, "_rope_patched", False):
            continue
        orig_ga = cls.__dict__.get("__getattr__")
        def _ga(self, name, _o=orig_ga):
            if name == "rope_theta":
                return _ROPE_DEFAULT
            if _o: return _o(self, name)
            raise AttributeError(name)
        cls.__getattr__ = _ga
        cls._rope_patched = True
        try: cls.rope_theta = _ROPE_DEFAULT
        except Exception: pass
        print(f"[PATCH] rope_theta → {key}.Qwen2Config ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER  (three-tier VRAM strategy)
# ══════════════════════════════════════════════════════════════════════════════
def _load_model(model_path: str, load_mode: str):
    """
    Load the model using the requested strategy.

    load_mode:
      "full"    – standard bfloat16, entire model on GPU
      "offload" – device_map="auto", overflow to CPU RAM (needs accelerate)
      "4bit"    – 4-bit NF4 quantization (needs bitsandbytes)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*torch_dtype.*deprecated.*")
        warnings.filterwarnings("ignore", ".*GenerationMixin.*")
        warnings.filterwarnings("ignore", ".*use_fast.*")

        # Trigger custom-code download + patch rope_theta
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        _patch_rope_theta()

        # Ensure rope_theta on the config objects themselves
        for sub in [cfg, getattr(cfg, "text_config", None)]:
            if sub is None: continue
            try:
                if not hasattr(sub, "rope_theta"):
                    object.__setattr__(sub, "rope_theta", _ROPE_DEFAULT)
            except Exception: pass

        common_kw = dict(config=cfg, trust_remote_code=True)

        # ── Tier 1: full bfloat16 on GPU ─────────────────────────────────
        if load_mode == "full":
            print(f"[LOAD] Mode: full bfloat16 on {DEVICE}")
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    dtype=DTYPE,                       # not torch_dtype= !
                    **common_kw,
                )
                return model.to(DEVICE).eval()
            except torch.OutOfMemoryError:
                print("[WARN] OOM in full mode → falling back to offload")
                torch.cuda.empty_cache()
                load_mode = "offload"

        # ── Tier 2: device_map="auto" CPU offload ─────────────────────────
        if load_mode == "offload":
            print("[LOAD] Mode: device_map=auto (CPU offload enabled)")
            try:
                import accelerate  # noqa – just verify it's installed
            except ImportError:
                _die("accelerate not found (needed for --load_mode offload).\n"
                     "  pip install accelerate")
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    dtype=DTYPE,
                    device_map="auto",                 # auto CPU/GPU split
                    **common_kw,
                )
                return model.eval()
            except torch.OutOfMemoryError:
                print("[WARN] OOM in offload mode → falling back to 4bit")
                torch.cuda.empty_cache()
                load_mode = "4bit"

        # ── Tier 3: 4-bit NF4 quantization ───────────────────────────────
        if load_mode == "4bit":
            print("[LOAD] Mode: 4-bit NF4 quantization (bitsandbytes)")
            try:
                import bitsandbytes  # noqa
            except ImportError:
                _die("bitsandbytes not found (needed for --load_mode 4bit).\n"
                     "  pip install bitsandbytes")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModel.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                device_map="auto",                     # required for bnb
                **common_kw,
            )
            return model.eval()

        raise ValueError(f"Unknown load_mode: {load_mode}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL WORKER
# ══════════════════════════════════════════════════════════════════════════════
class LocateAnythingWorker:
    def __init__(self, model_path=MODEL_ID, load_mode="auto"):
        self.load_mode = _pick_load_mode(load_mode)
        # For 4bit / offload the model manages its own device placement
        self.infer_device = DEVICE

        print("[INFO] Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("[INFO] Loading processor …")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*use_fast.*")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"[INFO] Loading model (mode={self.load_mode}) …")
        t0 = time.time()
        self.model = _load_model(model_path, self.load_mode)
        elapsed = time.time() - t0

        # Stage 2: patch any generate_utils bytecode loaded from stale .pyc
        _patch_generate_utils()

        # Report actual VRAM used
        if DEVICE == "cuda":
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[INFO] ✅ Model ready in {elapsed:.1f}s  |  GPU VRAM used: {used:.2f} GB\n")
        else:
            print(f"[INFO] ✅ Model ready in {elapsed:.1f}s  (CPU mode)\n")

    # ── inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image: Image.Image, question: str,
                gen_mode="hybrid", max_tokens=256,
                temperature=0.7, verbose=False) -> str:
        """
        Run one forward + generate pass.

        OOM guard: if CUDA OOM is hit, empty cache and retry once with
        a smaller max_tokens budget.  A second OOM re-raises so the
        webcam loop can log [WARN] and skip the frame gracefully.
        """
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
        ]}]
        text      = self.processor.py_apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
        imgs, _   = self.processor.process_vision_info(messages)

        # Move inputs to whatever device the model's first parameter lives on
        model_device = next(self.model.parameters()).device

        # Always clear the CUDA cache before building KV-cache / attention
        # tensors: reclaims fragmented allocations from the previous frame.
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

        inputs = self.processor(text=[text], images=imgs,
                                return_tensors="pt").to(model_device)

        # Keep bfloat16 for pixel_values — the model was trained in bfloat16
        # and float16 causes silent NaN in the decoder on some CUDA kernels.
        pv_dtype = torch.bfloat16 if model_device.type == "cuda" else torch.float32
        pv  = inputs["pixel_values"].to(pv_dtype)
        iid = inputs["input_ids"]
        ghw = inputs.get("image_grid_hws", None)

        def _generate(tok_budget):
            return self.model.generate(
                pixel_values       = pv,
                input_ids          = iid,
                attention_mask     = inputs["attention_mask"],
                image_grid_hws     = ghw,
                tokenizer          = self.tokenizer,
                max_new_tokens     = tok_budget,
                use_cache          = True,
                generation_mode    = gen_mode,
                temperature        = temperature,
                do_sample          = True,
                top_p              = 0.9,
                repetition_penalty = 1.1,
                verbose            = verbose,
            )

        try:
            response = _generate(max_tokens)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"[WARN] OOM during generate (budget={max_tokens}), "
                  "clearing cache and retrying with budget=128 ...")
            torch.cuda.empty_cache()
            response = _generate(128)   # minimal budget -- enough for bbox coords

        # Synchronize CUDA before returning — ensures all GPU ops are
        # complete before OpenCV's display pipeline touches the framebuffer.
        # Without this, cv2.imshow() can segfault on Linux + NVIDIA drivers
        # due to a race between PyTorch CUDA streams and OpenGL.
        if model_device.type == "cuda":
            torch.cuda.synchronize()

        return response[0] if isinstance(response, tuple) else response

    # ── task helpers ──────────────────────────────────────────────────────────
    def detect(self, img, cats, **kw):
        return self.predict(img,
            f"Locate all the instances that matches the following description: {'</c>'.join(cats)}.", **kw)
    def ground_single(self, img, phrase, **kw):
        return self.predict(img, f"Locate a single instance that matches the following description: {phrase}.", **kw)
    def ground_multi(self, img, phrase, **kw):
        return self.predict(img, f"Locate all the instances that match the following description: {phrase}.", **kw)
    def detect_text(self, img, **kw):
        return self.predict(img, "Detect all the text in box format.", **kw)
    def ground_gui(self, img, phrase, output_type="box", **kw):
        if output_type == "point":
            return self.predict(img, f"Point to: {phrase}.", **kw)
        return self.predict(img, f"Locate the region that matches the following description: {phrase}.", **kw)
    def point(self, img, phrase, **kw):
        return self.predict(img, f"Point to: {phrase}.", **kw)

    # ── parse model output ─────────────────────────────────────────────────────
    @staticmethod
    def parse_boxes(answer, W, H):
        return [
            {"x1":int(int(a)/1000*W), "y1":int(int(b)/1000*H),
             "x2":int(int(c)/1000*W), "y2":int(int(d)/1000*H)}
            for a,b,c,d in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
        ]
    @staticmethod
    def parse_points(answer, W, H):
        return [
            {"x":int(int(x)/1000*W), "y":int(int(y)/1000*H)}
            for x,y in re.findall(r"<box><(\d+)><(\d+)></box>", answer)
        ]


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_boxes(frame, boxes, labels=None, ci=0):
    for i,b in enumerate(boxes):
        col = PALETTE[(ci+i)%len(PALETTE)]
        lbl = (labels[i] if labels and i<len(labels) else f"obj {i+1}")
        cv2.rectangle(frame,(b["x1"],b["y1"]),(b["x2"],b["y2"]),col,BOX_TH)
        (tw,th),bl = cv2.getTextSize(lbl,FONT,FSCALE,1)
        lx,ly = b["x1"], b["y1"]-th-PAD
        if ly<0: ly = b["y2"]+th+PAD
        cv2.rectangle(frame,(lx,ly-th-PAD),(lx+tw+PAD*2,ly+bl),col,-1)
        cv2.putText(frame,lbl,(lx+PAD,ly),FONT,FSCALE,(0,0,0),1,cv2.LINE_AA)

def draw_points(frame, pts, ci=0):
    for i,p in enumerate(pts):
        col=PALETTE[(ci+i)%len(PALETTE)]
        cx,cy=p["x"],p["y"]
        cv2.circle(frame,(cx,cy),10,col,-1)
        cv2.circle(frame,(cx,cy),12,(255,255,255),2)
        cv2.line(frame,(cx-18,cy),(cx+18,cy),col,2)
        cv2.line(frame,(cx,cy-18),(cx,cy+18),col,2)

def draw_hud(frame, fps, task, query, n, ms, load_mode):
    h,w=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,110),(15,15,15),-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    rows=[
        (f"NVIDIA LocateAnything-3B  [{load_mode.upper()} mode]",  (0,200,255)),
        (f"Task: {task.upper()}  |  Query: {textwrap.shorten(query,50)}", (210,210,210)),
        (f"FPS {fps:.1f}  |  Inference {ms:.0f}ms  |  Detections {n}", (150,150,150)),
    ]
    for i,(t,c) in enumerate(rows):
        cv2.putText(frame,t,(12,26+i*28),FONT,0.60,c,1,cv2.LINE_AA)
    cv2.putText(frame,"Q=quit  SPACE=pause",(w-210,h-10),FONT,0.45,(90,90,90),1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(worker, pil_img, args, display_wh=None):
    """
    display_wh : (W, H) of the frame that will be drawn on.
    When set, bounding-box coordinates are scaled back to display size
    even though inference runs on a smaller resized image.
    """
    W, H  = display_wh if display_wh else pil_img.size
    task  = args.task
    query = args.query or ""
    cats  = [c.strip() for c in query.split(",") if c.strip()] or ["object"]
    t0    = time.time()

    if task=="detect":
        ans=worker.detect(pil_img,cats)
        boxes=worker.parse_boxes(ans,W,H)
        labels=[cats[i%len(cats)] for i in range(len(boxes))]
        pts=[]
    elif task=="ground_single":
        ans=worker.ground_single(pil_img,query)
        boxes=worker.parse_boxes(ans,W,H)[:1]; labels=[query]*len(boxes); pts=[]
    elif task=="ground_multi":
        ans=worker.ground_multi(pil_img,query)
        boxes=worker.parse_boxes(ans,W,H); labels=[query]*len(boxes); pts=[]
    elif task=="text":
        ans=worker.detect_text(pil_img)
        boxes=worker.parse_boxes(ans,W,H)
        labels=[f"text {i+1}" for i in range(len(boxes))]; pts=[]
    elif task=="gui":
        ot=getattr(args,"gui_type","box")
        ans=worker.ground_gui(pil_img,query,output_type=ot)
        if ot=="point": boxes=[]; pts=worker.parse_points(ans,W,H); labels=[]
        else: boxes=worker.parse_boxes(ans,W,H); labels=[query]*len(boxes); pts=[]
    elif task=="point":
        ans=worker.point(pil_img,query)
        boxes=[]; pts=worker.parse_points(ans,W,H); labels=[]
    else:
        raise ValueError(f"Unknown task: {task}")

    return ans, boxes, pts, labels, (time.time()-t0)*1000


# ══════════════════════════════════════════════════════════════════════════════
#  WEBCAM LOOP
# ══════════════════════════════════════════════════════════════════════════════
# ── Display backend ──────────────────────────────────────────────────────────
# We use matplotlib (TkAgg backend) instead of cv2.imshow.
# Reason: on Ubuntu 22/24 with NVIDIA drivers + conda OpenCV, cv2.namedWindow
# and cv2.imshow segfault because the Qt5/GTK plugin in the conda OpenCV build
# conflicts with the system display stack.  Matplotlib/TkAgg uses Tk which is
# always available and never touches OpenGL or the CUDA GPU context.
import matplotlib
matplotlib.use("TkAgg")          # pure-Tk, no OpenGL/EGL
import matplotlib.pyplot as _plt
import matplotlib.animation as _anim


class _Display:
    """Lightweight matplotlib window for showing annotated webcam frames."""
    def __init__(self, title="LocateAnything"):
        self.fig, self.ax = _plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title(title)
        self.fig.tight_layout(pad=0)
        self.ax.axis("off")
        self._im = None
        self._closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        _plt.ion()
        _plt.show(block=False)

    def _on_close(self, _event):
        self._closed = True

    def show(self, bgr_frame):
        """Display a BGR OpenCV frame. Returns False if window was closed."""
        if self._closed:
            return False
        rgb = bgr_frame[:, :, ::-1]          # BGR → RGB
        if self._im is None:
            self._im = self.ax.imshow(rgb)
        else:
            self._im.set_data(rgb)
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self._closed = True
            return False
        return True

    def key_pressed(self):
        """Return the last key pressed (lowercase str) or empty string."""
        try:
            # matplotlib stores the last key in the canvas key event
            k = getattr(self.fig.canvas, "_last_key", "")
            self.fig.canvas._last_key = ""
            return k
        except Exception:
            return ""


def _connect_key(fig):
    """Wire up key-press storage on the matplotlib figure."""
    def _on_key(event):
        fig.canvas._last_key = event.key or ""
    fig.canvas._last_key = ""
    fig.canvas.mpl_connect("key_press_event", _on_key)


def webcam_loop(worker, args):
    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Camera {args.camera} unavailable. Try --camera 1")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)

    display = _Display("LocateAnything — NVIDIA  [Q=quit  SPACE=pause]")
    _connect_key(display.fig)

    print("[INFO] Webcam running. Q=quit  SPACE=pause/resume\n")

    fps_cnt=0; fps_t=time.time(); fps=0.0; paused=False
    lb=[]; lp=[]; ll=[]; lms=0.0; nd=0; fn=0

    while True:
        ret,bgr=cap.read()
        if not ret: time.sleep(0.05); continue
        if args.flip: bgr=cv2.flip(bgr,1)

        _k = display.key_pressed()
        if _k == "q": break
        if _k == " ": paused=not paused; print("[PAUSE]" if paused else "[RESUME]")

        if not paused and fn%max(1,args.skip_frames)==0:
            pil=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            # ── Resize for inference to cap VRAM usage ──────────────────────
            iw, ih = pil.size
            longest = max(iw, ih)
            if longest > args.infer_size:
                scale = args.infer_size / longest
                pil = pil.resize(
                    (max(1, int(iw * scale)), max(1, int(ih * scale))),
                    Image.BILINEAR
                )
            # ───────────────────────────────────────────────────────────────
            try:
                dh, dw = bgr.shape[:2]
                _,lb,lp,ll,lms=run_inference(worker,pil,args, display_wh=(dw, dh))
                nd=len(lb)+len(lp)
            except Exception as ex:
                print(f"[WARN] {ex}")

        vis=bgr.copy()
        if lb: draw_boxes(vis,lb,ll)
        if lp: draw_points(vis,lp)

        fps_cnt+=1
        if time.time()-fps_t>=1.0:
            fps=fps_cnt/(time.time()-fps_t); fps_cnt=0; fps_t=time.time()

        draw_hud(vis,fps,args.task,args.query or "",nd,lms,worker.load_mode)
        if paused:
            cv2.putText(vis,"PAUSED",(vis.shape[1]//2-55,vis.shape[0]//2),
                        FONT,1.5,(0,230,255),3,cv2.LINE_AA)
        if not display.show(vis):
            print("[INFO] Display window closed — exiting.")
            break
        fn+=1

    cap.release()
    _plt.close(display.fig)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
def image_mode(worker, args):
    path=Path(args.image)
    if not path.exists(): print(f"[ERROR] Not found: {path}"); sys.exit(1)
    pil=Image.open(path).convert("RGB"); W,H=pil.size
    bgr=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
    print(f"[INFO] Running on '{path.name}' ({W}×{H}) …")

    ans,boxes,pts,labels,ms=run_inference(worker,pil,args)
    print(f"\n── RAW OUTPUT ──────────\n{ans}\n")
    print(f"  Boxes  : {len(boxes)}\n  Points : {len(pts)}\n  Time   : {ms:.0f} ms\n")

    vis=bgr.copy()
    if boxes: draw_boxes(vis,boxes,labels)
    if pts:   draw_points(vis,pts)
    draw_hud(vis,0,args.task,args.query or "",len(boxes)+len(pts),ms,worker.load_mode)

    out=path.stem+"_located.jpg"
    cv2.imwrite(out,vis)
    print(f"[SAVED] {out}")
    # Use matplotlib so we don't hit the cv2.imshow segfault
    _d = _Display("LocateAnything Result — press Q or close to exit")
    _connect_key(_d.fig)
    print("[INFO] Close the window or press Q to exit …")
    while not _d._closed:
        _d.show(vis)
        if _d.key_pressed() == "q":
            break
    _plt.close(_d.fig)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap=argparse.ArgumentParser(
        description="Real-Time LocateAnything by NVIDIA — with VRAM-aware loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 locate_anything_realtime.py                            # auto VRAM mode
          python3 locate_anything_realtime.py --load_mode 4bit          # force 4-bit (any GPU)
          python3 locate_anything_realtime.py --load_mode offload       # force CPU offload
          python3 locate_anything_realtime.py --mode webcam --query "person, cup"
          python3 locate_anything_realtime.py --mode webcam --task text
          python3 locate_anything_realtime.py --mode image --image photo.jpg --query "red car"
        """))
    ap.add_argument("--mode",       choices=["webcam","image"], default="webcam")
    ap.add_argument("--task",       default="detect",
                    choices=["detect","ground_single","ground_multi","text","gui","point"])
    ap.add_argument("--query",      default="person, cup, laptop")
    ap.add_argument("--image",      default="")
    ap.add_argument("--load_mode",  default="auto",
                    choices=["auto","full","offload","4bit"],
                    help="VRAM strategy: auto (default), full, offload, 4bit")
    ap.add_argument("--camera",     type=int, default=0)
    ap.add_argument("--width",      type=int, default=1280)
    ap.add_argument("--height",     type=int, default=720)
    ap.add_argument("--flip",       action="store_true")
    ap.add_argument("--skip_frames",type=int, default=5,
                    help="Run inference every N frames (default 5, increase to reduce VRAM pressure)")
    ap.add_argument("--infer_size", type=int, default=640,
                    help="Resize the longest edge of the frame to this before inference (default 640). "
                         "Lower = less VRAM. Try 480 or 512 if you still get OOM.")
    ap.add_argument("--gui_type",   default="box", choices=["box","point"])
    ap.add_argument("--model",      default=MODEL_ID)
    return ap.parse_args()


def main():
    args=parse_args()
    free=_free_vram_gb(); total=_total_vram_gb()
    print("="*64)
    print("  NVIDIA LocateAnything-3B  |  Real-Time Inference")
    print(f"  GPU : {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
    if DEVICE=="cuda":
        print(f"  VRAM: {total:.1f} GB total  |  {free:.1f} GB free")
    print(f"  Mode: {args.mode}  |  Task: {args.task}  |  Load: {args.load_mode}")
    print(f"  Query: {args.query}")
    print("="*64)

    worker=LocateAnythingWorker(model_path=args.model, load_mode=args.load_mode)

    if args.mode=="webcam":
        webcam_loop(worker,args)
    else:
        if not args.image: print("[ERROR] --image required"); sys.exit(1)
        image_mode(worker,args)


if __name__=="__main__":
    main()
