# =============================================================================
# F5-TTS Algerian Darja — Streamlit Web Application
# Model: algerian-nlp/Hadra-TTS-f5 (Step 48,574)
# Generates 10 audio variations per sentence so the user can pick the best.
# Deployable to Streamlit Community Cloud, Hugging Face Spaces, or Local
# =============================================================================

import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import streamlit as st


def ensure_package(module_name: str, pypi_name: str = None, extra_args: list = None):
    """Ensure a Python module can be imported. If missing, install it dynamically via uv or pip."""
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    if pypi_name is None:
        pypi_name = module_name.replace("_", "-")

    cmds = []
    # 1. uv pip install (Streamlit Cloud's default fast installer)
    uv_bin = shutil.which("uv")
    if not uv_bin:
        for candidate in ["/home/adminuser/.cargo/bin/uv", "/usr/local/bin/uv", "/root/.cargo/bin/uv"]:
            if Path(candidate).exists():
                uv_bin = candidate
                break
    if uv_bin:
        uv_cmd = [str(uv_bin), "pip", "install", "--python", sys.executable, pypi_name]
        if extra_args:
            uv_cmd.extend(extra_args)
        cmds.append(uv_cmd)

    # 2. python -m pip
    py_cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q"]
    if extra_args:
        py_cmd.extend(extra_args)
    py_cmd.append(pypi_name)
    cmds.append(py_cmd)

    # 3. standard pip
    pip_bin = shutil.which("pip")
    if pip_bin:
        pip_cmd = [str(pip_bin), "install", "--no-cache-dir", "-q"]
        if extra_args:
            pip_cmd.extend(extra_args)
        pip_cmd.append(pypi_name)
        cmds.append(pip_cmd)

    for cmd in cmds:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                importlib.invalidate_caches()
                try:
                    return importlib.import_module(module_name)
                except ImportError:
                    pass
        except Exception:
            continue

    return None



try:
    import torch
except ImportError:
    torch = ensure_package(
        "torch",
        "torch>=2.2.0",
        extra_args=["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
    )

try:
    import soundfile as sf
except ImportError:
    sf = ensure_package("soundfile", "soundfile>=0.12.1")


# Optional Hugging Face Spaces ZeroGPU integration
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="F5-TTS — Algerian Darja Speech Synthesis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .darja-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
        border: 1px solid #bae6fd;
    }
    .stTextArea textarea {
        font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
        font-size: 1.25rem !important;
        line-height: 1.8 !important;
        direction: rtl !important;
        text-align: right !important;
        border-radius: 10px !important;
    }
    /* Try card */
    .try-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .try-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .try-meta {
        font-size: 0.78rem;
        color: #64748b;
    }
    /* Stat row */
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }
    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Constants & Paths
# -----------------------------------------------------------------------------
DARJA_REPO  = "algerian-nlp/Hadra-TTS-f5"
BASE_REPO   = "IbrahimSalah/Arabic-F5-TTS-v2"
SAMPLE_RATE = 24000

CACHE_DIR = Path(os.environ.get("CACHE_DIR", tempfile.gettempdir())) / "f5tts_darja_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REFS_DIR  = CACHE_DIR / "refs"
REFS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. The 10 Generation Tries — diverse voice, NFE, CFG, and speed combos
# -----------------------------------------------------------------------------
GEN_TRIES = [
    {
        "id": "t01", "label": "Try 1",
        "profile": "Kahwa (Conversational)",
        "ref_domain": "kahwa",
        "nfe": 48, "cfg": 1.8, "speed": 0.95,
        "tag": "Winning Try 5 — NFE=48, CFG=1.8, Speed=0.95",
    },
    {
        "id": "t02", "label": "Try 2",
        "profile": "Rawi (Male Narrator)",
        "ref_domain": "rawi",
        "nfe": 48, "cfg": 2.0, "speed": 0.92,
        "tag": "Winning Try 6 — NFE=48, CFG=2.0, Speed=0.92",
    },
    {
        "id": "t03", "label": "Try 3",
        "profile": "Loubna (Studio Female)",
        "ref_domain": "loubna",
        "nfe": 48, "cfg": 1.8, "speed": 0.95,
        "tag": "Studio Clean — NFE=48, CFG=1.8, Speed=0.95",
    },
    {
        "id": "t04", "label": "Try 4",
        "profile": "Kahwa (Conversational)",
        "ref_domain": "kahwa",
        "nfe": 64, "cfg": 1.8, "speed": 0.95,
        "tag": "High Resolution — NFE=64, CFG=1.8, Speed=0.95",
    },
    {
        "id": "t05", "label": "Try 5",
        "profile": "Rawi (Male Narrator)",
        "ref_domain": "rawi",
        "nfe": 64, "cfg": 2.0, "speed": 0.92,
        "tag": "HD Narrator — NFE=64, CFG=2.0, Speed=0.92",
    },
    {
        "id": "t06", "label": "Try 6",
        "profile": "Loubna (Studio Female)",
        "ref_domain": "loubna",
        "nfe": 64, "cfg": 2.0, "speed": 0.95,
        "tag": "HD Studio — NFE=64, CFG=2.0, Speed=0.95",
    },
    {
        "id": "t07", "label": "Try 7",
        "profile": "Kahwa (Conversational)",
        "ref_domain": "kahwa",
        "nfe": 48, "cfg": 1.5, "speed": 0.90,
        "tag": "Relaxed Pacing — NFE=48, CFG=1.5, Speed=0.90",
    },
    {
        "id": "t08", "label": "Try 8",
        "profile": "Kahwa (Conversational)",
        "ref_domain": "kahwa",
        "nfe": 48, "cfg": 2.5, "speed": 0.95,
        "tag": "Strong Guidance — NFE=48, CFG=2.5, Speed=0.95",
    },
    {
        "id": "t09", "label": "Try 9",
        "profile": "Rawi (Male Narrator)",
        "ref_domain": "rawi",
        "nfe": 48, "cfg": 1.8, "speed": 0.88,
        "tag": "Slow Deliberate — NFE=48, CFG=1.8, Speed=0.88",
    },
    {
        "id": "t10", "label": "Try 10",
        "profile": "Loubna (Studio Female)",
        "ref_domain": "loubna",
        "nfe": 32, "cfg": 1.8, "speed": 1.0,
        "tag": "Fast Baseline — NFE=32, CFG=1.8, Speed=1.0",
    },
]


# -----------------------------------------------------------------------------
# 4. Asset Download (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_model_assets():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        ensure_package("huggingface_hub", "huggingface-hub>=0.25.0")
        from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(DARJA_REPO, "model_last.pt", local_dir=str(CACHE_DIR))
    try:
        vocab = hf_hub_download(DARJA_REPO, "vocab.txt", local_dir=str(CACHE_DIR))
        cfg = hf_hub_download(DARJA_REPO, "F5TTS_Base_8_18.yaml", local_dir=str(CACHE_DIR))
    except Exception:
        vocab = hf_hub_download(BASE_REPO, "vocab.txt", local_dir=str(CACHE_DIR))
        cfg = hf_hub_download(BASE_REPO, "F5TTS_Base_8_18.yaml", local_dir=str(CACHE_DIR))
    base_ref = hf_hub_download(BASE_REPO, "reference.wav", local_dir=str(REFS_DIR))
    return {"ckpt": ckpt, "vocab": vocab, "config": cfg, "base_ref": base_ref}


# -----------------------------------------------------------------------------
# 5. Reference Voice Fetcher (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_preset_references():
    try:
        from datasets import load_dataset, Audio as AudioFeature
    except ImportError:
        ensure_package("datasets", "datasets>=2.14.0")
        try:
            from datasets import load_dataset, Audio as AudioFeature
        except ImportError:
            load_dataset = None
            AudioFeature = None

    presets = {
        "kahwa": {
            "name": "Kahwa Podcast (Conversational)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
            "file": REFS_DIR / "ref_kahwa.wav",
            "text": "واش راك خويا، لاباس عليك، كلشي مليح الحمد لله.",
        },
        "rawi": {
            "name": "Rawi Folklore (Male Narrator)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-rawi",
            "file": REFS_DIR / "ref_rawi.wav",
            "text": "كان يا ما كان في قديم الزمان، كان كاين راجل عاقل.",
        },
        "loubna": {
            "name": "Loubna Stories (Studio Female)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-loubna-stories",
            "file": REFS_DIR / "ref_loubna.wav",
            "text": "في قديم الزمان كان كاين حكايات بزاف ملاح.",
        },
    }

    for key, info in presets.items():
        if info["file"].exists() and info["file"].stat().st_size > 5000:
            continue
        try:
            if load_dataset is None:
                raise RuntimeError("datasets library unavailable")
            ds = load_dataset(info["ds_id"], split="train", streaming=True)
            ds = ds.cast_column("audio", AudioFeature(sampling_rate=SAMPLE_RATE))
            for item in ds:
                audio_obj = item.get("audio", {})
                if not audio_obj:
                    continue
                waveform = audio_obj["array"]
                dur = len(waveform) / audio_obj["sampling_rate"]
                if 4.0 <= dur <= 8.0:
                    if sf is not None:
                        sf.write(str(info["file"]), waveform.astype(np.float32), SAMPLE_RATE)
                    else:
                        import wave
                        with wave.open(str(info["file"]), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            int16_data = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                            wf.writeframes(int16_data.tobytes())
                    raw_txt = item.get("transcript_text") or item.get("text")
                    if raw_txt and len(raw_txt.split()) >= 4:
                        info["text"] = raw_txt.strip()
                    break
        except Exception:
            base_ref = REFS_DIR / "reference.wav"
            if base_ref.exists():
                shutil.copy(str(base_ref), str(info["file"]))

    return presets


# -----------------------------------------------------------------------------
# 6. Core Inference Routine
# -----------------------------------------------------------------------------
def check_f5tts_installed():
    try:
        import f5_tts
        return True
    except ImportError:
        pkg = ensure_package(
            "f5_tts",
            "f5-tts>=0.1.0",
            extra_args=["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        )
        return pkg is not None


def run_synthesis(
    gen_text: str,
    ref_audio: str,
    ref_text: str,
    ckpt_file: str,
    vocab_file: str,
    model_cfg: str,
    nfe: int,
    cfg_strength: float,
    speed: float,
    output_name: str,
) -> str:
    check_f5tts_installed()
    out_file = str(CACHE_DIR / f"{output_name}_{int(time.time()*1000)}.wav")
    cmd = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "--model",        "F5TTS_Base",
        "--model_cfg",    model_cfg,
        "--ckpt_file",    ckpt_file,
        "--vocab_file",   vocab_file,
        "--ref_audio",    ref_audio,
        "--gen_text",     gen_text,
        "--output_file",  out_file,
        "--nfe_step",     str(nfe),
        "--cfg_strength", str(cfg_strength),
        "--speed",        str(speed),
        "--remove_silence",
    ]
    if ref_text and ref_text.strip():
        cmd.extend(["--ref_text", ref_text.strip()])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err_msg = res.stderr[-500:] if res.stderr else res.stdout[-500:]
        raise RuntimeError(err_msg or "Inference CLI exited with an error code")
    if not Path(out_file).exists():
        raise FileNotFoundError(f"Generated file not found: {out_file}")
    return out_file


# -----------------------------------------------------------------------------
# 7. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Custom Voice Reference")
    device_name = "CUDA GPU" if (torch is not None and torch.cuda.is_available()) else "CPU"
    st.markdown(f"<span class='darja-badge'>Hardware: {device_name}</span>", unsafe_allow_html=True)
    if torch is None:
        st.info("Dependencies are loading. If this persists, click 'Manage app' (bottom-right) > '...' > 'Reboot app'.")
    st.caption("All 10 tries always run across Kahwa, Rawi, and Loubna voices.")

    st.markdown("---")
    custom_audio_file = st.file_uploader(
        "Upload your own voice reference (3–10 sec WAV/MP3):",
        type=["wav", "mp3", "ogg", "flac"],
        help="If uploaded, 3 extra tries using your voice clone will be appended."
    )
    custom_ref_text = st.text_input(
        "Reference audio transcript (optional):", value=""
    )

    st.markdown("---")
    st.markdown("""
**Model Summary:**
- DiT (dim=1024, depth=22, heads=18)
- 24,000 Hz Vocos Vocoder
- Trained on ~399h OddAdmix Darja
- 48,574 training steps
    """)


# -----------------------------------------------------------------------------
# 8. Main UI
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>F5-TTS: Algerian Arabic (Darja) Speech Synthesis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>الدارجة الجزائرية — 10 توليدات مختلفة لكل جملة، اختار اللي عجبك</div>", unsafe_allow_html=True)

# 20 Benchmark Phrases
BENCHMARK_PRESETS = [
    ("Greeting",       "واش راك خويا، واش راهم العائلة والدراري، إن شاء الله كامل بخير."),
    ("Morning",        "صباح الخير عليكم، اليوم كاين خدمة بزاف، لازم نزربو شوية باش نلحقو."),
    ("Leisure",        "صحا قهوتك، واش رايك نروحو نتمشاو شوية فالعشية كي تبرد الحالة؟"),
    ("Warmth",         "والله غير توحشناك يا صاحبي، شحال هادي ما تلاقينا وما قصرنا كيف كيف."),
    ("WhatsApp",       "بعثتلك ميساج فالواتساب، شوفو و ريبونديلي كي تكون ديسبونيبل يرحم والديك."),
    ("Connection",     "الكونكسيون اليوم راهي ثقيلة بزاف، التيليشارجومون حابس قاع ومقدرتش نخدم."),
    ("App",            "الأبليكاسيون هادي جديدة و براتيك، تعاونك تنظم الوقت تاعك كل يوم بلا تكسار راس."),
    ("Meeting",        "غدوة إن شاء الله عندنا ريونيون مع ليكيب، لازم نوجدو البروجي قبل الموعد."),
    ("Market",         "شحال يدير هاد الكيلو تاع الطماطيش، و عندك صرف تاع ألفين دينار؟"),
    ("Navigation",     "وين راهي البوسطة القريبة منا، نقدر نروح ليها على رجلية ولا بعيدة ولازم طاكسي؟"),
    ("Transport",      "وقتاش يقلع الكار تاع وهران، مازال كاين بلايص ولا خلاصو كامل التواكر؟"),
    ("Clarification",  "سمحلي خويا ما فهمتش واش قصدك، عاود فهمني بالعقل يرحم والديك."),
    ("Proverb 1",      "الحديث قياس، والفاهم يفهم بالغمزة، والغافل حتى تدق فودنو."),
    ("Proverb 2",      "اللي فات مات، واللي راح ما يولي، تهلى فاليوم وخدم للغدوى باش تنجح."),
    ("Patience",       "الصبر مفتاح الفرج، كل عطلة فيها خير، والشدة ما تدوم لحتى واحد فهاد الدنيا."),
    ("Wisdom",         "خالط العاقل تكسب عقلو، وما تمشيش مع الجاهل اللي يضيعك فالطريق."),
    ("Folklore",       "كان يا ما كان في قديم الزمان، كان كاين سلطان عادل يحب الخير و يعاون قاع ناسو."),
    ("Goha Story",     "خرج جحا للغابة فالصباح الباكر، وفي نص الطريق سمع صوت غريب ورا الشجرة الكبيرة."),
    ("Heritage",       "في بلادنا كاين تقاليد عريقة، القعدة الزينة تاع زمان والقصايد اللي تحكي تاريخ الرجال الأحرار."),
    ("Sunset",         "الشمس راهي تغرب ورا الجبال العالية، والهدوء سكن القرية مع وقت صوت أذان المغرب."),
]

with st.expander("20 Algerian Benchmark Phrases — Click to Load", expanded=False):
    cols = st.columns(2)
    for idx, (label, phrase) in enumerate(BENCHMARK_PRESETS):
        col = cols[idx % 2]
        if col.button(f"{idx+1}. {label}: {phrase[:42]}...", key=f"btn_{idx}"):
            st.session_state["input_text"] = phrase

default_text = "السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو، هادا تيست للمودال الجديد، أدخلو جربوه و قولولي."
if "input_text" not in st.session_state:
    st.session_state["input_text"] = default_text

user_text = st.text_area(
    label="Text to Synthesize (الدارجة الجزائرية):",
    value=st.session_state["input_text"],
    height=110,
    help="Use natural commas (،) and full stops (.) for better pacing and breath control.",
)

col_btn, col_clear = st.columns([5, 1])
generate_btn = col_btn.button(
    "Generate 10 Audio Versions | توليد 10 نسخ صوتية",
    type="primary",
    use_container_width=True,
)
if col_clear.button("Reset", use_container_width=True):
    st.session_state["input_text"] = default_text
    st.rerun()


# -----------------------------------------------------------------------------
# 9. Generation: 10 tries rendered live as each completes
# -----------------------------------------------------------------------------
if generate_btn:
    if not user_text.strip():
        st.warning("Please enter a sentence to synthesize.")
        st.stop()

    # Determine effective tries: 10 standard + up to 3 custom voice
    tries_to_run = list(GEN_TRIES)
    if custom_audio_file is not None:
        custom_path = str(CACHE_DIR / f"upload_{custom_audio_file.name}")
        with open(custom_path, "wb") as f:
            f.write(custom_audio_file.getbuffer())
        for extra_idx, (nfe, cfg, spd) in enumerate([(48, 1.8, 0.95), (64, 2.0, 0.92), (48, 1.5, 0.90)], 11):
            tries_to_run.append({
                "id": f"custom_{extra_idx}",
                "label": f"Try {extra_idx} (Your Voice)",
                "profile": "Custom Cloned Voice",
                "ref_domain": "_custom_",
                "nfe": nfe, "cfg": cfg, "speed": spd,
                "tag": f"Your Voice Clone — NFE={nfe}, CFG={cfg}, Speed={spd}",
                "_custom_path": custom_path,
                "_custom_text": custom_ref_text,
            })

    # Load assets
    with st.spinner("Downloading and caching Hadra-TTS-f5 model checkpoint (~3.5 GB)... This only happens once."):
        try:
            assets  = download_model_assets()
            presets = get_preset_references()
        except Exception as e:
            st.error(
                f"Failed to load model assets: {e}\n\n"
                "Tip: Click the bottom-right '...' menu and choose 'Clear cache and reboot' to ensure clean container dependencies."
            )
            st.stop()

    total = len(tries_to_run)
    st.markdown(f"### Generating {total} Audio Versions")
    st.caption("Each version is rendered live below as it completes. Listen and pick the best one.")

    # Progress
    progress_bar = st.progress(0)
    status_text  = st.empty()

    results = []

    # Run all tries sequentially and stream results into page
    for i, t in enumerate(tries_to_run):
        status_text.markdown(f"Synthesizing **{t['label']}** — {t['profile']} ({t['tag']})...")
        try:
            # Resolve reference audio & text
            if t["ref_domain"] == "_custom_":
                ref_audio = t["_custom_path"]
                ref_text  = t["_custom_text"]
            else:
                ref_info  = presets.get(t["ref_domain"]) or list(presets.values())[0]
                ref_audio = str(ref_info["file"])
                ref_text  = ref_info.get("text", "")

            t0 = time.time()
            out_path = run_synthesis(
                gen_text    = user_text,
                ref_audio   = ref_audio,
                ref_text    = ref_text,
                ckpt_file   = assets["ckpt"],
                vocab_file  = assets["vocab"],
                model_cfg   = assets["config"],
                nfe         = t["nfe"],
                cfg_strength= t["cfg"],
                speed       = t["speed"],
                output_name = t["id"],
            )
            elapsed = time.time() - t0
            if sf is not None:
                data, sr = sf.read(out_path)
                dur = len(data) / sr
            else:
                import wave
                with wave.open(out_path, "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())

            results.append({
                "try": t,
                "path": out_path,
                "elapsed": elapsed,
                "dur": dur,
                "ok": True,
            })

        except Exception as e:
            results.append({"try": t, "ok": False, "error": str(e)})

        progress_bar.progress((i + 1) / total)

    status_text.markdown("All versions generated. Listen below and pick your favourite.")

    # Separator
    st.markdown("---")
    st.markdown("## Results — Listen and Choose")

    # Display all results in a 2-column grid
    left_col, right_col = st.columns(2)
    col_map = {0: left_col, 1: right_col}

    ok_results = [r for r in results if r["ok"]]

    for idx, r in enumerate(results):
        t       = r["try"]
        col     = col_map[idx % 2]

        with col:
            if r["ok"]:
                audio_bytes = open(r["path"], "rb").read()

                col.markdown(f"""
<div class='try-card'>
    <div class='try-title'>{t['label']} — {t['profile']}</div>
    <div class='try-meta'>{t['tag']}</div>
    <div class='try-meta' style='margin-top:0.2rem;'>Duration: {r['dur']:.2f}s &nbsp;|&nbsp; Latency: {r['elapsed']:.1f}s &nbsp;|&nbsp; RTF: {r['elapsed']/max(r['dur'],0.01):.2f}x</div>
</div>
""", unsafe_allow_html=True)
                col.audio(audio_bytes, format="audio/wav")
                col.download_button(
                    label=f"Download {t['label']}",
                    data=audio_bytes,
                    file_name=f"darja_{t['id']}.wav",
                    mime="audio/wav",
                    key=f"dl_{t['id']}",
                    use_container_width=True,
                )
            else:
                col.markdown(f"""
<div class='try-card' style='border-color:#fca5a5;background:#fff5f5;'>
    <div class='try-title' style='color:#dc2626;'>{t['label']} — Failed</div>
    <div class='try-meta'>{r.get('error','Unknown error')[:120]}</div>
</div>
""", unsafe_allow_html=True)

    # Summary stats
    if ok_results:
        st.markdown("---")
        st.markdown("### Summary Statistics")
        m1, m2, m3, m4 = st.columns(4)
        avg_latency = sum(r["elapsed"] for r in ok_results) / len(ok_results)
        avg_dur     = sum(r["dur"]     for r in ok_results) / len(ok_results)
        fastest     = min(ok_results, key=lambda x: x["elapsed"])
        m1.markdown(f"<div class='stat-card'><div class='stat-label'>Versions Generated</div><div class='stat-val'>{len(ok_results)} / {total}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='stat-card'><div class='stat-label'>Avg Latency</div><div class='stat-val'>{avg_latency:.1f}s</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='stat-card'><div class='stat-label'>Avg Audio Duration</div><div class='stat-val'>{avg_dur:.2f}s</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='stat-card'><div class='stat-label'>Fastest Try</div><div class='stat-val'>{fastest['try']['label']} ({fastest['elapsed']:.1f}s)</div></div>", unsafe_allow_html=True)
