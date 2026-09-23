# =============================================================================
# F5-TTS Algerian Darja — Streamlit Web Application
# Model: touati-kamel/f5tts-algerian-darja (Step 48,574)
# Deployable to Streamlit Community Cloud, Hugging Face Spaces, or Local
# =============================================================================

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import streamlit as st

# Optional Hugging Face Spaces ZeroGPU integration
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Clean, Modern, RTL Support)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="F5-TTS — Algerian Darja Speech Synthesis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    
    .darja-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
        border: 1px solid #bae6fd;
    }
    
    /* Arabic RTL Input Area */
    .stTextArea textarea {
        font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
        font-size: 1.25rem !important;
        line-height: 1.8 !important;
        direction: rtl !important;
        text-align: right !important;
        border-radius: 10px !important;
    }
    
    /* Metric Cards */
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Example button styling */
    .sample-pill {
        display: inline-block;
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        font-size: 0.95rem;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem;
        background: #f1f5f9;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model Constants & Paths
# -----------------------------------------------------------------------------
DARJA_REPO  = "touati-kamel/f5tts-algerian-darja"
BASE_REPO   = "IbrahimSalah/Arabic-F5-TTS-v2"
SAMPLE_RATE = 24000

CACHE_DIR   = Path(os.environ.get("CACHE_DIR", tempfile.gettempdir())) / "f5tts_darja_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REFS_DIR    = CACHE_DIR / "refs"
REFS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 3. Model & Assets Download Function (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_model_assets():
    """Download checkpoint, vocab, and config from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id=DARJA_REPO,
        filename="model_last.pt",
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    vocab_path = hf_hub_download(
        repo_id=BASE_REPO,
        filename="vocab.txt",
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    cfg_path = hf_hub_download(
        repo_id=BASE_REPO,
        filename="F5TTS_Base_8_18.yaml",
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    
    # Download clean base Arabic reference as fallback
    base_ref_path = hf_hub_download(
        repo_id=BASE_REPO,
        filename="reference.wav",
        local_dir=str(REFS_DIR),
        local_dir_use_symlinks=False,
    )

    return {
        "ckpt": ckpt_path,
        "vocab": vocab_path,
        "config": cfg_path,
        "base_ref": base_ref_path,
    }


# -----------------------------------------------------------------------------
# 4. Streamlit Reference Voice Fetcher
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_preset_references():
    """
    Ensure reference audios are available for winning profiles:
    - Kahwa Podcast (Conversational)
    - Rawi Folklore (Male Storyteller)
    - Loubna Stories (Studio Female)
    """
    from datasets import load_dataset, Audio as AudioFeature

    presets = {
        "kahwa": {
            "name": "Kahwa Podcast (Conversational)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
            "file": REFS_DIR / "ref_kahwa.wav",
            "text": "واش راك خويا، لاباس عليك، كلشي مليح الحمد لله.",
            "desc": "Conversational Algerian tone with spontaneous cadence (Winning Try 5).",
        },
        "rawi": {
            "name": "Rawi Folklore (Male Storyteller)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-rawi",
            "file": REFS_DIR / "ref_rawi.wav",
            "text": "كان يا ما كان في قديم الزمان، كان كاين راجل عاقل.",
            "desc": "Deep resonant oral storytelling male voice (Winning Try 6).",
        },
        "loubna": {
            "name": "Loubna Stories (Studio Female)",
            "ds_id": "oddadmix/arabic-audio-collection-algerian-loubna-stories",
            "file": REFS_DIR / "ref_loubna.wav",
            "text": "في قديم الزمان كان كاين حكايات بزاف ملاح.",
            "desc": "Clean studio narration with high signal-to-noise ratio.",
        },
    }

    for key, info in presets.items():
        if info["file"].exists() and info["file"].stat().st_size > 5000:
            continue
        try:
            ds = load_dataset(info["ds_id"], split="train", streaming=True)
            ds = ds.cast_column("audio", AudioFeature(sampling_rate=SAMPLE_RATE))
            for item in ds:
                audio_obj = item.get("audio", {})
                if not audio_obj:
                    continue
                waveform = audio_obj["array"]
                dur = len(waveform) / audio_obj["sampling_rate"]
                if 4.0 <= dur <= 8.0:
                    sf.write(str(info["file"]), waveform.astype(np.float32), SAMPLE_RATE)
                    raw_txt = item.get("transcript_text") or item.get("text")
                    if raw_txt and len(raw_txt.split()) >= 4:
                        info["text"] = raw_txt.strip()
                    break
        except Exception:
            # If dataset streaming is unavailable, use base reference as fallback
            base_ref = REFS_DIR / "reference.wav"
            if base_ref.exists():
                shutil.copy(str(base_ref), str(info["file"]))

    return presets


# -----------------------------------------------------------------------------
# 5. Core Synthesis Routine
# -----------------------------------------------------------------------------
def run_f5_inference(
    gen_text: str,
    ref_audio: str,
    ref_text: str,
    ckpt_file: str,
    vocab_file: str,
    model_cfg: str,
    nfe_steps: int = 48,
    cfg_strength: float = 1.8,
    speed: float = 0.95,
    remove_silence: bool = True,
) -> str:
    """Execute F5-TTS inference and return output WAV path."""
    out_file = str(CACHE_DIR / f"gen_{int(time.time()*1000)}.wav")

    cmd = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "--model",        "F5TTS_Base",
        "--model_cfg",    model_cfg,
        "--ckpt_file",    ckpt_file,
        "--vocab_file",   vocab_file,
        "--ref_audio",    ref_audio,
        "--gen_text",     gen_text,
        "--output_file",  out_file,
        "--nfe_step",     str(nfe_steps),
        "--cfg_strength", str(cfg_strength),
        "--speed",        str(speed),
    ]
    if remove_silence:
        cmd.append("--remove_silence")
    if ref_text and ref_text.strip():
        cmd.extend(["--ref_text", ref_text.strip()])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"F5-TTS inference failed:\n{res.stderr[-500:]}")
    if not Path(out_file).exists():
        raise FileNotFoundError(f"Generated output not found at: {out_file}")

    return out_file


# -----------------------------------------------------------------------------
# 6. Sidebar: Voice Selection & Generation Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Voice Profile Selection")
    
    # Device indicator
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f"<span class='darja-badge'>Hardware: {device_name}</span>", unsafe_allow_html=True)

    voice_mode = st.radio(
        "Choose Algerian Voice Profile:",
        options=[
            "Kahwa Podcast (Conversational) — Try 5 Winner",
            "Rawi Folklore (Male Storyteller) — Try 6 Winner",
            "Loubna Stories (Studio Female)",
            "Custom Voice Cloning (Upload Audio)",
        ],
        index=0,
    )

    custom_audio_file = None
    custom_ref_text   = ""

    if "Custom" in voice_mode:
        st.markdown("---")
        st.markdown("#### Upload Voice Reference")
        custom_audio_file = st.file_uploader(
            "Upload reference audio (3–10 seconds WAV or MP3):",
            type=["wav", "mp3", "ogg", "flac"]
        )
        custom_ref_text = st.text_input(
            "Reference audio transcript (optional):",
            value="",
            help="If provided, improves in-context voice alignment."
        )
    else:
        st.markdown("---")
        st.caption("Active voice profile presets have been pre-tuned to the winning conditions discovered during evaluation.")

    # Advanced Generation Parameters
    st.markdown("---")
    with st.expander("Advanced Inference Parameters", expanded=False):
        # Default presets matching winning conditions
        if "Kahwa" in voice_mode:
            default_nfe, default_cfg, default_spd = 48, 1.8, 0.95
        elif "Rawi" in voice_mode:
            default_nfe, default_cfg, default_spd = 48, 2.0, 0.92
        else:
            default_nfe, default_cfg, default_spd = 48, 1.8, 0.95

        nfe_steps = st.slider("ODE Euler Steps (NFE)", min_value=16, max_value=64, value=default_nfe, step=4,
                              help="Higher steps (48-64) yield sharper consonant transitions.")
        cfg_strength = st.slider("Classifier-Free Guidance (CFG)", min_value=1.0, max_value=3.0, value=default_cfg, step=0.1,
                                 help="1.8-2.0 provides optimal balance between clarity and natural intonation.")
        speed = st.slider("Speech Rate (Speed)", min_value=0.75, max_value=1.25, value=default_spd, step=0.05,
                          help="0.92-0.95 gives natural spacing for rapid Darja contractions.")
        remove_silence = st.checkbox("Trim Silence Padding", value=True)

    st.markdown("---")
    st.markdown("""
    **Model Architecture:**
    - DiT Backbone (dim=1024, depth=22, heads=18)
    - 24,000 Hz Vocos Vocoder
    - Trained on ~399h OddAdmix Darja Speech
    """)


# -----------------------------------------------------------------------------
# 7. Main UI: Title, Benchmark Phrases & Generation Box
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>F5-TTS: Algerian Arabic (Darja) Speech Synthesis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>الدارجة الجزائرية — توليد الصوت بالذكاء الاصطناعي بنموذج F5-TTS المحسن</div>", unsafe_allow_html=True)

# 20 Benchmark Phrases Matrix
BENCHMARK_PRESETS = [
    ("Greeting / Everyday", "واش راك خويا، واش راهم العائلة والدراري، إن شاء الله كامل بخير."),
    ("Morning Motivation", "صباح الخير عليكم، اليوم كاين خدمة بزاف، لازم نزربو شوية باش نلحقو."),
    ("Casual Leisure", "صحا قهوتك، واش رايك نروحو نتمشاو شوية فالعشية كي تبرد الحالة؟"),
    ("Warm Reunion", "والله غير توحشناك يا صاحبي، شحال هادي ما تلاقينا وما قصرنا كيف كيف."),
    ("Tech / WhatsApp", "بعثتلك ميساج فالواتساب، شوفو و ريبونديلي كي تكون ديسبونيبل يرحم والديك."),
    ("Tech / Connection", "الكونكسيون اليوم راهي ثقيلة بزاف، التيليشارجومون حابس قاع ومقدرتش نخدم."),
    ("Tech / Application", "الأبليكاسيون هادي جديدة و براتيك، تعاونك تنظم الوقت تاعك كل يوم بلا تكسار راس."),
    ("Workplace / Project", "غدوة إن شاء الله عندنا ريونيون مع ليكيب، لازم نوجدو البروجي قبل الموعد."),
    ("Market / Price", "شحال يدير هاد الكيلو تاع الطماطيش، و عندك صرف تاع ألفين دينار؟"),
    ("Navigation / Post", "وين راهي البوسطة القريبة منا، نقدر نروح ليها على رجلية ولا بعيدة ولازم طاكسي؟"),
    ("Transport / Bus", "وقتاش يقلع الكار تاع وهران، مازال كاين بلايص ولا خلاصو كامل التواكر؟"),
    ("Polite Inquiry", "سمحلي خويا ما فهمتش واش قصدك، عاود فهمني بالعقل يرحم والديك."),
    ("Proverb / Measure", "الحديث قياس، والفاهم يفهم بالغمزة، والغافل حتى تدق فودنو."),
    ("Proverb / Present", "اللي فات مات، واللي راح ما يولي، تهلى فاليوم وخدم للغدوى باش تنجح."),
    ("Proverb / Patience", "الصبر مفتاح الفرج، كل عطلة فيها خير، والشدة ما تدوم لحتى واحد فهاد الدنيا."),
    ("Proverb / Wisdom", "خالط العاقل تكسب عقلو، وما تمشيش مع الجاهل اللي يضيعك فالطريق."),
    ("Folklore Opener", "كان يا ما كان في قديم الزمان، كان كاين سلطان عادل يحب الخير و يعاون قاع ناسو."),
    ("Folklore / Goha", "خرج جحا للغابة فالصباح الباكر، وفي نص الطريق سمع صوت غريب ورا الشجرة الكبيرة."),
    ("Heritage / Gathering", "في بلادنا كاين تقاليد عريقة، القعدة الزينة تاع زمان والقصايد اللي تحكي تاريخ الرجال الأحرار."),
    ("Atmospheric / Sunset", "الشمس راهي تغرب ورا الجبال العالية، والهدوء سكن القرية مع وقت صوت أذان المغرب."),
]

# Quick Phrase Picker
with st.expander("Explore 20 Algerian Benchmark Test Phrases (Click to load)", expanded=False):
    st.caption("Click any preset sentence to populate the generation input below:")
    cols = st.columns(2)
    for idx, (label, phrase) in enumerate(BENCHMARK_PRESETS):
        col = cols[idx % 2]
        if col.button(f"{idx+1}. {label}: {phrase[:45]}...", key=f"btn_{idx}"):
            st.session_state["input_text"] = phrase

# Default winning sentence
default_text = "السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو، هادا تيست للمودال الجديد، أدخلو جربوه و قولولي."
if "input_text" not in st.session_state:
    st.session_state["input_text"] = default_text

# Text Input Area
user_text = st.text_area(
    label="Text to Synthesize (نص بالدارجة الجزائرية):",
    value=st.session_state["input_text"],
    height=120,
    help="Tip: Include natural commas (،) and full stops (.) to provide natural breathing pauses for flow matching."
)

col_gen, col_clear = st.columns([4, 1])
generate_btn = col_gen.button("Synthesize Speech | توليد الصوت", type="primary", use_container_width=True)
if col_clear.button("Reset Text", use_container_width=True):
    st.session_state["input_text"] = default_text
    st.rerun()


# -----------------------------------------------------------------------------
# 8. Generation Handler & Output Presentation
# -----------------------------------------------------------------------------
if generate_btn:
    if not user_text.strip():
        st.warning("Please enter a sentence to synthesize.")
    else:
        with st.spinner("Downloading assets and synthesizing speech..."):
            try:
                # 1. Download/verify model assets
                assets = download_model_assets()
                presets = get_preset_references()

                # 2. Determine reference audio & text
                if "Custom" in voice_mode:
                    if custom_audio_file is None:
                        st.error("Please upload a custom audio file, or select a preset voice profile.")
                        st.stop()
                    ref_audio_path = str(CACHE_DIR / f"upload_{custom_audio_file.name}")
                    with open(ref_audio_path, "wb") as f:
                        f.write(custom_audio_file.getbuffer())
                    ref_text_val = custom_ref_text
                    voice_display_label = "Custom User Voice"
                elif "Kahwa" in voice_mode:
                    ref_info = presets["kahwa"]
                    ref_audio_path = str(ref_info["file"])
                    ref_text_val = ref_info["text"]
                    voice_display_label = ref_info["name"]
                elif "Rawi" in voice_mode:
                    ref_info = presets["rawi"]
                    ref_audio_path = str(ref_info["file"])
                    ref_text_val = ref_info["text"]
                    voice_display_label = ref_info["name"]
                else:
                    ref_info = presets["loubna"]
                    ref_audio_path = str(ref_info["file"])
                    ref_text_val = ref_info["text"]
                    voice_display_label = ref_info["name"]

                # 3. Execute inference
                t_start = time.time()
                
                # Apply spaces.GPU if available on Hugging Face Spaces
                if has_spaces:
                    gpu_infer = spaces.GPU(run_f5_inference)
                    output_wav = gpu_infer(
                        gen_text=user_text,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text_val,
                        ckpt_file=assets["ckpt"],
                        vocab_file=assets["vocab"],
                        model_cfg=assets["config"],
                        nfe_steps=nfe_steps,
                        cfg_strength=cfg_strength,
                        speed=speed,
                        remove_silence=remove_silence,
                    )
                else:
                    output_wav = run_f5_inference(
                        gen_text=user_text,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text_val,
                        ckpt_file=assets["ckpt"],
                        vocab_file=assets["vocab"],
                        model_cfg=assets["config"],
                        nfe_steps=nfe_steps,
                        cfg_strength=cfg_strength,
                        speed=speed,
                        remove_silence=remove_silence,
                    )
                
                t_elapsed = time.time() - t_start

                # Read output audio
                audio_bytes = open(output_wav, "rb").read()
                data, sr = sf.read(output_wav)
                audio_dur = len(data) / sr

                st.success("Synthesis complete!")

                # Audio Player Section
                st.markdown("### Synthesized Audio Result")
                st.audio(audio_bytes, format="audio/wav")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='stat-card'><div class='stat-label'>Voice Profile</div><div class='stat-val'>{voice_display_label.split('(')[0]}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='stat-card'><div class='stat-label'>Latency</div><div class='stat-val'>{t_elapsed:.2f}s</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='stat-card'><div class='stat-label'>Audio Duration</div><div class='stat-val'>{audio_dur:.2f}s</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='stat-card'><div class='stat-label'>Real-Time Factor</div><div class='stat-val'>{t_elapsed/max(audio_dur, 0.01):.2f}x</div></div>", unsafe_allow_html=True)

                # Download Button
                st.download_button(
                    label="Download Synthesized WAV File",
                    data=audio_bytes,
                    file_name=f"darja_f5tts_{int(time.time())}.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Error during synthesis: {str(e)}")
