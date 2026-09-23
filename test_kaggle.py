# =============================================================================
# F5-TTS Algerian Darja — Evaluation Notebook & 20-Phrase Benchmark (Kaggle)
# Model: touati-kamel/f5tts-algerian-darja (Step 48,574)
# Winning Profiles: Try 5 (Kahwa Conversational) & Try 6 (Rawi Storyteller)
# =============================================================================
# Cell 1 — Install dependencies
# =============================================================================

import subprocess, sys

print("Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "f5-tts", "huggingface_hub>=0.25.0", "soundfile", "datasets",
    "accelerate", "IPython"
], check=False)
print("Done.")


# =============================================================================
# Cell 2 — Imports & Config
# =============================================================================

import os, re, gc, json, shutil, time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from IPython.display import Audio, display, HTML
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio as AudioFeature

# Paths
WORK_DIR    = Path("/kaggle/working/f5tts_test")
REF_DIR     = WORK_DIR / "refs"
OUT_DIR     = WORK_DIR / "outputs"
MODEL_DIR   = WORK_DIR / "model"
for d in [REF_DIR, OUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Repositories & Audio Config
DARJA_REPO  = "touati-kamel/f5tts-algerian-darja"
BASE_REPO   = "IbrahimSalah/Arabic-F5-TTS-v2"
SAMPLE_RATE = 24000

print(f"Work directory : {WORK_DIR}")
print(f"Compute device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU Model      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# Cell 3 — Download model files & configs
# =============================================================================

print("Downloading model checkpoints and configs...")

# Fine-tuned Algerian Darja checkpoint
darja_ckpt = hf_hub_download(
    repo_id=DARJA_REPO,
    filename="model_last.pt",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)
print(f"  [OK] Darja model checkpoint : {darja_ckpt} ({Path(darja_ckpt).stat().st_size / 1e9:.2f} GB)")

# Shared Arabic vocabulary file
vocab_file = hf_hub_download(
    repo_id=BASE_REPO,
    filename="vocab.txt",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)
print(f"  [OK] Vocabulary file       : {vocab_file}")

# Architecture configuration YAML (DiT 8_18)
model_cfg = hf_hub_download(
    repo_id=BASE_REPO,
    filename="F5TTS_Base_8_18.yaml",
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)
print(f"  [OK] Model config YAML      : {model_cfg}")


# =============================================================================
# Cell 4 — Fetch reference voices for winning profiles (Kahwa & Rawi)
# =============================================================================

print("\nStreaming reference voices from OddAdmix datasets...")

# Darja text normalizer matching training pipeline
_FR_TAG_RE   = re.compile(r"\[\s*(?:French|FR)\s*:\s*(.*?)\]", re.IGNORECASE)
_BRACKET_RE  = re.compile(r"\[.*?\]|<.*?>|\(.*?\)")
_NOISE_CHARS = re.compile(r"[*#@~_\^&%$+=/\\|{}\[\]`\"«»]")

def normalize_darja(text: str) -> str:
    if not text:
        return ""
    text = _FR_TAG_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub("", text)
    text = _NOISE_CHARS.sub("", text)
    text = text.replace("\u0640", "")
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")
    return " ".join(text.split()).strip()

# Target reference domains: Kahwa (podcast conversational) & Rawi (folklore male)
DATASET_CONFIGS = [
    ("oddadmix/arabic-audio-collection-algerian-kahwa-postcast", "kahwa", "Kahwa Podcast (Conversational Voice)"),
    ("oddadmix/arabic-audio-collection-algerian-rawi",           "rawi",  "Rawi Folklore (Male Storyteller Voice)"),
    ("oddadmix/arabic-audio-collection-algerian-loubna-stories", "loubna","Loubna Stories (Studio Female Voice)"),
]

ref_samples = {}

for ds_id, domain, label in DATASET_CONFIGS:
    print(f"  Loading reference candidate for {domain}...")
    try:
        ds = load_dataset(ds_id, split="train", streaming=True)
        ds = ds.cast_column("audio", AudioFeature(sampling_rate=SAMPLE_RATE))

        for sample in ds:
            audio_obj = sample.get("audio", {})
            if not audio_obj:
                continue

            waveform = audio_obj["array"]
            sr = audio_obj["sampling_rate"]
            duration = len(waveform) / sr

            # Optimal reference length: 4.0 to 8.0 seconds with clean speech
            if not (4.0 <= duration <= 8.0):
                continue

            raw_text = sample.get("transcript_text") or sample.get("text") or ""
            clean_text = normalize_darja(raw_text)
            if len(clean_text.split()) < 5:
                continue

            out_path = str(REF_DIR / f"ref_{domain}.wav")
            sf.write(out_path, waveform.astype(np.float32), sr)

            ref_samples[domain] = {
                "audio_path": out_path,
                "text": clean_text,
                "duration": round(duration, 2),
                "label": label,
            }
            print(f"    [OK] {domain}: {duration:.1f}s | '{clean_text[:60]}...'")
            break
    except Exception as e:
        print(f"    [WARN] Failed loading {domain}: {e}")

# Optional: Check if user uploaded a custom reference to /kaggle/working/custom_ref.wav
custom_ref = Path("/kaggle/working/custom_ref.wav")
if custom_ref.exists():
    ref_samples["custom"] = {
        "audio_path": str(custom_ref),
        "text": "",
        "duration": round(len(sf.read(str(custom_ref))[0]) / SAMPLE_RATE, 2),
        "label": "User Custom Uploaded Reference",
    }
    print(f"  [OK] Detected custom user reference: {custom_ref}")

print("\n=== Loaded Reference Audio Clips ===")
for domain, info in ref_samples.items():
    print(f"\n[{domain.upper()}] ({info['duration']}s): {info['label']}")
    if info.get("text"):
        print(f"  Transcript: {info['text']}")
    display(Audio(info["audio_path"], rate=SAMPLE_RATE))


# =============================================================================
# Cell 5 — Inference function
# =============================================================================

def synthesize(
    gen_text: str,
    ref_audio: str,
    ref_text: str,
    ckpt_file: str,
    output_name: str,
    nfe_steps: int = 48,
    cfg_strength: float = 1.8,
    speed: float = 0.95,
) -> str:
    """Run F5-TTS inference and return path to generated WAV."""
    output_file = str(OUT_DIR / f"{output_name}.wav")

    cmd = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "--model",        "F5TTS_Base",
        "--model_cfg",    model_cfg,
        "--ckpt_file",    ckpt_file,
        "--vocab_file",   vocab_file,
        "--ref_audio",    ref_audio,
        "--gen_text",     gen_text,
        "--output_file",  output_file,
        "--nfe_step",     str(nfe_steps),
        "--cfg_strength", str(cfg_strength),
        "--speed",        str(speed),
        "--remove_silence",
    ]
    if ref_text and ref_text.strip():
        cmd.extend(["--ref_text", ref_text.strip()])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] CLI failed: {result.stderr[-400:]}")
        return None
    if not Path(output_file).exists():
        print(f"  [ERROR] Output file not found: {output_file}")
        return None

    return output_file


# =============================================================================
# Cell 6 — Winning Configurations Matrix (Try 5 & Try 6)
# =============================================================================

# The Winning Primary Test Sentence Variations
TEXT_ARABIC_TEST = "السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو، هادا تيست للمودال الجديد، أدخلو جربوه و قولولي."
TEXT_LATIN_TEST  = "السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو، هادا test للمودال الجديد، أدخلو جربوه و قولولي."

WINNING_CONFIGS = [
    # Winning Condition 1 (from Try 5): Kahwa Conversational Voice + Arabic Transliteration
    {
        "id": "primary_try_5_kahwa",
        "title": "Try 5: Kahwa Conversational Voice (Arabic 'تيست')",
        "profile": "Conversational / Modern Algerian Podcast",
        "description": "Natural Algerian conversational tone from Kahwa Podcast. Uses Arabic script 'تيست' for smooth vowel transitions, calibrated comma pauses, NFE=48, relaxed CFG=1.8 to prevent vocal strain, and speed=0.95 for realistic conversational pacing.",
        "text": TEXT_ARABIC_TEST,
        "ref_domain": "kahwa",
        "nfe_steps": 48,
        "cfg_strength": 1.8,
        "speed": 0.95,
    },
    # Winning Condition 2 (from Try 6): Rawi Storyteller Voice + Punctuated Deliberate Cadence
    {
        "id": "primary_try_6_rawi",
        "title": "Try 6: Rawi Storyteller Voice (Male Narrator | Deliberate Cadence)",
        "profile": "Traditional Oral Narrative / Announcement",
        "description": "Deep male narrative cadence from Rawi Folklore. Uses punctuated pauses for natural breath control, deliberate pacing (speed=0.92), solid CFG=2.0 for vocal presence, and NFE=48 for clean oral intonation.",
        "text": TEXT_LATIN_TEST,
        "ref_domain": "rawi",
        "nfe_steps": 48,
        "cfg_strength": 2.0,
        "speed": 0.92,
    },
]

# =============================================================================
# 20 Curated Benchmark Phrases (Algerian Darja Across 5 Dialectal Domains)
# =============================================================================

BENCHMARK_20_PHRASES = [
    # ── Category 1: Conversational & Daily Greetings (Kahwa Profile) ──────────
    {
        "id": "phrase_01_greeting",
        "category": "Conversational / Greeting",
        "text": "واش راك خويا، واش راهم العائلة والدراري، إن شاء الله كامل بخير.",
        "profile": "kahwa",
        "english": "How are you brother, how are the family and kids, hopefully all is well.",
    },
    {
        "id": "phrase_02_morning_rush",
        "category": "Conversational / Daily Life",
        "text": "صباح الخير عليكم، اليوم كاين خدمة بزاف، لازم نزربو شوية باش نلحقو.",
        "profile": "kahwa",
        "english": "Good morning to you all, today there is a lot of work, we have to hurry a bit to arrive on time.",
    },
    {
        "id": "phrase_03_coffee_walk",
        "category": "Conversational / Leisure",
        "text": "صحا قهوتك، واش رايك نروحو نتمشاو شوية فالعشية كي تبرد الحالة؟",
        "profile": "kahwa",
        "english": "Enjoy your coffee, what do you think about going for a walk in the evening when it cools down?",
    },
    {
        "id": "phrase_04_missed_you",
        "category": "Conversational / Warmth",
        "text": "والله غير توحشناك يا صاحبي، شحال هادي ما تلاقينا وما قصرنا كيف كيف.",
        "profile": "kahwa",
        "english": "I swear we missed you my friend, it's been so long since we met and chatted together.",
    },

    # ── Category 2: Tech, Social Media & Code-Switching (Kahwa Profile) ───────
    {
        "id": "phrase_05_whatsapp_msg",
        "category": "Tech / Code-Switching",
        "text": "بعثتلك ميساج فالواتساب، شوفو و ريبونديلي كي تكون ديسبونيبل يرحم والديك.",
        "profile": "kahwa",
        "english": "I sent you a message on WhatsApp, check it and reply to me when you're available please.",
    },
    {
        "id": "phrase_06_slow_connection",
        "category": "Tech / Code-Switching",
        "text": "الكونكسيون اليوم راهي ثقيلة بزاف، التيليشارجومون حابس قاع ومقدرتش نخدم.",
        "profile": "kahwa",
        "english": "The connection today is very slow, download is completely frozen and I couldn't work.",
    },
    {
        "id": "phrase_07_practical_app",
        "category": "Tech / Dialectal Slang",
        "text": "الأبليكاسيون هادي جديدة و براتيك، تعاونك تنظم الوقت تاعك كل يوم بلا تكسار راس.",
        "profile": "kahwa",
        "english": "This application is new and practical, helps you organize your time every day without headache.",
    },
    {
        "id": "phrase_08_team_meeting",
        "category": "Workplace / Code-Switching",
        "text": "غدوة إن شاء الله عندنا ريونيون مع ليكيب، لازم نوجدو البروجي قبل الموعد.",
        "profile": "kahwa",
        "english": "Tomorrow inshallah we have a meeting with the team, we must prepare the project before deadline.",
    },

    # ── Category 3: Commerce & Street Inquiries (Kahwa Profile) ───────────────
    {
        "id": "phrase_09_market_price",
        "category": "Commerce / Daily Life",
        "text": "شحال يدير هاد الكيلو تاع الطماطيش، و عندك صرف تاع ألفين دينار؟",
        "profile": "kahwa",
        "english": "How much is this kilo of tomatoes, and do you have change for two thousand dinars?",
    },
    {
        "id": "phrase_10_post_office",
        "category": "Navigation / Question",
        "text": "وين راهي البوسطة القريبة منا، نقدر نروح ليها على رجلية ولا بعيدة ولازم طاكسي؟",
        "profile": "kahwa",
        "english": "Where is the nearest post office, can I walk there or is it far and needs a taxi?",
    },
    {
        "id": "phrase_11_bus_station",
        "category": "Transport / Maghrebi Terms",
        "text": "وقتاش يقلع الكار تاع وهران، مازال كاين بلايص ولا خلاصو كامل التواكر؟",
        "profile": "kahwa",
        "english": "When does the bus to Oran depart, are there still seats or are all tickets sold out?",
    },
    {
        "id": "phrase_12_clarification",
        "category": "Polite Inquiry",
        "text": "سمحلي خويا ما فهمتش واش قصدك، عاود فهمني بالعقل يرحم والديك.",
        "profile": "kahwa",
        "english": "Excuse me brother I didn't understand what you meant, explain to me again slowly please.",
    },

    # ── Category 4: Algerian Proverbs & Cultural Wisdom (Rawi Profile) ────────
    {
        "id": "phrase_13_proverb_speech",
        "category": "Proverb / Oral Heritage",
        "text": "الحديث قياس، والفاهم يفهم بالغمزة، والغافل حتى تدق فودنو.",
        "profile": "rawi",
        "english": "Speech has its measure: the wise understands with a wink, the heedless needs knocking on his ear.",
    },
    {
        "id": "phrase_14_proverb_past",
        "category": "Proverb / Wisdom",
        "text": "اللي فات مات، واللي راح ما يولي، تهلى فاليوم وخدم للغدوى باش تنجح.",
        "profile": "rawi",
        "english": "What has passed is dead, what is gone won't return; take care of today and work for tomorrow to succeed.",
    },
    {
        "id": "phrase_15_proverb_patience",
        "category": "Proverb / Patience",
        "text": "الصبر مفتاح الفرج، كل عطلة فيها خير، والشدة ما تدوم لحتى واحد فهاد الدنيا.",
        "profile": "rawi",
        "english": "Patience is key to relief, every delay carries a blessing, and hardship never lasts for anyone.",
    },
    {
        "id": "phrase_16_proverb_integrity",
        "category": "Proverb / Guidance",
        "text": "خالط العاقل تكسب عقلو، وما تمشيش مع الجاهل اللي يضيعك فالطريق.",
        "profile": "rawi",
        "english": "Keep company with the wise to gain his wisdom, and do not walk with the ignorant who misleads you.",
    },

    # ── Category 5: Folklore Narrative & Storytelling (Rawi Profile) ──────────
    {
        "id": "phrase_17_fairytale_intro",
        "category": "Folklore / Story Opener",
        "text": "كان يا ما كان في قديم الزمان، كان كاين سلطان عادل يحب الخير و يعاون قاع ناسو.",
        "profile": "rawi",
        "english": "Once upon a time in ancient days, there was a just sultan who loved good and helped all his people.",
    },
    {
        "id": "phrase_18_goha_forest",
        "category": "Folklore / Narrative Suspense",
        "text": "خرج جحا للغابة فالصباح الباكر، وفي نص الطريق سمع صوت غريب ورا الشجرة الكبيرة.",
        "profile": "rawi",
        "english": "Goha went to the forest early in the morning, and midway he heard a strange sound behind the big tree.",
    },
    {
        "id": "phrase_19_traditional_gathering",
        "category": "Cultural / Heritage",
        "text": "في بلادنا كاين تقاليد عريقة، القعدة الزينة تاع زمان والقصايد اللي تحكي تاريخ الرجال الأحرار.",
        "profile": "rawi",
        "english": "In our homeland there are ancient traditions, the gatherings of old and poems telling history of free men.",
    },
    {
        "id": "phrase_20_sunset_village",
        "category": "Descriptive / Atmospheric",
        "text": "الشمس راهي تغرب ورا الجبال العالية، والهدوء سكن القرية مع وقت صوت أذان المغرب.",
        "profile": "rawi",
        "english": "The sun is setting behind the high mountains, and tranquility settled over the village with the evening adhan.",
    },
]

print(f"Configured {len(WINNING_CONFIGS)} primary test configurations and {len(BENCHMARK_20_PHRASES)} benchmark phrases.")


# =============================================================================
# Cell 7 — Execute Primary Winning Configurations
# =============================================================================

print("=" * 65)
print(f"RUNNING PRIMARY TESTS ({len(WINNING_CONFIGS)} configurations)")
print("Target: 'السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو هادا test للمودال الجديد، أدخلو جربوه و قولولي'")
print("=" * 65)

winning_results = {}

for c in WINNING_CONFIGS:
    config_id = c["id"]
    title     = c["title"]
    domain    = c["ref_domain"]
    text      = c["text"]
    nfe       = c["nfe_steps"]
    cfg       = c["cfg_strength"]
    speed     = c["speed"]

    print(f"\n>>> {title}")
    print(f"    Profile    : {c['profile']}")
    print(f"    Text       : {text}")
    print(f"    Parameters : Domain={domain} | NFE={nfe} | CFG={cfg} | Speed={speed}")

    if domain not in ref_samples:
        domain = list(ref_samples.keys())[0] if ref_samples else None
    if not domain:
        print(f"    [SKIP] No reference available for {config_id}")
        continue

    ref_info = ref_samples[domain]

    t0 = time.time()
    out_path = synthesize(
        gen_text    = text,
        ref_audio   = ref_info["audio_path"],
        ref_text    = ref_info.get("text", ""),
        ckpt_file   = darja_ckpt,
        output_name = config_id,
        nfe_steps   = nfe,
        cfg_strength= cfg,
        speed       = speed,
    )
    elapsed = time.time() - t0

    if out_path and Path(out_path).exists():
        size_kb = Path(out_path).stat().st_size / 1024
        winning_results[config_id] = {
            "config": c,
            "path": out_path,
            "elapsed_s": round(elapsed, 2),
            "size_kb": round(size_kb, 1),
        }
        print(f"    [OK] Finished in {elapsed:.1f}s ({size_kb:.0f} KB) -> {Path(out_path).name}")
    else:
        print(f"    [FAIL] Synthesis failed for {config_id}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nPrimary tests complete: {len(winning_results)} samples generated.")


# =============================================================================
# Cell 8 — Interactive Audio Gallery: Primary Outputs
# =============================================================================

print("\n" + "=" * 65)
print("INTERACTIVE AUDIO GALLERY — PRIMARY WINNING OUTPUTS")
print("Target: 'السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، كيما راكم تشوفو هادا test للمودال الجديد، أدخلو جربوه و قولولي'")
print("=" * 65)

for c in WINNING_CONFIGS:
    config_id = c["id"]
    if config_id not in winning_results:
        continue
    info = winning_results[config_id]

    print(f"\n{'-'*65}")
    print(f"{c['title'].upper()}")
    print(f"{'-'*65}")
    print(f"  Profile           : {c['profile']}")
    print(f"  Description       : {c['description']}")
    print(f"  Text Fed          : {c['text']}")
    print(f"  Reference Voice   : {c['ref_domain'].upper()} ({ref_samples[c['ref_domain']]['label']})")
    print(f"  Configuration     : NFE={c['nfe_steps']} | CFG={c['cfg_strength']} | Speed={c['speed']}")
    print(f"  Runtime           : {info['elapsed_s']}s ({info['size_kb']} KB)")
    print(f"  WAV File          : {info['path']}")
    display(Audio(info["path"], rate=SAMPLE_RATE))


# =============================================================================
# Cell 9 — 20-Phrase Benchmark Runner (Winning Conditions)
# =============================================================================

# Configuration: Set how many phrases to synthesize (1 to 20)
RUN_BENCHMARK = True
BENCHMARK_COUNT = 20  # Set to 5 for quick check, or 20 for complete test

benchmark_results = {}

if RUN_BENCHMARK:
    print("\n" + "=" * 65)
    print(f"RUNNING BENCHMARK ON {min(BENCHMARK_COUNT, len(BENCHMARK_20_PHRASES))} PHRASES")
    print("=" * 65)

    phrases_to_run = BENCHMARK_20_PHRASES[:BENCHMARK_COUNT]

    for idx, item in enumerate(phrases_to_run, 1):
        p_id    = item["id"]
        p_cat   = item["category"]
        p_text  = item["text"]
        profile = item["profile"]

        # Route dynamically to the winning profile conditions
        if profile == "kahwa":
            nfe, cfg, spd, domain = 48, 1.8, 0.95, "kahwa"
        else:
            nfe, cfg, spd, domain = 48, 2.0, 0.92, "rawi"

        ref_info = ref_samples.get(domain) or list(ref_samples.values())[0]

        print(f"\n[{idx:02d}/{len(phrases_to_run):02d}] {p_cat.upper()} | Voice: {domain.upper()} (NFE={nfe}, CFG={cfg}, Speed={spd})")
        print(f"  Text : {p_text}")
        print(f"  Mean : {item['english']}")

        t0 = time.time()
        out = synthesize(
            gen_text    = p_text,
            ref_audio   = ref_info["audio_path"],
            ref_text    = ref_info.get("text", ""),
            ckpt_file   = darja_ckpt,
            output_name = f"benchmark_{p_id}",
            nfe_steps   = nfe,
            cfg_strength= cfg,
            speed       = spd,
        )
        elapsed = time.time() - t0

        if out and Path(out).exists():
            size_kb = Path(out).stat().st_size / 1024
            benchmark_results[p_id] = {
                "item": item,
                "path": out,
                "elapsed_s": round(elapsed, 2),
                "size_kb": round(size_kb, 1),
            }
            print(f"  [OK] Generated in {elapsed:.1f}s ({size_kb:.0f} KB)")
            display(Audio(out, rate=SAMPLE_RATE))
        else:
            print(f"  [FAIL] Failed: {p_id}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nBenchmark complete: {len(benchmark_results)} / {len(phrases_to_run)} phrases generated.")


# =============================================================================
# Cell 10 — Production Sandbox: Generate Any Custom Darja Sentence
# =============================================================================

def generate_darja(
    text: str,
    profile: str = "kahwa",  # 'kahwa' (Try 5 condition) or 'rawi' (Try 6 condition)
    tag: str = "custom",
):
    """
    Generate speech for any Darja text using strictly the winning conditions.

    Profiles:
      - 'kahwa' : Try 5 conditions (NFE=48, CFG=1.8, Speed=0.95, Kahwa podcast voice)
      - 'rawi'  : Try 6 conditions (NFE=48, CFG=2.0, Speed=0.92, Rawi male voice)
    """
    if profile == "kahwa":
        nfe, cfg, spd, domain = 48, 1.8, 0.95, "kahwa"
    elif profile == "rawi":
        nfe, cfg, spd, domain = 48, 2.0, 0.92, "rawi"
    else:
        nfe, cfg, spd, domain = 48, 1.8, 0.95, "loubna"

    ref_info = ref_samples.get(domain) or list(ref_samples.values())[0]

    out = synthesize(
        gen_text    = text,
        ref_audio   = ref_info["audio_path"],
        ref_text    = ref_info.get("text", ""),
        ckpt_file   = darja_ckpt,
        output_name = f"sandbox_{profile}_{tag}",
        nfe_steps   = nfe,
        cfg_strength= cfg,
        speed       = spd,
    )
    if out:
        print(f"\n[WINNING PROFILE: {profile.upper()} ({domain.capitalize()}) | NFE={nfe} | CFG={cfg} | Speed={spd}]")
        print(f"Text: {text}")
        display(Audio(out, rate=SAMPLE_RATE))
    return out

# Interactive testing example:
# generate_darja("واش راك خويا، واش راهم العائلة والدراري، إن شاء الله كامل بخير.", profile="kahwa")


# =============================================================================
# Cell 11 — Export, Package, and Create Downloadable Zip
# =============================================================================

FINAL_DIR = Path("/kaggle/working/f5tts_darja_winning_outputs")
FINAL_DIR.mkdir(exist_ok=True)

shutil.copytree(str(OUT_DIR), str(FINAL_DIR / "audio"), dirs_exist_ok=True)
shutil.copytree(str(REF_DIR), str(FINAL_DIR / "references"), dirs_exist_ok=True)

report = {
    "model": DARJA_REPO,
    "checkpoint": "model_last.pt",
    "total_training_steps": 48574,
    "primary_tests": [
        {
            "id": c["id"],
            "title": c["title"],
            "profile": c["profile"],
            "text": c["text"],
            "ref_domain": c["ref_domain"],
            "nfe_steps": c["nfe_steps"],
            "cfg_strength": c["cfg_strength"],
            "speed": c["speed"],
            "output_path": winning_results.get(c["id"], {}).get("path"),
            "elapsed_s": winning_results.get(c["id"], {}).get("elapsed_s"),
        }
        for c in WINNING_CONFIGS
    ],
    "benchmark_20_phrases": [
        {
            "id": item["id"],
            "category": item["category"],
            "text": item["text"],
            "profile": item["profile"],
            "english": item["english"],
            "output_path": benchmark_results.get(item["id"], {}).get("path"),
            "elapsed_s": benchmark_results.get(item["id"], {}).get("elapsed_s"),
        }
        for item in BENCHMARK_20_PHRASES[:BENCHMARK_COUNT]
    ],
}

report_path = FINAL_DIR / "benchmark_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nAll outputs packaged in: {FINAL_DIR}")
print(f"Total WAV files generated: {len(list(FINAL_DIR.rglob('*.wav')))}")
for f in sorted(FINAL_DIR.rglob("*.wav")):
    print(f"  {f.relative_to(FINAL_DIR)} ({f.stat().st_size // 1024} KB)")

# Create zip archive for single-click download from Kaggle output
zip_path = shutil.make_archive("/kaggle/working/f5tts_darja_winning_outputs", "zip", str(FINAL_DIR))
print(f"\nZip archive created: {zip_path} ({Path(zip_path).stat().st_size // 1024} KB)")
print("You can download this zip directly from the Kaggle Output tab.")
print("\n=== Evaluation & Benchmark Complete ===")
