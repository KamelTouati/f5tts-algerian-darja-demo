# F5-TTS: Algerian Arabic (Darja) Speech Synthesis Suite & Streamlit App

An end-to-end Text-to-Speech (TTS) adaptation framework and interactive Streamlit web application tailored for colloquial **Algerian Arabic (Darja)** using **F5-TTS** (Flow Matching with Diffusion Transformer).

Fine-tuned from [IbrahimSalah/Arabic-F5-TTS-v2](https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2) across ~399 hours of multi-domain Algerian dialectal speech on Hugging Face Hub: [touati-kamel/f5tts-algerian-darja](https://huggingface.co/touati-kamel/f5tts-algerian-darja).

---

## Features

- **Interactive Streamlit Web Interface**: Fast speech synthesis from text with waveform display and WAV download.
- **Winning Condition Profiles**:
  - **Kahwa Podcast (Conversational)**: Spontaneous everyday podcast cadence, calibrated for colloquial expressions and code-switching (NFE=48, CFG=1.8, Speed=0.95).
  - **Rawi Folklore (Male Storyteller)**: Deep, resonant oral narrative cadence (NFE=48, CFG=2.0, Speed=0.92).
  - **Loubna Stories (Studio Female)**: Clean studio-recorded voice with high signal-to-noise ratio.
- **Zero-Shot Voice Cloning**: Upload any 3–10 second audio clip to synthesize Darja in that speaker's voice.
- **20 Curated Benchmark Phrases**: One-click sample test phrases spanning conversational, technology, street inquiries, proverbs, and folklore.
- **No G2P Bottleneck**: Character-level modeling handles Maghrebi consonant clusters, short-vowel elisions, and French loanwords naturally.

---

## Directory Structure

```text
├── app.py              # Streamlit Web Application (Production / Cloud Ready)
├── test_kaggle.py      # Automated 20-phrase Kaggle notebook evaluation runner
├── infer.py            # Command-line CLI synthesis script
├── train_kaggle.py     # Multi-session Kaggle background training runner with auto-sync
├── train.py            # Local / VM training pipeline runner
├── prepare_dataset.py  # OddAdmix audio preprocessing, filtering, and normalization
├── requirements.txt    # Python package dependencies
├── packages.txt       # Linux system dependencies (ffmpeg, libsndfile1)
└── README.md           # Documentation
```

---

## Quickstart: Run Locally

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/KamelTouati/f5tts-algerian-darja-demo.git
cd f5tts-algerian-darja-demo

pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```

The application will automatically download the checkpoint (`model_last.pt`, step 48,574) and reference voices on first run.

---

## Deploying to the Cloud

### Option A: Streamlit Community Cloud (Free, via GitHub)

1. Push this repository to your GitHub account (`https://github.com/KamelTouati/f5tts-algerian-darja-demo`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and `packages.txt`.

### Option B: Hugging Face Spaces (Streamlit + ZeroGPU)

Hugging Face Spaces provides dynamic access to free NVIDIA A100/H100 GPUs via ZeroGPU:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK.
3. Select **ZeroGPU (free)** as the hardware accelerator.
4. Push the files in this directory to your Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/f5tts-darja-demo
   git push space main
   ```
5. `app.py` automatically detects Hugging Face Spaces and applies the `@spaces.GPU` decorator for accelerated sub-second synthesis.

---

## Command-Line Inference (`infer.py`)

To synthesize speech directly from the terminal:

```bash
python infer.py \
    --gen_text "السلام عليكم خاوتي، وش أحوالكم إن شاء الله راكم ملاح، هادا تيست للمودال الجديد." \
    --output_file "./output_darja.wav" \
    --nfe_steps 48 \
    --cfg_strength 1.8 \
    --speed 0.95
```

---

## Model & Architecture Summary

| Property | Specification |
|---|---|
| **Base Model** | [IbrahimSalah/Arabic-F5-TTS-v2](https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2) |
| **Fine-Tuned Checkpoint** | [touati-kamel/f5tts-algerian-darja](https://huggingface.co/touati-kamel/f5tts-algerian-darja) (`model_last.pt`) |
| **Architecture** | Flow Matching DiT (`8_18` configuration: dim=1024, depth=22, heads=18) |
| **Vocoder** | Vocos (24,000 Hz, 100 mel channels) |
| **Training Steps** | 48,574 updates (8 Kaggle GPU sessions) |
| **Dataset** | ~399h OddAdmix Algerian Speech (Kahwa Podcast, Loubna Stories, Rawi Folklore) |
| **Final Loss** | 0.4950 (CFM vector field MSE) |

---

## The 20 Curated Benchmark Phrases

| Category | Darja Phrase | Recommended Profile |
|---|---|---|
| **Greeting** | واش راك خويا، واش راهم العائلة والدراري، إن شاء الله كامل بخير. | Kahwa (Conversational) |
| **Workplace** | صباح الخير عليكم، اليوم كاين خدمة بزاف، لازم نزربو شوية باش نلحقو. | Kahwa (Conversational) |
| **Casual** | صحا قهوتك، واش رايك نروحو نتمشاو شوية فالعشية كي تبرد الحالة؟ | Kahwa (Conversational) |
| **Warmth** | والله غير توحشناك يا صاحبي، شحال هادي ما تلاقينا وما قصرنا كيف كيف. | Kahwa (Conversational) |
| **Tech** | بعثتلك ميساج فالواتساب، شوفو و ريبونديلي كي تكون ديسبونيبل يرحم والديك. | Kahwa (Conversational) |
| **Tech** | الكونكسيون اليوم راهي ثقيلة بزاف، التيليشارجومون حابس قاع ومقدرتش نخدم. | Kahwa (Conversational) |
| **Tech** | الأبليكاسيون هادي جديدة و براتيك، تعاونك تنظم الوقت تاعك كل يوم بلا تكسار راس. | Kahwa (Conversational) |
| **Workplace** | غدوة إن شاء الله عندنا ريونيون مع ليكيب، لازم نوجدو البروجي قبل الموعد. | Kahwa (Conversational) |
| **Commerce** | شحال يدير هاد الكيلو تاع الطماطيش، و عندك صرف تاع ألفين دينار؟ | Kahwa (Conversational) |
| **Navigation** | وين راهي البوسطة القريبة منا، نقدر نروح ليها على رجلية ولا بعيدة ولازم طاكسي؟ | Kahwa (Conversational) |
| **Transport** | وقتاش يقلع الكار تاع وهران، مازال كاين بلايص ولا خلاصو كامل التواكر؟ | Kahwa (Conversational) |
| **Inquiry** | سمحلي خويا ما فهمتش واش قصدك، عاود فهمني بالعقل يرحم والديك. | Kahwa (Conversational) |
| **Proverb** | الحديث قياس، والفاهم يفهم بالغمزة، والغافل حتى تدق فودنو. | Rawi (Storyteller) |
| **Proverb** | اللي فات مات، واللي راح ما يولي، تهلى فاليوم وخدم للغدوى باش تنجح. | Rawi (Storyteller) |
| **Proverb** | الصبر مفتاح الفرج، كل عطلة فيها خير، والشدة ما تدوم لحتى واحد فهاد الدنيا. | Rawi (Storyteller) |
| **Proverb** | خالط العاقل تكسب عقلو، وما تمشيش مع الجاهل اللي يضيعك فالطريق. | Rawi (Storyteller) |
| **Folklore** | كان يا ما كان في قديم الزمان، كان كاين سلطان عادل يحب الخير و يعاون قاع ناسو. | Rawi (Storyteller) |
| **Folklore** | خرج جحا للغابة فالصباح الباكر، وفي نص الطريق سمع صوت غريب ورا الشجرة الكبيرة. | Rawi (Storyteller) |
| **Heritage** | في بلادنا كاين تقاليد عريقة، القعدة الزينة تاع زمان والقصايد اللي تحكي تاريخ الرجال الأحرار. | Rawi (Storyteller) |
| **Atmospheric** | الشمس راهي تغرب ورا الجبال العالية، والهدوء سكن القرية مع وقت صوت أذان المغرب. | Rawi (Storyteller) |

---

## Citation

```bibtex
@misc{touati2026f5ttsdarja,
  author = {Kamel Touati},
  title = {F5-TTS: Adapting Diffusion Transformers for Algerian Arabic (Darja) Speech Synthesis},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/touati-kamel/f5tts-algerian-darja}}
}
```
