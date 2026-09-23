# =============================================================================
# Preparation & Preprocessing du Dataset pour le Fine-Tuning de F5-TTS
# Algerian Arabic (Darja) Speech Synthesis
# Inspiré de la pipeline de train_sequential.py (Whisper Darja)
# =============================================================================

import os
import re
import string
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio

# -----------------------------------------------------------------------------
# 1. Constantes & Configuration
# -----------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 24000  # F5-TTS requiert nativement du 24 kHz mono
MIN_DURATION_SEC   = 2.0    # Chunks < 2s manquent de contexte prosodique
MAX_DURATION_SEC   = 12.0   # Chunks > 12s risquent l'OOM en Flow Matching / DiT

DATASET_CONFIGS = {
    "kahwa": {
        "dataset_id": "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
        "description": "Conversational Algerian podcast (multi-speaker)",
    },
    "loubna": {
        "dataset_id": "oddadmix/arabic-audio-collection-algerian-loubna-stories",
        "description": "Expressive female storytelling (ideal single-speaker TTS baseline)",
    },
    "rawi": {
        "dataset_id": "oddadmix/arabic-audio-collection-algerian-rawi",
        "description": "Cultural narratives and expressive folklore storytelling",
    },
}

# -----------------------------------------------------------------------------
# 2. Règles de Nettoyage et Normalisation du Texte Darja pour TTS
# -----------------------------------------------------------------------------
# Note importante pour TTS vs ASR:
# En ASR (Whisper), toute ponctuation est supprimée pour le calcul du WER.
# En TTS (F5-TTS), la ponctuation (virgule، point. point d'interrogation؟)
# est INDISPENSABLE pour que le modèle apprenne les pauses et l'intonation!

_FR_TAG_RE = re.compile(r"\[\s*(?:French|FR)\s*:\s*(.*?)\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[.*?\]|<.*?>|\(.*?\)")
_TATWEEL = "\u0640"

# Caractères indésirables à supprimer (symboles non prononçables)
_NOISE_SYMBOLS = re.compile(r"[*#@~_\^&%$+=/\\|{}\[\]`\"«»]")

# Ponctuations arabes et latines autorisées pour la prosodie TTS
ALLOWED_PUNCTUATION = {".", "!", "?", "،", "؛", "؟", ",", ";", ":"}


def normalize_darja_tts_text(text: str) -> str:
    """
    Nettoie et normalise le texte transcrit en Darja algérienne
    en préservant les marqueurs prosodiques pour la synthèse vocale.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Extraction du texte dans les balises de code-switching [French: ...]
    text = _FR_TAG_RE.sub(r"\1", text)

    # 2. Suppression des annotations de bruit non verbales [rires], <musique>, etc.
    text = _BRACKET_RE.sub("", text)

    # 3. Suppression des symboles parasites
    text = _NOISE_SYMBOLS.sub("", text)

    # 4. Suppression du tatweel / kashida (ـ)
    text = text.replace(_TATWEEL, "")

    # 5. Normalisation des variantes de l'Alif (harmonisation orthographique)
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")  # Alif Maqsura -> Ya

    # 6. Nettoyage des espaces multiples
    text = " ".join(text.split()).strip()

    return text


def is_valid_tts_sample(example: dict, min_words: int = 3) -> bool:
    """
    Vérifie la validité acoustique et textuelle pour F5-TTS.
    """
    duration = example.get("duration", 0)
    if duration is None or not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
        return False

    raw_text = example.get("transcript_text") or example.get("text") or ""
    clean_text = normalize_darja_tts_text(raw_text)

    # Doit contenir au minimum quelques mots pour avoir une phrase cohérente
    if len(clean_text.split()) < min_words:
        return False

    return True


# -----------------------------------------------------------------------------
# 3. Pipeline d'extraction et de formatage pour F5-TTS
# -----------------------------------------------------------------------------
def process_subset(
    subset_key: str,
    output_dir: str,
    max_samples: int = None,
    streaming: bool = True,
):
    """
    Traite un sous-ensemble audio, rééchantillonne à 24kHz et génère metadata.csv.
    """
    config = DATASET_CONFIGS[subset_key]
    dataset_id = config["dataset_id"]
    print(f"\n========================================================")
    print(f"Chargement du sous-ensemble : {subset_key.upper()} ({dataset_id})")
    print(f"Description : {config['description']}")
    print(f"========================================================")

    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Chargement du dataset avec rééchantillonnage natif en 24 000 Hz
    ds = load_dataset(dataset_id, split="train", streaming=streaming)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    metadata_records = []
    vocab_chars = set()

    count_processed = 0
    count_skipped = 0

    progress_bar = tqdm(total=max_samples if max_samples else 10000, desc=f"Traitement {subset_key}")

    for idx, sample in enumerate(ds):
        if max_samples and count_processed >= max_samples:
            break

        audio_obj = sample.get("audio")
        if not audio_obj:
            count_skipped += 1
            continue

        waveform = audio_obj["array"]
        sample_rate = audio_obj["sampling_rate"]

        # Calcul précis de la durée si absente des métadonnées
        duration = len(waveform) / sample_rate
        sample["duration"] = duration

        if not is_valid_tts_sample(sample):
            count_skipped += 1
            continue

        raw_text = sample.get("transcript_text") or sample.get("text") or ""
        clean_text = normalize_darja_tts_text(raw_text)

        # Enregistrement du fichier WAV 24kHz mono
        wav_filename = f"{subset_key}_{idx:06d}.wav"
        wav_path = os.path.join(wavs_dir, wav_filename)

        # Assurer un format float32 / int16 compatible soundfile
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # Si stéréo, convertir en mono
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)

        sf.write(wav_path, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")

        # F5-TTS attend dans metadata.csv : audio_path|text ou audio_path|text|duration
        # Chemin relatif vers le wav depuis la racine du dataset
        rel_audio_path = os.path.join("wavs", wav_filename).replace("\\", "/")
        metadata_records.append({
            "audio_path": rel_audio_path,
            "text": clean_text,
            "duration": round(duration, 3)
        })

        # Mise à jour du vocabulaire de caractères
        for char in clean_text:
            vocab_chars.add(char)

        count_processed += 1
        progress_bar.update(1)

    progress_bar.close()
    print(f"Extraction terminée pour {subset_key} :")
    print(f"  - Echantillons valides : {count_processed}")
    print(f"  - Echantillons rejetés : {count_skipped}")

    return metadata_records, vocab_chars


def build_vocab_file(vocab_chars: set, output_file: str):
    """
    Construit le fichier vocab.txt complet pour F5-TTS
    incluant l'alphabet arabe, chiffres, ponctuations, et caractères latins (code-switching).
    """
    sorted_chars = sorted(list(vocab_chars))
    with open(output_file, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            if char != " ":  # l'espace est généralement géré implicitement ou séparément
                f.write(f"{char}\n")
    print(f"Fichier de vocabulaire généré ({len(sorted_chars)} tokens) -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Préparation du dataset Darja pour F5-TTS")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["loubna"],
        choices=["loubna", "rawi", "kahwa", "all"],
        help="Sous-ensembles à extraire. 'loubna' est vivement conseillé pour commencer (voix studio féminine expressive).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./f5tts_darja_dataset",
        help="Dossier de destination du dataset préparé",
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=None,
        help="Nombre maximum d'échantillons par sous-ensemble (None = tout le dataset)",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Désactiver le streaming HF (télécharge tout d'abord)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_subsets = ["loubna", "rawi", "kahwa"] if "all" in args.subsets else args.subsets

    all_metadata = []
    combined_vocab = set()

    for subset in all_subsets:
        meta, chars = process_subset(
            subset_key=subset,
            output_dir=args.output_dir,
            max_samples=args.max_samples_per_subset,
            streaming=not args.no_streaming,
        )
        all_metadata.extend(meta)
        combined_vocab.update(chars)

    # 1. Écriture du fichier metadata.csv (séparateur pipe '|')
    metadata_csv_path = os.path.join(args.output_dir, "metadata.csv")
    print(f"\nÉcriture de {len(all_metadata)} lignes dans {metadata_csv_path}...")
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        f.write("audio_path|text|duration\n")
        for rec in all_metadata:
            f.write(f"{rec['audio_path']}|{rec['text']}|{rec['duration']}\n")

    # 2. Écriture du fichier vocab.txt
    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    build_vocab_file(combined_vocab, vocab_path)

    print("\nDataset F5-TTS préparé avec succès !")
    print(f"Total audio WAVs : {len(all_metadata)}")
    print(f"Dossier dataset  : {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
