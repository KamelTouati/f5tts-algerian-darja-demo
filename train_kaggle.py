# =============================================================================
# Fine-Tuning F5-TTS — Darja Algérienne — KAGGLE BACKGROUND RUN (Version 7)
# =============================================================================
# Optimisations majeures v7 :
# - Fix TypeError : suppression du monkey-patch SafeDataLoader global (qui cassait
#   le generic typing class de StatefulDataLoader/DataLoaderShard sous Python 3.12).
#   Le DataLoader est configuré directement et nativement avec num_workers=0,
#   pin_memory=False, persistent_workers=False dans CleanTrainer.train().
# - Zéro crash mémoire hôte : 0 sous-processus worker = aucune prolifération de RAM.
# - Déchargement propre de skipped_dataloader dès la fin de l'époque de reprise.
# - load_checkpoint robuste avec weights_only=False et try/except sur l'optimizer.
# - Conversion du dataset en liste mémoire pure (~3 Mo au lieu de PyArrow C++ objects).
# - Nettoyage périodique explicite (gc.collect + torch.cuda.empty_cache) à chaque époque.
# - Thread Hub Syncer avec détection mtime sur model_last.pt et uploads non-bloquants.
# - Reprise automatique depuis votre Hugging Face Hub (model_last.pt / update 500).
# =============================================================================

import os
import sys
import subprocess

# Éviter la fragmentation mémoire CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 1. Installation & Mise à jour AVANT tout import Hugging Face ─────────────
print("Verification et mise a jour des dependances F5-TTS...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "hf_xet"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "huggingface_hub>=0.25.0", "f5-tts", "accelerate", "soundfile", "datasets", "bitsandbytes"
], check=False)
print("Dependances pretes.", flush=True)

import gc
import re
import glob
import time
import json
import shutil
import string
import argparse
import threading
from pathlib import Path

import soundfile as sf
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
from datasets.arrow_writer import ArrowWriter
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login as hf_login

# ── 2. Configuration des chemins et disques Kaggle ────────────────────────────
KAGGLE_WORKING  = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./kaggle_working")
KAGGLE_TMP      = "/tmp" if os.path.exists("/tmp") else os.path.abspath("./kaggle_tmp")

HF_CACHE_DIR    = os.path.join(KAGGLE_TMP, "hf_home")
DATASET_DIR     = os.path.join(KAGGLE_TMP, "f5_dataset_darja")
SCRATCH_DIR     = os.path.join(KAGGLE_TMP, "scratch")
TORCH_DIR       = os.path.join(KAGGLE_TMP, "torch_home")
WANDB_DIR       = os.path.join(KAGGLE_TMP, "wandb")
OUTPUT_CKPT_DIR = os.path.join(KAGGLE_WORKING, "ckpts", "darja")

for d in [HF_CACHE_DIR, DATASET_DIR, SCRATCH_DIR, TORCH_DIR, WANDB_DIR, OUTPUT_CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ["HF_HOME"]            = HF_CACHE_DIR
os.environ["TMPDIR"]             = SCRATCH_DIR
os.environ["TORCH_HOME"]         = TORCH_DIR
os.environ["WANDB_DIR"]          = WANDB_DIR
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ── 3. Secrets Kaggle & Authentification ──────────────────────────────────────
HF_USERNAME   = "touati-kamel"
HF_REPO_ID    = f"{HF_USERNAME}/f5tts-algerian-darja"
WANDB_PROJECT = "f5tts-algerian-darja"
DATASET_NAME  = "darja"

HF_TOKEN = os.environ.get("HF_TOKEN")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN") or HF_TOKEN
    WANDB_API_KEY = user_secrets.get_secret("wandb-api-key") or WANDB_API_KEY
    print("Secrets Kaggle charges avec succes.", flush=True)
except Exception as e:
    print(f"Mode hors-Kaggle ou secrets non trouves : {e}", flush=True)

if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("Connecte a Hugging Face.", flush=True)

if WANDB_API_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        print("Connecte a Weights & Biases.", flush=True)
    except Exception as e:
        print(f"Note WandB login : {e}", flush=True)

api = HfApi()

try:
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
    print(f"Repository cible HF valide : https://huggingface.co/{HF_REPO_ID}", flush=True)
except Exception as e:
    print(f"Note repo HF : {e}", flush=True)


# ── 4. Preprocessing & Normalisation du texte Darja pour TTS ──────────────────
TARGET_SAMPLE_RATE = 24000
MIN_DURATION_SEC   = 2.0
MAX_DURATION_SEC   = 12.0

_FR_TAG_RE   = re.compile(r"\[\s*(?:French|FR)\s*:\s*(.*?)\]", re.IGNORECASE)
_BRACKET_RE  = re.compile(r"\[.*?\]|<.*?>|\(.*?\)")
_TATWEEL     = "\u0640"
_NOISE_CHARS = re.compile(r"[*#@~_\^&%$+=/\\|{}\[\]`\"«»]")


def normalize_darja_tts_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = _FR_TAG_RE.sub(r"\1", text)
    text = _BRACKET_RE.sub("", text)
    text = _NOISE_CHARS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.replace("\u0625", "\u0627").replace("\u0623", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064A")
    return " ".join(text.split()).strip()


def prepare_dataset_if_needed(dataset_ids: list = None, max_samples_per_ds: int = None):
    """
    Extrait les fichiers WAV à 24kHz et construit directement raw.arrow,
    duration.json et vocab.txt dans /tmp (sans toucher au quota /kaggle/working).
    """
    meta_file     = os.path.join(DATASET_DIR, "metadata.csv")
    arrow_file    = os.path.join(DATASET_DIR, "raw.arrow")
    duration_file = os.path.join(DATASET_DIR, "duration.json")
    vocab_file    = os.path.join(DATASET_DIR, "vocab.txt")

    # 1. Vocabulaire compatible 2580 tokens d'origine arabe
    if not os.path.exists(vocab_file):
        print("Telechargement du vocab.txt compatible depuis IbrahimSalah/Arabic-F5-TTS-v2...", flush=True)
        hf_hub_download(
            repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
            filename="vocab.txt",
            local_dir=DATASET_DIR,
        )

    if os.path.exists(meta_file) and os.path.exists(arrow_file) and os.path.exists(duration_file):
        print(f"Dataset deja pret dans {DATASET_DIR}.", flush=True)
        return

    if dataset_ids is None:
        dataset_ids = [
            "oddadmix/arabic-audio-collection-algerian-loubna-stories",
            "oddadmix/arabic-audio-collection-algerian-rawi",
            "oddadmix/arabic-audio-collection-algerian-kahwa-postcast",
        ]

    wavs_dir = os.path.join(DATASET_DIR, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    metadata = []
    arrow_records = []
    durations_list = []
    total_count = 0

    print(f"\nPreparation multi-domaines de {len(dataset_ids)} sous-ensembles Darja...", flush=True)
    for ds_id in dataset_ids:
        prefix = "loubna" if "loubna" in ds_id else ("rawi" if "rawi" in ds_id else "kahwa")
        print(f"\n--- Chargement de {prefix.upper()} ({ds_id}) ---", flush=True)

        ds = load_dataset(ds_id, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

        count = 0
        for idx, sample in enumerate(ds):
            if max_samples_per_ds and count >= max_samples_per_ds:
                break

            audio_obj = sample.get("audio")
            if not audio_obj:
                continue

            waveform = audio_obj["array"]
            sample_rate = audio_obj["sampling_rate"]
            duration = len(waveform) / sample_rate

            if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
                continue

            raw_text = sample.get("transcript_text") or sample.get("text") or ""
            clean_text = normalize_darja_tts_text(raw_text)

            if len(clean_text.split()) < 3:
                continue

            wav_name = f"{prefix}_{idx:06d}.wav"
            wav_path = os.path.join(wavs_dir, wav_name)

            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                waveform = np.mean(waveform, axis=1)

            sf.write(wav_path, waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")

            dur_rounded = round(duration, 3)
            abs_wav_path = os.path.abspath(wav_path)

            metadata.append(f"wavs/{wav_name}|{clean_text}|{dur_rounded}")
            arrow_records.append({
                "audio_path": abs_wav_path,
                "text": clean_text,
                "duration": dur_rounded
            })
            durations_list.append(dur_rounded)

            count += 1
            total_count += 1
            if count % 200 == 0:
                print(f"   [{prefix}] {count} fichiers WAV ecrits dans {wavs_dir}...", flush=True)

        print(f"   [{prefix}] Termine : {count} echantillons valides.", flush=True)

    print(f"\nTotal global echantillons prepares : {total_count}", flush=True)

    # 2. Écriture de metadata.csv
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("audio_path|text|duration\n" + "\n".join(metadata) + "\n")

    # 3. Écriture directe et sécurisée de raw.arrow via ArrowWriter
    print(f"Ecriture de {len(arrow_records)} enregistrements dans raw.arrow...", flush=True)
    writer = ArrowWriter(path=arrow_file)
    for rec in arrow_records:
        writer.write(rec)
    writer.finalize()
    print(f"raw.arrow finalise ({os.path.getsize(arrow_file) / 1e6:.1f} MB).", flush=True)

    # 4. Écriture directe de duration.json
    print("Ecriture de duration.json...", flush=True)
    with open(duration_file, "w", encoding="utf-8") as f:
        json.dump({"duration": durations_list}, f)
    print("duration.json finalise.", flush=True)
    print("Dataset multi-domaines complet pret pour l'entrainement.", flush=True)

    # 5. Nettoyage immédiat du cache streaming Hugging Face pour libérer de l'espace dans /tmp
    if os.path.exists(HF_CACHE_DIR):
        shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        print("Cache de telechargement Hugging Face nettoye.", flush=True)


# ── 5. Gestion des Checkpoints (Reprise Hub vs Base Arabe) ───────────────────
def setup_initial_checkpoint():
    """
    1. Vérifie si un checkpoint existe déjà sur Hugging Face Hub touati-kamel/f5tts-algerian-darja.
       Si oui, télécharge model_last.pt directement dans OUTPUT_CKPT_DIR pour reprendre
       l'entraînement là où il s'est arrêté (ex: update 500).
    2. Sinon, télécharge model_547500_8_18.pt, supprime les compteurs step/update pour le
       démarrage propre du fine-tuning, et le place comme pretrained_model_...
    """
    os.makedirs(OUTPUT_CKPT_DIR, exist_ok=True)

    # 1. Vérification de reprise sur notre repo personnel HF
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID, token=HF_TOKEN)
        pts = [f for f in files if f.startswith("last-checkpoint/") and f.endswith(".pt")]
        if pts:
            # Priorité à model_last.pt si présent, sinon le plus récent
            if "last-checkpoint/model_last.pt" in pts:
                latest_ckpt_name = "last-checkpoint/model_last.pt"
            else:
                latest_ckpt_name = sorted(pts)[-1]

            fname = os.path.basename(latest_ckpt_name)
            target_path = os.path.join(OUTPUT_CKPT_DIR, fname)
            print(f"Reprise detectee sur votre Hub : {latest_ckpt_name}", flush=True)
            hf_hub_download(repo_id=HF_REPO_ID, filename=latest_ckpt_name, local_dir=OUTPUT_CKPT_DIR, token=HF_TOKEN)
            nested = os.path.join(OUTPUT_CKPT_DIR, "last-checkpoint", fname)
            if os.path.exists(nested):
                shutil.move(nested, target_path)
                shutil.rmtree(os.path.join(OUTPUT_CKPT_DIR, "last-checkpoint"), ignore_errors=True)
            print(f"Checkpoint precedent pret pour reprise : {target_path}", flush=True)
            return
    except Exception as e:
        print(f"Pas de checkpoint precedent sur le Hub : {e}", flush=True)

    # 2. Checkpoint de départ : IbrahimSalah/Arabic-F5-TTS-v2
    pretrained_target = os.path.join(OUTPUT_CKPT_DIR, "pretrained_model_547500_8_18.pt")
    if not os.path.exists(pretrained_target):
        print("Telechargement du checkpoint de base : IbrahimSalah/Arabic-F5-TTS-v2 (model_547500_8_18.pt)...", flush=True)
        tmp_base = os.path.join(KAGGLE_TMP, "base_ckpt")
        os.makedirs(tmp_base, exist_ok=True)
        base_dl = hf_hub_download(
            repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
            filename="model_547500_8_18.pt",
            local_dir=tmp_base,
        )

        print("Preparation du checkpoint de depart (reinitialisation step/update pour fine-tuning)...", flush=True)
        import torch
        ckpt = torch.load(base_dl, map_location="cpu", weights_only=True)
        # Supprimer les anciens compteurs d'étapes de pré-entraînement
        if "step" in ckpt:
            del ckpt["step"]
        if "update" in ckpt:
            del ckpt["update"]
        if "optimizer_state_dict" in ckpt:
            del ckpt["optimizer_state_dict"]
        if "scheduler_state_dict" in ckpt:
            del ckpt["scheduler_state_dict"]

        torch.save(ckpt, pretrained_target)
        del ckpt
        shutil.rmtree(tmp_base, ignore_errors=True)
        print(f"Checkpoint de base pret pour fine-tuning : {pretrained_target}", flush=True)


# ── 6. Thread de Sauvegarde Asynchrone vers Hugging Face Hub ──────────────────
def start_hub_syncer(interval_sec: int = 180):
    """
    Thread d'arrière-plan qui surveille le dossier des checkpoints et upload
    les nouveaux checkpoints ainsi que model_last.pt vers HF Hub sans bloquer l'entraînement.
    """
    def syncer_loop():
        uploaded_mtimes = {}
        while True:
            time.sleep(interval_sec)
            try:
                pt_files = glob.glob(os.path.join(OUTPUT_CKPT_DIR, "*.pt"))
                for pt in pt_files:
                    fname = os.path.basename(pt)
                    if fname.startswith("pretrained_"):
                        continue
                    mtime = os.path.getmtime(pt)
                    # Upload si nouveau ou si model_last.pt a été mis à jour
                    if fname not in uploaded_mtimes or (fname == "model_last.pt" and mtime > uploaded_mtimes[fname] + 15):
                        s1 = os.path.getsize(pt)
                        if s1 < 100_000_000:
                            continue
                        time.sleep(3)
                        if os.path.getsize(pt) != s1:
                            continue

                        print(f"\n[Hub Syncer] Upload de {fname} ({s1 / 1e9:.2f} Go) vers {HF_REPO_ID}...", flush=True)
                        gc.collect()
                        api.upload_file(
                            path_or_fileobj=pt,
                            path_in_repo=f"last-checkpoint/{fname}",
                            repo_id=HF_REPO_ID,
                            token=HF_TOKEN,
                            commit_message=f"Kaggle checkpoint {fname}",
                        )
                        uploaded_mtimes[fname] = mtime
                        gc.collect()
                        print(f"[Hub Syncer] Upload reussi pour {fname} !", flush=True)
            except Exception as ex:
                print(f"[Hub Syncer] Erreur upload non-fatale : {ex}", flush=True)

    t = threading.Thread(target=syncer_loop, daemon=True)
    t.start()


# ── 7. Génération du Script d'Entraînement Adapté à l'Architecture 8_18 ──────
def write_training_runner():
    """
    Génère le runner d'entraînement exécuté via `accelerate launch`.
    Intègre :
      - CleanTrainer avec configuration native de torch.utils.data.DataLoader
        (num_workers=0, pin_memory=False, persistent_workers=False sans monkey-patch).
      - load_checkpoint tolérant et complet (weights_only=False).
      - Libération immédiate du skipped_dataloader et garbage collection
        systématique à chaque frontière d'époque et toutes les 100 mises à jour.
      - Conversion du dataset en liste de dictionnaires purs (~3 Mo de RAM).
    """
    runner_path = os.path.join(KAGGLE_WORKING, "train_runner.py")
    vocab_path = os.path.join(DATASET_DIR, "vocab.txt")
    arrow_path = os.path.join(DATASET_DIR, "raw.arrow")
    duration_path = os.path.join(DATASET_DIR, "duration.json")

    code = f'''import os
import sys
import gc
import math
import json
import torch
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.utils.data
from datasets import Dataset as Dataset_
from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import CustomDataset, DynamicBatchSampler, collate_fn
from f5_tts.model.utils import get_tokenizer, exists


# ── CleanTrainer avec gestion rigoureuse de la mémoire et des types ──────────
class CleanTrainer(Trainer):
    def save_checkpoint(self, update, last=False):
        super().save_checkpoint(update, last=last)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if self.is_main:
            print(f"[F5-TTS] Chargement du checkpoint : {{latest_checkpoint}}")

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoint = load_file(f"{{self.checkpoint_path}}/{{latest_checkpoint}}", device="cpu")
            checkpoint = {{"ema_model_state_dict": checkpoint}}
        elif latest_checkpoint.endswith(".pt"):
            try:
                checkpoint = torch.load(
                    f"{{self.checkpoint_path}}/{{latest_checkpoint}}", weights_only=False, map_location="cpu"
                )
            except Exception:
                checkpoint = torch.load(
                    f"{{self.checkpoint_path}}/{{latest_checkpoint}}", map_location="cpu"
                )

        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint.get("ema_model_state_dict", {{}}):
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main and "ema_model_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if "model_state_dict" in checkpoint and key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            if "model_state_dict" in checkpoint:
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as ex:
                    if self.is_main:
                        print(f"[F5-TTS] Note reprise optimizer state : {{ex}}")
            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as ex:
                    if self.is_main:
                        print(f"[F5-TTS] Note reprise scheduler state : {{ex}}")
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {{
                k.replace("ema_model.", ""): v
                for k, v in checkpoint.get("ema_model_state_dict", {{}}).items()
                if k not in ["initted", "update", "step"]
            }}
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset, num_workers=0, resumable_with_seed=None):
        from torch.utils.data import SequentialSampler
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        generator = torch.Generator()
        if exists(resumable_with_seed):
            generator.manual_seed(resumable_with_seed)

        # Création directe et native de DataLoader avec num_workers=0
        # (sans aucun monkey-patching global qui corrompt le sous-typage Generic)
        if self.batch_size_type == "sample":
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type invalide : {{self.batch_size_type}}")

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = max(1, total_updates - warmup_updates)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )

        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            if skipped_batch > 0:
                skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
            else:
                skipped_dataloader = train_dataloader
        else:
            skipped_epoch = 0
            skipped_dataloader = None

        if self.accelerator.is_local_main_process:
            print(f"[Trainer] Demarrage effectif : start_update={{start_update}}, start_epoch={{skipped_epoch + 1}}/{{self.epochs}}")

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch and skipped_dataloader is not None:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {{epoch + 1}}/{{self.epochs}}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=f"{{loss.item():.4f}}")

                    if global_update % 100 == 0:
                        gc.collect()

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {{"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}}, step=global_update
                    )

                if self.last_per_updates and global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if self.save_per_updates and global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

            # Libération immédiate du dataloader sauté pour libérer la mémoire
            if epoch == skipped_epoch and skipped_dataloader is not None:
                if skipped_dataloader is not train_dataloader:
                    del skipped_dataloader
                skipped_dataloader = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()


def run():
    vocab_file = r"{vocab_path}"
    arrow_file = r"{arrow_path}"
    duration_file = r"{duration_path}"
    ckpt_dir = r"{OUTPUT_CKPT_DIR}"

    # 1. Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    print(f"\\n[F5-TTS] Taille du vocabulaire : {{vocab_size}} tokens")

    # 2. Architecture DiT exacte pour IbrahimSalah/Arabic-F5-TTS-v2 (8_18)
    #    AVEC checkpoint_activations=True pour réduire drastiquement l'empreinte VRAM
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=18,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=False,
        conv_layers=8,
        pe_attn_head=1,
        checkpoint_activations=True,
    )

    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )

    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # 3. Optimiseur 8-bit AdamW de bitsandbytes
    try:
        import bitsandbytes
        use_bnb = True
        print("[F5-TTS] bitsandbytes detecte -> activation de bnb_optimizer (AdamW 8-bit).")
    except Exception:
        use_bnb = False
        print("[F5-TTS] bitsandbytes indisponible -> fallback sur AdamW standard.")

    # 4. CleanTrainer configuré pour la stabilité GPU T4
    use_wandb = "wandb" if os.environ.get("WANDB_API_KEY") else None
    trainer = CleanTrainer(
        model,
        epochs=50,
        learning_rate=2e-5,
        num_warmup_updates=2000,
        save_per_updates=500,
        keep_last_n_checkpoints=2,
        checkpoint_path=ckpt_dir,
        batch_size_per_gpu=2000,
        batch_size_type="frame",
        max_samples=32,
        grad_accumulation_steps=6,
        max_grad_norm=1.0,
        logger=use_wandb,
        wandb_project="{WANDB_PROJECT}",
        wandb_run_name="f5tts_darja_v1",
        last_per_updates=500,
        bnb_optimizer=use_bnb,
    )

    # 5. Conversion du dataset Arrow en liste Python pure (~3 Mo de RAM)
    print("[F5-TTS] Chargement du dataset en memoire optimisee...")
    raw_data = Dataset_.from_file(arrow_file)
    data_list = [
        {{"audio_path": r["audio_path"], "text": r["text"], "duration": r["duration"]}}
        for r in raw_data
    ]
    del raw_data
    gc.collect()

    with open(duration_file, "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]

    train_dataset = CustomDataset(
        data_list,
        durations=durations,
        preprocessed_mel=False,
        **mel_spec_kwargs,
    )

    print(f"[F5-TTS] Demarrage effectif de l'entrainement sur {{len(train_dataset)}} echantillons (num_workers=0)...")
    trainer.train(train_dataset, num_workers=0, resumable_with_seed=666)

if __name__ == "__main__":
    run()
'''
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(code)
    return runner_path


# ── 8. Entraînement Principal ────────────────────────────────────────────────
def main():
    print("=" * 70, flush=True)
    print("LANCEMENT DU RUN KAGGLE F5-TTS (ALGERIAN DARJA) - VERSION 7", flush=True)
    print(f"Working Dir : {KAGGLE_WORKING}", flush=True)
    print(f"Tmp Dir     : {KAGGLE_TMP} (Scratch space hors quota 20GB)", flush=True)
    print(f"Ckpt Dir    : {OUTPUT_CKPT_DIR}", flush=True)
    print("=" * 70, flush=True)

    # 1. Préparation du dataset (Loubna + Rawi + Kahwa)
    prepare_dataset_if_needed()

    # 2. Mise en place du checkpoint initial (reprise automatique depuis HF Hub si dispo)
    setup_initial_checkpoint()

    # 3. Synchroniseur asynchrone Hugging Face Hub (toutes les 3 min)
    start_hub_syncer(interval_sec=180)

    # 4. Écriture et exécution du runner d'entraînement
    runner_script = write_training_runner()

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "fp16",
        runner_script,
    ]

    print("\nLancement de l'entrainement distribue F5-TTS (Mixed Precision FP16)...", flush=True)
    print("Commande :", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # 5. Upload final du modèle complet sur Hugging Face
    print("\nEntrainement termine ! Upload du modele final sur Hugging Face Hub...", flush=True)
    try:
        api.upload_folder(
            folder_path=OUTPUT_CKPT_DIR,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            commit_message="Final trained F5-TTS Algerian Darja model",
        )
        print("Modele final uploade avec succes sur Hugging Face !", flush=True)
    except Exception as e:
        print(f"Erreur lors de l'upload final : {e}", flush=True)


if __name__ == "__main__":
    main()
