# =============================================================================
# Fine-Tuning F5-TTS — Algerian Arabic (Darja) Speech Synthesis
# Compatible avec Kaggle (T4/P100), Modal Cloud, ou GPU local
# Inspiré de train_sequential.py pour la structure des secrets, WandB et Hub
# =============================================================================

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login as hf_login

# -----------------------------------------------------------------------------
# 1. Configuration & Hyperparamètres
# -----------------------------------------------------------------------------
DEFAULT_PROJECT_NAME = "f5tts-algerian-darja"
DEFAULT_EXP_NAME     = "darja_loubna_v1"
BASE_ARABIC_MODEL    = "IbrahimSalah/Arabic-F5-TTS-v2"
BASE_EN_MODEL        = "SWivid/F5-TTS"

# Paramètres d'entraînement recommandés pour GPU 16GB-24GB (T4 / P100 / A10G)
DEFAULT_LEARNING_RATE     = 2e-5
DEFAULT_BATCH_SIZE_FRAMES = 3200  # Frame-based batching (évite les OOM)
DEFAULT_GRAD_ACCUM        = 4
DEFAULT_EPOCHS            = 50
DEFAULT_SAVE_STEP         = 500


def setup_credentials():
    """Récupère les tokens Hugging Face et WandB (Kaggle Secrets ou Environnement)."""
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN") or hf_token
        wandb_key = user_secrets.get_secret("wandb-api-key") or wandb_key
    except Exception:
        pass

    if hf_token:
        hf_login(token=hf_token)
        print("Authentification Hugging Face validée.")

    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            print("Authentification Weights & Biases (WandB) validée.")
        except Exception as e:
            print(f"Avertissement WandB : {e}")

    return hf_token, wandb_key


def download_base_checkpoint(base_model_id: str, ckpt_dir: str):
    """
    Télécharge le checkpoint pré-entraîné de base depuis Hugging Face.
    Si base_model_id est IbrahimSalah/Arabic-F5-TTS-v2, télécharge le modèle arabe.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\nTéléchargement du modèle de départ : {base_model_id}...")

    if "Arabic" in base_model_id:
        ckpt_file = hf_hub_download(
            repo_id=base_model_id,
            filename="model_547500_8_18.pt",
            local_dir=ckpt_dir,
        )
        vocab_file = hf_hub_download(
            repo_id=base_model_id,
            filename="vocab.txt",
            local_dir=ckpt_dir,
        )
        return ckpt_file, vocab_file
    else:
        # Modèle générique SWivid/F5-TTS
        ckpt_file = hf_hub_download(
            repo_id=base_model_id,
            filename="F5TTS_v1_Base/model_1250000.safetensors",
            local_dir=ckpt_dir,
        )
        return ckpt_file, None


def ensure_f5tts_installed():
    """Vérifie que f5-tts est installé dans l'environnement."""
    try:
        import f5_tts
        print("Package f5-tts détecté avec succès.")
    except ImportError:
        print("f5-tts non trouvé. Installation en cours via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "f5-tts", "accelerate"])
        print("Installation terminée.")


def prepare_dataset_arrow(dataset_dir: str):
    """
    Vérifie la présence de metadata.csv et utilise le script de conversion F5-TTS
    pour générer raw.arrow et duration.json si nécessaire.
    """
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {metadata_path}. "
            "Veuillez d'abord exécuter prepare_f5tts_darja.py pour créer le dataset."
        )

    arrow_path = os.path.join(dataset_dir, "raw.arrow")
    if os.path.exists(arrow_path):
        print(f"Dataset déjà pré-converti : {arrow_path}")
        return

    print("Conversion de metadata.csv en format Arrow F5-TTS...")
    try:
        from f5_tts.train.datasets.prepare_csv_wavs import prepare_csv_wavs
        prepare_csv_wavs(dataset_dir, dataset_dir)
        print("Conversion Arrow réussie.")
    except Exception as e:
        print(f"Note: Préparation Arrow directe via module : {e}")
        # Commande de fallback CLI
        cmd = [
            sys.executable,
            "-m", "f5_tts.train.datasets.prepare_csv_wavs",
            dataset_dir,
            dataset_dir
        ]
        subprocess.run(cmd, check=False)


def run_training(
    dataset_dir: str,
    output_dir: str,
    exp_name: str,
    pretrain_ckpt: str,
    vocab_file: str,
    learning_rate: float,
    batch_size_frames: int,
    grad_accum: int,
    epochs: int,
    save_step: int,
):
    """
    Lance l'entraînement F5-TTS via accelerate.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Commande de lancement F5-TTS finetune CLI
    cmd = [
        "accelerate", "launch",
        "-m", "f5_tts.train.finetune_cli",
        "--exp_name", exp_name,
        "--dataset_name", os.path.abspath(dataset_dir),
        "--finetune",
        "--pretrain", os.path.abspath(pretrain_ckpt),
        "--learning_rate", str(learning_rate),
        "--batch_size_per_gpu", str(batch_size_frames),
        "--batch_size_type", "frame",
        "--grad_accumulation_steps", str(grad_accum),
        "--epochs", str(epochs),
        "--save_step", str(save_step),
        "--tokenizer", "char",
    ]

    if vocab_file and os.path.exists(vocab_file):
        cmd.extend(["--vocab_file", os.path.abspath(vocab_file)])

    print("\n" + "=" * 60)
    print("Démarrage de l'entraînement F5-TTS :")
    print(f"  Expérience : {exp_name}")
    print(f"  Dataset    : {dataset_dir}")
    print(f"  Base Ckpt  : {pretrain_ckpt}")
    print(f"  Learning R : {learning_rate}")
    print(f"  Batch size : {batch_size_frames} frames")
    print(f"  Accum      : {grad_accum}")
    print("=" * 60 + "\n")

    print(f"Exécution : {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Script de Fine-Tuning F5-TTS pour Darja Algérienne")
    parser.add_argument("--dataset_dir", type=str, default="./f5tts_darja_dataset", help="Dossier contenant metadata.csv et wavs/")
    parser.add_argument("--output_dir", type=str, default="./ckpts_f5tts_darja", help="Dossier de sortie des checkpoints")
    parser.add_argument("--exp_name", type=str, default=DEFAULT_EXP_NAME, help="Nom de l'expérience")
    parser.add_argument("--base_model", type=str, default=BASE_ARABIC_MODEL, help="Modèle de base (IbrahimSalah/Arabic-F5-TTS-v2 ou SWivid/F5-TTS)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_frames", type=int, default=DEFAULT_BATCH_SIZE_FRAMES, help="Nombre de frames par batch GPU")
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Pas d'accumulation de gradients")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Nombre d'époques")
    parser.add_argument("--save_step", type=int, default=DEFAULT_SAVE_STEP, help="Fréquence de sauvegarde des checkpoints (steps)")

    args = parser.parse_args()

    ensure_f5tts_installed()
    setup_credentials()

    # 1. Vérification / préparation du dataset
    prepare_dataset_arrow(args.dataset_dir)

    # 2. Téléchargement du checkpoint de base
    ckpt_dir = os.path.join(args.output_dir, "base_ckpt")
    pretrain_ckpt, base_vocab = download_base_checkpoint(args.base_model, ckpt_dir)

    # Utilisation en priorité du vocabulaire extrait du dataset Darja
    darja_vocab = os.path.join(args.dataset_dir, "vocab.txt")
    vocab_to_use = darja_vocab if os.path.exists(darja_vocab) else base_vocab

    # 3. Lancement de l'entraînement
    run_training(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        pretrain_ckpt=pretrain_ckpt,
        vocab_file=vocab_to_use,
        learning_rate=args.lr,
        batch_size_frames=args.batch_frames,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        save_step=args.save_step,
    )


if __name__ == "__main__":
    main()
