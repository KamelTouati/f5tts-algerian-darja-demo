# =============================================================================
# Inférence / Test Audio F5-TTS pour la Darja Algérienne
# =============================================================================

import os
import sys
import argparse
import subprocess
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Synthèse vocale F5-TTS en Darja Algérienne")
    parser.add_argument("--gen_text", type=str, required=True, help="Texte en Darja algérienne à synthétiser")
    parser.add_argument("--ref_audio", type=str, default=None, help="Fichier WAV de référence (3 à 10 secondes)")
    parser.add_argument("--ref_text", type=str, default="", help="Transcription du fichier audio de référence")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Chemin du checkpoint F5-TTS (.pt ou .safetensors)")
    parser.add_argument("--vocab_file", type=str, default=None, help="Chemin du fichier vocab.txt")
    parser.add_argument("--model_cfg", type=str, default=None, help="Chemin du fichier YAML d'architecture (ex: F5TTS_Base_8_18.yaml)")
    parser.add_argument("--output_file", type=str, default="./output_darja.wav", help="Fichier WAV de sortie généré")
    parser.add_argument("--nfe_steps", type=int, default=32, help="Nombre d'étapes d'échantillonnage Flow Matching (NFE)")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="Force du Classifier-Free Guidance (CFG)")
    parser.add_argument("--speed", type=float, default=1.0, help="Vitesse d'élocution (1.0 = normale)")

    args = parser.parse_args()

    # Si aucun checkpoint n'est spécifié, télécharger le modèle arabe de base
    if not args.ckpt_file:
        print("Aucun checkpoint local spécifié. Téléchargement du modèle de base IbrahimSalah/Arabic-F5-TTS-v2...")
        args.ckpt_file = hf_hub_download(
            repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
            filename="model_547500_8_18.pt",
        )
        if not args.vocab_file:
            args.vocab_file = hf_hub_download(
                repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
                filename="vocab.txt",
            )
        if not args.ref_audio:
            args.ref_audio = hf_hub_download(
                repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
                filename="reference.wav",
            )

    if not args.model_cfg:
        args.model_cfg = hf_hub_download(
            repo_id="IbrahimSalah/Arabic-F5-TTS-v2",
            filename="F5TTS_Base_8_18.yaml",
        )

    cmd = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "--model", "F5TTS_Base",
        "--model_cfg", os.path.abspath(args.model_cfg),
        "--ckpt_file", os.path.abspath(args.ckpt_file),
        "--gen_text", args.gen_text,
        "--output_file", os.path.abspath(args.output_file),
        "--nfe_step", str(args.nfe_steps),
        "--cfg_strength", str(args.cfg_strength),
        "--speed", str(args.speed),
    ]

    if args.vocab_file and os.path.exists(args.vocab_file):
        cmd.extend(["--vocab_file", os.path.abspath(args.vocab_file)])

    if args.ref_audio and os.path.exists(args.ref_audio):
        cmd.extend(["--ref_audio", os.path.abspath(args.ref_audio)])

    if args.ref_text:
        cmd.extend(["--ref_text", args.ref_text])

    print(f"\nGénération audio en cours pour : '{args.gen_text}'...")
    subprocess.run(cmd, check=True)
    print(f"Audio synthétisé avec succès -> {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
