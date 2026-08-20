#!/bin/bash
#SBATCH -A AIFAC_S07_041
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --job-name=ft_audiocaps_sca
#SBATCH -o ./slurm_scripts/logs/ft_audiocaps_sca_%j.out
#SBATCH -e ./slurm_scripts/logs/ft_audiocaps_sca_%j.out

# DISABLED -- this finetune was train-on-test.
#
# config/sca/finetune_cfg/retrieval-audiocaps.json pointed BOTH its training split and its
# validation split at benchmark_eval/audiocaps_tva_annotation.json, the 704-clip AudioCaps
# test annotation, with training=true. Anything it produced was trained on the data it was
# scored on, so the "SCA ft AudioCaps 51.6/50.6" row is invalid and has been quarantined.
#
# There is also no baseline to compare against: GRAM does not finetune AudioCaps at all --
# their Tab. 5 gives it no finetuning epochs and their AudioCaps numbers (Tab. 3) are
# zero-shot. AudioCaps belongs in the zero-shot table only.
#
# The config has been deleted rather than repaired: repairing it would need a genuine
# AudioCaps train split, which neither this repo nor the published protocol uses.
echo "REFUSING TO RUN: AudioCaps finetuning was train-on-test; see the comment in this file." >&2
echo "AudioCaps is a ZERO-SHOT benchmark in this family -- use the e1 zero-shot path." >&2
exit 2
