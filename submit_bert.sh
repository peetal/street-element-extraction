#!/bin/bash
#SBATCH --account=hulacon  
#SBATCH --partition=gpu 
#SBATCH --job-name=finetunebert 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1    
#SBATCH --mem=150GB  
#SBATCH --time=1-00:00:00  
#SBATCH --output=%x_%A_%a.log 

#conda activate jupyterlab-tf-pyt-20211020
python3 finetune_bert.py
