from config.default import config

config.name = 'longformer'
config.model_py = 'model/longformer.py'
config.strategy = 'deepspeed_stage_2' # ddp, ddp_find_unused_parameters_true, deepspeed_stage_2
config.wandb['tags'] = [config.name]

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt' # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = True,          # Whether to resume from checkpoint

# Create directories if they don't exist
import os
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)