from argparse import Namespace
import torch
import os

# Configuration for the entire training pipeline
config = Namespace(
    name = 'default',
    # data 
    data_py = 'data/mirror_seq.py', # Path to data module
    seq_len = 64,                 # Length of input sequences
    n_input_values = 8,           # Number of possible input values
    n_output_values = 8,          # Number of possible output values
    mask_token = 0,               # Token used for masking
    mask_frac = 0.1,             # Fraction of tokens to mask during training
    train_size = 1024*1000,        # Number of training examples
    val_size = 1024,              # Number of validation examples
    bs_train = 64,                # Batch size for training
    bs_val = 64,                  # Batch size for validation

    # model 
    model_py = 'model/longformer.py', # Path to model module
    n_layers = 8,                 # Number of transformer layers
    n_heads = 8,                  # Number of attention heads
    d_model = 64,                 # Model dimension
    d_ffn = 64*4,                 # Feed-forward network dimension
    d_id = 64,                    # Dimension of ID embeddings
    d_value = 64,                  # Dimension of value embeddings
    dropout = 0.1,                # Dropout rate
    dtype = torch.bfloat16,       # Data type for model parameters

    # longformer
    attention_window = 16,        # Attention window size
    attention_dilation = 1,       # Attention dilation

    # training 
    max_epochs = 20,              # Maximum number of training epochs
    strategy = 'ddp_find_unused_parameters_true', # Training strategy: 'ddp', 'deepspeed_stage_2', 'ddp_find_unused_parameters_true'
    precision = 'bf16-true',      # Precision for training: 'bf16-true', '16-mixed', '16-true', '32-true'
    lr = 1e-3,                    # Learning rate
    weight_decay = 1e-4,          # Weight decay for optimizer
    max_grad_norm = 0,            # Maximum gradient norm for clipping. 1.0 is conservative, 0.5 is aggressive, 0.0 is no clipping, 5.0 allows large gradients
    val_check_interval = 0.1,     # Fraction of training epoch after which to run validation
    log_every_n_steps = 50,       # Log metrics every N steps
    nnodes = 1,                   # Number of nodes for distributed training
    devices = 'auto',             # GPU ids to use ('auto' for all available)

)

# logging 
config.use_wandb = True
config.wandb = {
    'project': 'test',       # W&B project name
    'tags': [config.name],   # Tags for the run
    'group': 'mirror_seq',   # Group name for the run
}

# checkpoint 
config.outdir = f'results/{config.name}'           # Output directory for all results
config.ckpt_dir = f'results/{config.name}/ckpt'    # Directory for saving checkpoints
config.ckpt_path = f'results/{config.name}/ckpt/last.ckpt' # Path to specific checkpoint to load (set at runtime)
config.ckpt_resume = True,          # Whether to resume from checkpoint

# Create directories if they don't exist
os.makedirs(config.outdir, exist_ok=True)
os.makedirs(config.ckpt_dir, exist_ok=True)