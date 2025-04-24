#! /usr/bin/env python

# IMPORT LIBRARIES
import os
import torch
# Set deterministic behavior for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# IMPORT MY FUNCTIONS
from trainer.pl_data import data_wrapper
from trainer.pl_module import model_wrapper
from trainer.pl_trainer import setup_trainer

def load_module_from_path(file_path):
    module_namespace = {}
    with open(file_path, 'r') as file:
        exec(file.read(), module_namespace)
    return type('Module', (), module_namespace)

# Main execution function
def prepare(config):
    """Main function to run the training process."""

    # DATA
    data_module = load_module_from_path(config.data_py)
    theDataset = data_module.theDataset
    train_ds = theDataset(config, config.train_size)
    val_ds = theDataset(config, config.val_size)
    
    # MODEL
    model_module = load_module_from_path(config.model_py)
    theModel = model_module.theModel
    model = theModel(config)

    # wrap with pytorch lightning
    pl_data = data_wrapper(config, train_ds, val_ds)
    pl_model = model_wrapper(config, model) 
    pl_trainer = setup_trainer(config)

    return config, pl_data, pl_model, pl_trainer

def main(config):
    config, pl_data, pl_model, pl_trainer = prepare(config)
    train_loader = pl_data.train_dataloader()
    val_loader = pl_data.val_dataloader()
    if os.path.exists(config.ckpt_path) and config.ckpt_resume:
        print(f"Resuming from checkpoint: {config.ckpt_path}")
        pl_trainer.fit(pl_model, train_loader, val_loader, ckpt_path=config.ckpt_path)
    else:
        print("Starting from scratch")
        pl_trainer.fit(pl_model, train_loader, val_loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.py")
    args = parser.parse_args()
    config_module = load_module_from_path(args.config)
    config = config_module.config
    main(config)