import lightning.pytorch as pl
from trainer.pl_callbacks import best_ckpt_callback, confusion_plot_callback, plot_predictions_callback

# Lightning Trainer
def setup_trainer(config):
    """Set up and return the PyTorch Lightning Trainer
    """

    # setup logger
    if config.use_wandb:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project=config.wandb['project'], 
                             group=config.wandb['group'],
                             config=config, tags=config.wandb['tags'], 
                             save_code=True, 
                             save_dir=f'{config.outdir}', settings=wandb.Settings(code_dir="."), 
                             )
    else:
        from lightning.pytorch.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=config.outdir, name="tf_logs")

    # setup trainer
    trainer = pl.Trainer(
        # lightning specific
        strategy=config.strategy,
        num_nodes=config.nnodes,        
        devices='auto' if config.devices=='auto' else [int(i) for i in config.devices], 

        # shared between ray and pl
        precision=config.precision,
        max_epochs=config.max_epochs,
        accelerator='gpu',
        val_check_interval=config.val_check_interval, # fraction of train epoch
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.max_grad_norm, # 1.0 is conservative, 0.5 is aggressive, 0.0 is no clipping, 5.0 allows large gradients
        logger=logger,

        callbacks=[
            best_ckpt_callback(dirpath=config.ckpt_dir),
            confusion_plot_callback(config),
            plot_predictions_callback(config),
        ]
    )

    return trainer
