from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Callback
import torch
from trainer.plots import plot_confusion_matrix, plot_predictions

# Callbacks
def best_ckpt_callback(dirpath):
    return ModelCheckpoint(
        dirpath=dirpath,
        filename='{epoch:02d}-{val_acc:.5f}',
        save_top_k=3,
        monitor='val_acc',
        mode='max',
        save_last=True, # save the last checkpoint
        enable_version_counter=False, # enable saving multiple versions of the same result inside an epoch
    )

class confusion_plot_callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.confusion_matrices = []
        self.config = config

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.global_rank == 0:
            # Move to CPU immediately
            conf_matrix = outputs['confusion_matrix'].detach().cpu()
            # Store as lists
            self.confusion_matrices.append(conf_matrix)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0: 
            # calculate sum of confusion matrix
            all_confusion_matrices = torch.stack(self.confusion_matrices)
            sum_confusion_matrix = torch.sum(all_confusion_matrices, dim=0)
            # convert to numpy
            sum_confusion_matrix = sum_confusion_matrix.numpy()
            outdir = self.config.outdir
            plot_confusion_matrix(sum_confusion_matrix, outdir, trainer.current_epoch)

        # Clear the lists for the next epoch
        self.confusion_matrices.clear()

class plot_predictions_callback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Get a batch of data
            val_dataloaders = trainer.val_dataloaders
            if isinstance(val_dataloaders, list):
                dataloader = val_dataloaders[0]
            else:
                dataloader = val_dataloaders
            batch = next(iter(dataloader))

            # move to correct device
            batch = {k: v.to(pl_module.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}

            # model prediction
            # with torch.cuda.amp.autocast():
            x, id = batch['input'], batch['pos_id']
            logits = pl_module.model(x, id)
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            input = batch['input'].detach().cpu().numpy()
            target = batch['target'].detach().cpu().numpy()
            outdir = self.config.outdir
            output_path = f'{outdir}/predictions_epoch{trainer.current_epoch}.png'
            plot_predictions(output_path, input, target, pred)