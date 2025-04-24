from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix

# --- !!! Do not use data loader !!! ---
from data.mirror_seq import make_batch

# PyTorch Lightning module for model and trainer
class model_wrapper(LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        # Create separate metrics for training and validation
        self.train_acc = Accuracy(task='multiclass', num_classes=self.config.n_output_values)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.config.n_output_values)
        self.get_confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.config.n_output_values)

    def _calculate_loss_acc(self, logits, y, mask, metric):
        # select masked positions
        mask = mask.bool()        
        logits_masked = logits[mask, :]
        y_masked = y[mask]
        loss = F.cross_entropy(logits_masked, y_masked)
        acc = metric(logits_masked, y_masked)
        return loss, acc, logits_masked, y_masked

    def training_step(self, batch, batch_idx):
        self.config.batch_size = self.config.bs_train
        batch = make_batch(self.config)
        x, id, y, mask = batch['input'], batch['pos_id'], batch['target'], batch['mask']
        x, id, y, mask = x.to(self.device), id.to(self.device), y.to(self.device), mask.to(self.device)
        logits = self.model(x, id)
        loss, acc, _, _ = self._calculate_loss_acc(logits, y, mask, self.train_acc)
            
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.config.batch_size = self.config.bs_val
        batch = make_batch(self.config)
        x, id, y, mask = batch['input'], batch['pos_id'], batch['target'], batch['mask']
        x, id, y, mask = x.to(self.device), id.to(self.device), y.to(self.device), mask.to(self.device)
        logits = self.model(x, id)
        loss, acc, logits_masked, y_masked = self._calculate_loss_acc(logits, y, mask, self.val_acc)
        confusion_matrix = self.get_confusion_matrix(logits_masked, y_masked)
        outputs = {
            'confusion_matrix': confusion_matrix,
        }

        # for checkpointing
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        # for logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val/acc',   acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer
