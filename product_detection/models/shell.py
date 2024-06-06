import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from typing import Literal
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix, MulticlassF1Score
import matplotlib.pyplot as plt
import itertools
import einops

from config import Params
from models.model import SpecificBERT

class Model_Lightning_Shell(L.LightningModule):
    def __init__(
            self,
            args: Params
        ) -> None:
        super().__init__()

        # Match model that we need
        self.inner_model = SpecificBERT(
            vocab_size=args.dataset.vocab_size, 
            embed_dim=args.model.embedding_dim, 
            pad_value=args.dataset.pad_value, 
            chunk_lenght=args.dataset.chunk_lenght,
            num_classes=args.model.num_output_classes, 
            depth=args.model.layers, 
            num_heads=args.model.heads, 
            mlp_dim=args.model.mlp_dim,
            norm_type=args.model.norm_type, 
            qkv_bias=args.model.qkv_bias, 
            drop_rate=args.model.dropout
        )

        self.metric_accuracy = MulticlassAccuracy(num_classes=args.model.num_output_classes)
        self.metric_f1 = MulticlassF1Score(num_classes=args.model.num_output_classes)
        self.matrix = ConfusionMatrix(task = "multiclass", num_classes = args.model.num_output_classes)
        self.flag_conf_matrix = True

        #-----
        self.args = args
        self.save_hyperparameters()
    
    def forward(self, x) -> torch.Any:
        return self.inner_model(x)
    
    def log_everything(self, pred_loss, out, y, name:str):
        self.log(f"{name}_loss", pred_loss)
        self.log(f"{name}_acc", self.metric_accuracy(out, y))
        self.log(f"{name}_f1", self.metric_f1(out, y))
    
    def loss(self, y, y_hat):
        return F.cross_entropy(y, y_hat)

    def lr_scheduler(self, optimizer):
        if self.args.scheduler.name == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = self.args.scheduler.patience, 
                factor = self.args.scheduler.factor
            )
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        
        if self.args.scheduler.name == "OneCycleLR":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr = self.args.training.lr * self.args.scheduler.expand_lr, 
                total_steps = self.args.training.epochs
            )
            scheduler_out = {"scheduler": sched}
        
        return scheduler_out
    
    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch
        x = torch.stack(x)
        x = einops.rearrange(x, "len batch -> batch len")

        out = self(x)[:,-1,:]
        pred_loss = self.loss(out, y)

        self.log_everything(pred_loss, out, y, name="train")
        
        return pred_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch
        x = torch.stack(x)
        x = einops.rearrange(x, "len batch -> batch len")

        out = self(x)[:,-1,:]
        pred_loss = self.loss(out, y)

        self.log_everything(pred_loss, out, y, name="val")

        if self.flag_conf_matrix:
            self.conf_matrix = self.matrix(torch.softmax(out, dim=-1), y)
            self.flag_conf_matrix = False
        else:
            self.conf_matrix += self.matrix(torch.softmax(out, dim=-1), y)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        pass
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.training.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )


class ConfMatrixLogging(L.Callback):
    def __init__(self, cls) -> None:
        super().__init__()
        self.cls = cls
    
    def make_img_matrix(self, matr):
        matr = matr.cpu()
        fig=plt.figure(figsize=(16, 8), dpi=80)
        plt.imshow(matr,  interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        tick_marks = np.arange(len(self.cls))
        plt.xticks(tick_marks, self.cls, rotation=90)
        plt.yticks(tick_marks, self.cls)

        fmt = 'd'
        thresh = matr.max() / 2.
        for i, j in itertools.product(range(matr.shape[0]), range(matr.shape[1])):
            plt.text(j, i, format(matr[i, j], fmt), horizontalalignment="center", color="white" if matr[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        return [fig]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        trainer.logger.log_image(key="Validation Confusion Matrix", images=self.make_img_matrix(pl_module.conf_matrix))
        plt.close()
        pl_module.flag_conf_matrix = True