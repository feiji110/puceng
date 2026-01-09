#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   PUCENG_module.py
@Time    :   2023/12/14 12:16:40
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''
from typing import Any, Dict, Tuple

import torch
import torchmetrics
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanSquaredError

class CustomAccuracy(torchmetrics.Metric):#定义正确率的计算方式： tolerance=0.05误差在5%以内的为正确，否则为错误
    def __init__(self, tolerance=0.05, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tolerance = tolerance 
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred, target):
        error = torch.abs(pred - target)
        # jicha = abs(error.max() - error.min())

        correct = torch.sum(error <= self.tolerance*abs(target))
        total = pred.numel()  # total number of elements
        self.correct += correct
        self.total += total
    
    def compute(self):
        return self.correct.float() / self.total


class PUCENGLitModule(LightningModule):
    def __init__( 
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None: 
        super().__init__()
        torch.set_float32_matmul_precision('high') 
        self.save_hyperparameters(logger=False,ignore=['net']) 
        self.net = net
        
        self.criterion = torch.nn.MSELoss()   # MSELoss、L1Loss

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_accuracy = CustomAccuracy()              # self.test_mse = MeanSquaredError()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mse_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return y_hat, y, loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        y_hat, y, loss = self.model_step(batch)
        self.train_loss(loss)
        self.train_mse(y_hat, y)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # loss, preds, targets = self.model_step(batch)
        y_hat, y, loss = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_mse(y_hat, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)   

    def on_validation_epoch_end(self) -> None:
        mse = self.val_mse.compute()
        self.val_mse_best(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        y_hat, y, loss = self.model_step(batch)
        self.test_loss(loss)
        self.test_accuracy(y_hat, y)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage =="fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
if __name__ == "__main__":
    _ = PUCENGLitModule(None, None, None, None)