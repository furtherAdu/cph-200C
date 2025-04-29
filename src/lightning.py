import os
import sys
from collections import defaultdict
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.directory import log_dir

pl.seed_everything(40)


def get_trainer(model_name, checkpoint_callback, monitor='val_loss', mode='min', max_epochs=100, **kwargs):
    # set up logging
    logger = CSVLogger(save_dir=log_dir, name=model_name)

    trainer_kwargs = dict(
        precision="bf16-mixed",
        accelerator='auto',
        logger=logger,
        callbacks=[
            EarlyStopping(monitor=monitor, mode=mode, patience=5),
            checkpoint_callback
        ],
        log_every_n_steps=1,
        max_epochs=max_epochs,
    )

    trainer_kwargs.update(kwargs)

    # set up trainer
    trainer = pl.Trainer(
        **trainer_kwargs
    )

    return trainer


def get_checkpoint_callback(model_name, dir_path, monitor='val_loss', mode='min'):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=dir_path,
        filename=model_name + '-{epoch:002d}-{val_loss:.2f}',
        save_top_k=1)

    return checkpoint_callback


def get_log_dir_path(model_name):
    dir_path = os.path.join(log_dir, model_name)
    if not os.path.isdir(dir_path):
        version = '0'
    else:
        version = str(int(sorted(os.listdir(dir_path))[-1].replace('version_', '')) + 1)
    dir_path = os.path.join(dir_path, f'version_{version}')

    return dir_path

def get_logger(model_name, project_name='CPH_200C', wandb_entity='furtheradu', dir_path='../',**kwargs):
    if kwargs.get('disable_wandb'):
        print("wandb logging is disabled.")
        return None
    else:
        wandb.finish()
        logger = pl.loggers.WandbLogger(
            project=project_name,
            entity=wandb_entity,
            group=model_name,
            dir=dir_path,
            **kwargs
        )
        return logger

class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        # if not sys.stdout.isatty():
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        # if not sys.stdout.isatty():
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=1e-4):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)
        self.outputs = defaultdict(list)

    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["X"], batch["Y"]
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx, stage='train'):
        return self.on_step(batch, stage)

    def validation_step(self, batch, batch_idx, stage='val'):
        return self.on_step(batch, stage)

    def test_step(self, batch, batch_idx, stage='test'):
        return self.on_step(batch, stage)

    def on_step(self, batch, stage):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        self.log(f'{stage}_loss', loss, sync_dist=True, prog_bar=True)
        self.log(f'{stage}_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.outputs[stage].append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self, stage='train'):
        self.on_epoch_end(stage)

    def on_validation_epoch_end(self, stage='val'):
        self.on_epoch_end(stage)

    def on_test_epoch_end(self, stage='test'):
        self.on_epoch_end(stage)
    
    def on_epoch_end(self, stage):
        y_hat = torch.cat([o["y_hat"] for o in self.outputs[stage]])
        y = torch.cat([o["y"] for o in self.outputs[stage]])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log(f"{stage}_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.outputs[stage] = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return optimizer
    
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        

class MLPClassifierLightning(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=2, init_lr=1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                         nn.ReLU())

        self.hidden_layers = []
        for _ in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    nn.ReLU())
                                         )

        self.final_layer = nn.Sequential(nn.Linear(self.hidden_dim, num_classes),    
                                         nn.Softmax(dim=-1)
                                         )

        self.model = nn.Sequential(self.first_layer,
                                   *self.hidden_layers,
                                   self.final_layer
                                   )
        
        self.model.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)
