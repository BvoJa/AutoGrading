import torch
import hydra
import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf.omegaconf import OmegaConf

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)



@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    # logger.info(f"Using the model: {cfg.model.name}")
    # logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")

    cola_data = DataModule(
        cfg.model.name, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", 
        monitor="val/loss_epoch", 
        mode="min",
    )

    wandb_logger = WandbLogger(project="MLOps_Basics")
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        max_epochs=cfg.training.max_epochs,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(cola_model, cola_data)

if __name__ == "__main__":
    main()