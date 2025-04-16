import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import get_dataloader
from lightning_model import EnCodecLightningModel

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    
    # Get dataloader
    train_loader, dataset = get_dataloader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        persistent_workers=True, pin_memory=True
    )
    
    # Create validation dataloader with a subset of the training data
    val_loader, _ = get_dataloader(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        persistent_workers=True, pin_memory=True
    )
    
    # Create model
    model = EnCodecLightningModel(config)
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name="encodec_singing"
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["logging"]["log_dir"], "checkpoints"),
        filename="encodec-{epoch:02d}-{val/total_loss:.4f}",
        monitor="val/total_loss",
        mode="min",
        save_top_k=2,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        check_val_every_n_epoch=config["logging"]["log_audio_every_n_epochs"],
        precision=config["training"]["precision"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EnCodec model for singing voice synthesis")
    parser.add_argument("--config", type=str, default="config/model.yaml", help="Path to config file")
    
    args = parser.parse_args()
    main(args)