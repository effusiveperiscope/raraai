
import pytorch_lightning as pl
from models.model_flowvae import MelFlowVAE # Assuming model.py exists
from torch.optim.lr_scheduler import LambdaLR
from dataset import MelDataloader # Assuming dataset.py exists
import torch
from common import mel_to_img # Assuming common.py exists
import math
import argparse # Added for command-line arguments
from omegaconf import OmegaConf # Added for config loading
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Tensorboard will log many extraneous warnings

class KLBetaScheduler(pl.Callback):
    def __init__(
        self,
        initial_beta=0.0,
        max_beta=1.0,
        start_epoch=0,
        end_epoch=10,
        scheduler_type="linear"
    ):
        """
        Custom callback to schedule the KL beta term in a VAE.
        
        Args:
            initial_beta: Starting beta value
            max_beta: Maximum beta value
            start_epoch: Epoch to start increasing beta
            end_epoch: Epoch to reach max_beta
            scheduler_type: Type of scheduler ("linear", "cosine", "cyclic", etc.)
        """
        super().__init__()
        self.initial_beta = initial_beta
        self.max_beta = max_beta
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.scheduler_type = scheduler_type
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Get current epoch
        current_epoch = trainer.current_epoch
        
        if current_epoch < self.start_epoch:
            beta = self.initial_beta
        elif current_epoch >= self.end_epoch:
            beta = self.max_beta
        else:
            # Calculate progress (0 to 1)
            progress = (current_epoch - self.start_epoch + (batch_idx / len(trainer.train_dataloader))) / (self.end_epoch - self.start_epoch)
            
            # Apply scheduling based on selected type
            if self.scheduler_type == "linear":
                beta = self.initial_beta + progress * (self.max_beta - self.initial_beta)
            elif self.scheduler_type == "cosine":
                beta = self.initial_beta + 0.5 * (1 - math.cos(math.pi * progress)) * (self.max_beta - self.initial_beta)
            elif self.scheduler_type == "exponential":
                beta = self.initial_beta + (math.exp(3 * progress) - 1) / (math.exp(3) - 1) * (self.max_beta - self.initial_beta)
            elif self.scheduler_type == "cyclic":
                # Example cyclic scheduler
                cycle_progress = (progress * 4) % 1.0  # 4 cycles during the annealing period
                beta = self.initial_beta + 0.5 * (1 - math.cos(2 * math.pi * cycle_progress)) * (self.max_beta - self.initial_beta)
            else:
                # Default to linear
                beta = self.initial_beta + progress * (self.max_beta - self.initial_beta)
        
        # Update the beta value in the model
        pl_module.kl_beta = beta
        
        # Optional: log the beta value
        pl_module.log("kl_beta", beta, prog_bar=True, logger=True)

def cosine_schedule_with_warmup(current_step, *, warmup_steps, total_steps, base_lr=1.0, min_lr=0.0):
    if current_step < warmup_steps:
        # Ensure warmup_steps is not zero to avoid division by zero
        if warmup_steps == 0:
             return base_lr # Or perhaps min_lr, depending on desired behavior at step 0
        return (current_step / warmup_steps) * base_lr
    else:
        # Ensure total_steps is greater than warmup_steps to avoid division by zero
        denominator = total_steps - warmup_steps
        if denominator <= 0:
            return min_lr # Or base_lr, depends on desired behavior if total_steps <= warmup_steps
            
        progress = (current_step - warmup_steps) / denominator
        # Clamp progress to avoid issues with steps slightly exceeding total_steps due to float precision or other factors
        progress = max(0.0, min(1.0, progress)) 
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

class VAETrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MelFlowVAE(config)
        self.config = config
        self.kl_beta = 0.0
        # Use save_hyperparameters() to automatically save config and allow reloading
        # Make sure your config object is pickleable (OmegaConf usually is)
        self.save_hyperparameters(config) 

    def training_step(self, batch, batch_idx):
        if self.config.model.decoder.pitch_conditioning:
            x, x_mask, pitch = batch
        else:
            x, x_mask = batch
            pitch = None

        disable_flow = self.current_epoch < self.config.train.flow_start
        x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask = self.model(x, x_mask,
            disable_flow=disable_flow)
        loss, recon_loss, kl_loss = self.model.loss(
            x, x_mask, x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask)
        
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True) 
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_kl_loss", kl_loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_log_det", log_det.mean().item(), on_step=True, on_epoch=False, logger=True)
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch < self.config.train.train_only_flows_end:
            self.model.encoder.requires_grad_(False)
            self.model.decoder.requires_grad_(False)
        else:
            self.model.encoder.requires_grad_(True)
            self.model.decoder.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        if self.config.model.decoder.pitch_conditioning:
            x, x_mask, pitch = batch
        else:
            x, x_mask = batch
            pitch = None

        x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask = self.model(x, x_mask)
        loss, recon_loss, kl_loss = self.model.loss(
            x, x_mask, x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, logger=True)

        self.log_validation_batch_images(x, x_recon, x_mask, batch_idx, include_individual=False)

        return loss

    def log_validation_batch_images(self, x, x_recon, x_mask, batch_idx, tag_prefix="val", 
                         nrow=4, include_individual=True):
        """
        Log all images in a batch during validation.
        
        Args:
            x: Original input batch tensor [batch_size, seq_len, features]
            x_recon: Reconstructed batch tensor [batch_size, seq_len, features]
            x_mask: Mask tensor indicating valid sequence lengths [batch_size, seq_len]
            batch_idx: Current batch index
            tag_prefix: Prefix for tensorboard tags (default: "val")
            nrow: Number of images per row in grid (default: 4)
            include_individual: Whether to also log first sample individually (default: True)
        
        Returns:
            None
        """
        # Only log first batch and if logger exists
        if batch_idx != 0 or self.logger is None or not hasattr(self.logger.experiment, 'add_image'):
            return
        
        try:
            # Import here to avoid potential import issues
            from torchvision.utils import make_grid
            
            # Get batch size
            batch_size = x.size(0)
            
            # Create lists for different image types
            orig_images = []
            recon_images = []
            delta_images = []
            
            # Process each sample in the batch
            for i in range(batch_size):
                # Get actual length for this sample
                sample_length = x_mask[i].sum().item()
                
                # Get original and reconstructed sample (trimmed to actual length)
                x_i = x[i][:sample_length]
                x_recon_i = x_recon[i][:sample_length]
                
                # Compute reconstruction error
                delta_x_i = x_i - x_recon_i
                
                # Convert to images (ensure on CPU)
                orig_x_img = mel_to_img(x_i.cpu())
                recon_x_img = mel_to_img(x_recon_i.cpu())
                delta_x_img = mel_to_img(delta_x_i.cpu())
                
                # Add to respective lists
                orig_images.append(orig_x_img)
                recon_images.append(recon_x_img)
                delta_images.append(delta_x_img)
            
            # Create image grids
            grid_nrow = min(nrow, batch_size)
            orig_grid = make_grid(torch.stack(orig_images), nrow=grid_nrow, padding=2)
            recon_grid = make_grid(torch.stack(recon_images), nrow=grid_nrow, padding=2)
            delta_grid = make_grid(torch.stack(delta_images), nrow=grid_nrow, padding=2)
            
            # Log grid images
            self.logger.experiment.add_image(f"{tag_prefix}/original_x_batch", orig_grid, global_step=self.global_step)
            self.logger.experiment.add_image(f"{tag_prefix}/reconstructed_x_batch", recon_grid, global_step=self.global_step)
            self.logger.experiment.add_image(f"{tag_prefix}/delta_x_batch", delta_grid, global_step=self.global_step)
            
            # Optionally log first sample individually
            if include_individual and batch_size > 0:
                self.logger.experiment.add_image(f"{tag_prefix}/original_x", orig_images[0], global_step=self.global_step)
                self.logger.experiment.add_image(f"{tag_prefix}/reconstructed_x", recon_images[0], global_step=self.global_step)
                self.logger.experiment.add_image(f"{tag_prefix}/delta_x", delta_images[0], global_step=self.global_step)
                
        except Exception as e:
            print(f"Warning: Failed to log {tag_prefix} images: {e}")
            import traceback
            traceback.print_exc()

# Example usage in your validation step:
# log_validation_batch_images(self, x, x_recon, x_mask, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.train.lr, weight_decay=self.config.train.get('weight_decay', 0.01))

        # Calculate total_steps dynamically if not provided or if it's -1
        total_steps = self.config.train.get('total_steps', -1)
        if total_steps == -1:
            # Estimate total steps if not specified in config
            # This requires the trainer object, which isn't available here directly.
            # A common workaround is to estimate based on dataloader length and max_epochs.
            # This estimation might be slightly off due to gradient accumulation, etc.
            # If you have access to the trainer here (e.g., via self.trainer after setup), use that.
            # For simplicity, we'll rely on the config value for now.
            # If you frequently need dynamic calculation, consider using LightningCLI
            # or calculating it before initializing the LightningModule.
            if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'estimated_stepping_batches'):
                 total_steps = self.trainer.estimated_stepping_batches
            else:
                 # Fallback estimation (might need adjustment)
                 len_dataloader = len(self.train_dataloader()) # Requires dataloader access
                 max_epochs = self.config.train.get('max_epochs', 5) # Or read from trainer args
                 total_steps = len_dataloader * max_epochs
                 print(f"Warning: Estimating total_steps={total_steps}. Provide config.train.total_steps for accuracy.")
                 # Update config value if you want it saved in checkpoint hparams
                 # self.config.train.total_steps = total_steps 
        else:
            total_steps = self.config.train.total_steps


        print(f"Configuring LR Scheduler: warmup_steps={self.config.train.warmup_steps}, total_steps={total_steps}, base_lr={self.config.train.lr}, min_lr={self.config.train.min_lr}")

        scheduler_config = {
            "scheduler": LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_schedule_with_warmup(
                    step,
                    warmup_steps=self.config.train.warmup_steps,
                    total_steps=total_steps, # Use the calculated/retrieved total_steps
                    base_lr=1.0, # Lambda applies multiplier to the base LR in optimizer
                    min_lr=self.config.train.min_lr / self.config.train.lr, # min_lr ratio relative to base_lr
                ),
            ),
            "interval": "step", # Call scheduler every step
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--init_weights", type=str, default=None, help="Init weights (non-strict loading)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint file")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max_epochs from config")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (cpu, gpu, tpu, auto)")
    parser.add_argument("--devices", default="auto", help="Devices to use (e.g., 1, [0, 1], auto)")


    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # --- Optionally override config values from args ---
    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs # Assuming max_epochs is under train section

    train_dataloader = MelDataloader(config)
    val_dataloader = MelDataloader(config, is_train=False)

    # --- Instantiate model ---
    # If resuming, Lightning will load state dict later.
    # If the checkpoint contains hyperparameters (saved via self.save_hyperparameters),
    # Lightning can potentially load the model using those, but it's often safer
    # to initialize with the current config and let load_from_checkpoint handle state.
    if args.pretrained is not None:
        if args.ckpt_path is not None:
            raise ValueError("Cannot specify both --ckpt_path and --pretrained")
        model = VAETrainer.load_from_checkpoint(args.pretrained)
        print(f"Loaded pretrained model from {args.pretrained}")
    else:
        model = VAETrainer(config)
        print(f"Loaded model from {args.config}")

    if args.init_weights is not None:
        ckpt = torch.load(args.init_weights, weights_only=False)
        if 'state_dict' in ckpt: # lightning checkpoint
            ckpt = ckpt['state_dict']
        model_state_dict = model.state_dict()
        filtered_dict = {
                k: v for k, v in ckpt.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
        result = model.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded initial weights from {args.init_weights}")

    logger = pl.loggers.TensorBoardLogger(config.train.get("log_dir", "logs"), name=config.train.get("exp_name", "mel_vae"))

    # Define checkpoint filename dynamically using config values
    checkpoint_filename = f"{config.train.exp_name}_mel_vae-{{epoch:02d}}-{{val_loss:.2f}}"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.train.get("checkpoint_dir", "checkpoints"),
        filename=checkpoint_filename,
        save_top_k=config.train.get("save_top_k", 3),
        mode="min",
        save_last=True # Often useful to save the latest state
    )
    
    # --- Determine devices ---
    devices_arg = args.devices
    if devices_arg != "auto":
       try:
           # Try parsing as integer list e.g., "0,1" -> [0, 1]
           devices_arg = [int(d.strip()) for d in devices_arg.split(',')]
           if len(devices_arg) == 1:
                devices_arg = devices_arg[0] # Keep as int if single device
       except ValueError:
           # Keep as string if parsing fails (e.g., "auto" or specific device names)
           pass 
    
    kl_scheduler = KLBetaScheduler(
        initial_beta=0.0,
        max_beta=config.train.kl_max,
        start_epoch=config.train.kl_start,
        end_epoch=config.train.kl_end,
    )
           
    trainer = pl.Trainer(
        precision=config.train.precision,
        max_epochs=config.train.max_epochs, # Use value potentially overridden by args
        logger=logger,
        callbacks=[checkpoint_callback, kl_scheduler],
        accelerator=args.accelerator,
        devices=devices_arg,
    )

    # --- Start Training (or resume) ---
    # Pass the model and dataloaders.
    # If resuming, ckpt_path inside trainer.fit handles loading.
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)