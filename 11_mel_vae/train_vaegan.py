
import traceback
import pytorch_lightning as pl
from models.model_flowvaegan import MelFlowVAEGAN # Assuming model.py exists
from torch.optim.lr_scheduler import LambdaLR
from dataset_f0 import MelF0Dataset, MelF0Collator
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from common import mel_to_img
import common
import math
import argparse # Added for command-line arguments
from omegaconf import OmegaConf # Added for config loading
from nsfhifigan.wav2mel import PitchAdjustableMelSpectrogram
import logging
import os

log = logging.getLogger(__name__)

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
        self.model = MelFlowVAEGAN(config)
        self.config = config
        self.kl_beta = 0.0
        self.steps_for_interval = 0
        # Use save_hyperparameters() to automatically save config and allow reloading
        # Make sure your config object is pickleable (OmegaConf usually is)
        self.save_hyperparameters(config) 
        self.automatic_optimization = False # Disable automatic optimization for GAN training

    def __setattr__(self, name, value):
        """Override attribute setting to monitor 'device' changes."""
        # Check if the attribute being set is 'device'
        if name == 'device':
            # Get the old value if it exists, otherwise note it's the first time
            old_value = getattr(self, name, 'AttributeNotSetYet')

            # Check if the value is actually changing or if it's the first set
            if value != old_value or old_value == 'AttributeNotSetYet':
                log.warning(f"!!! Setting '{name}': FROM '{old_value}' TO '{value}' !!!")
                # Print the traceback to see where the assignment is coming from
                log.warning("--- Traceback for device assignment ---")
                for line in traceback.format_stack():
                     # Optional: filter out lines from this __setattr__ itself for clarity
                     if '__setattr__' not in line and 'logging' not in line:
                         log.warning(line.strip())
                log.warning("------------------------------------")

        # Important: Call the original __setattr__ to actually perform the assignment
        # Use super() to avoid infinite recursion
        super().__setattr__(name, value)

    def setup(self, stage=None):
        # Log device information at setup time
        print(f"Current device in setup: {self.device}")
        # This method runs on every GPU in distributed training
        
    def on_fit_start(self):
        # Another place to check device
        print(f"Device at fit start: {self.device}")
        print(f"Is model on device: {next(self.model.parameters()).device}")
        self.reset_nan_batch_norms()

    def reset_nan_batch_norms(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                if torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any():             
                    print(f"Resetting NAN batch norm {name}")
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()
        
    def training_step(self, batch, batch_idx):
        assert self.device.type == "cuda"
        assert self.config.model.decoder.pitch_conditioning
        (x, pitch, x_mask) = batch
        assert x_mask.dtype == torch.bool

        x = common.normalize(self.config, x) # Input to model is normalized log mel spectrogram

        disable_flow = self.current_epoch < self.config.train.flow_start

        disc_optimizer, encoder_optimizer, decoder_optimizer = self.optimizers()
        disc_scheduler, encoder_scheduler, decoder_scheduler = self.lr_schedulers()

        loss = None

        # --- Shared Forward Pass for Discriminator and initial Encoder outputs ---
        # This x_recon is based on the current encoder and decoder parameters
        # before any updates in this step.
        # Outputs from this pass will be used for D update and E update.
        x_recon_shared, z_mean_shared, z_log_var_shared, z_shared, \
        z_transformed_shared, log_det_shared, z_mask_shared = self.model(
            x, x_mask, disable_flow=disable_flow, pitch=pitch
        )

        # --- Discriminator Update ---
        if (self.global_step % self.config.train.disc_every == 0
            and self.current_epoch > self.config.train.vae_only_end):
            disc_optimizer.zero_grad()
            # Discriminator sees real data and detached fake data from the shared forward pass
            real_loss, fake_loss, _, _ = self.model.disc_aug_losses(
                x, x_recon_shared.detach(), x_mask # Use .detach() for D training
            )
            disc_loss = real_loss + fake_loss
            loss_disc_scaled = self.config.train.lam_disc * disc_loss
            
            # Retain graph: True, because x_recon_shared and z_shared (and their graph
            # leading back to the encoder) will be used in the encoder's loss calculation next.
            # The graph components from self.model() need to persist.
            self.manual_backward(loss_disc_scaled, retain_graph=True) 
            disc_optimizer.step()
            disc_scheduler.step() # Schedulers usually step after optimizers

            self.log("train_disc_loss", disc_loss.mean().item(), on_step=True, on_epoch=True, logger=True)
            self.log("train_real_loss", real_loss.mean().item(), on_step=True, on_epoch=False, logger=True)
            self.log("train_fake_loss", fake_loss.mean().item(), on_step=True, on_epoch=False, logger=True)

        if (self.global_step % self.config.train.gen_every == 0
            and self.current_epoch > self.config.train.vae_only_end):
            # --- Encoder Update ---
            encoder_optimizer.zero_grad()
            # KL divergence uses outputs from the shared forward pass
            kl_div = self.model.kl_loss(
                z_shared, z_mean_shared, z_log_var_shared, z_mask_shared, 
                z_transformed_shared, log_det_shared
            ) 
            
            # Feature Matching loss for encoder:
            # Uses x_recon_shared (gradients flow through it to the encoder)
            # and the *UPDATED* discriminator (from disc_optimizer.step() above).
            _, _, fm_loss_enc, _ = self.model.disc_aug_losses(
                x, x_recon_shared, x_mask # x_recon_shared is NOT detached here
            ) 
            
            # Reconstruction loss for encoder uses x_recon_shared
            recon_loss_enc = self.model.recon_loss(x, x_recon_shared, x_mask)
            
            loss_enc = kl_div * self.kl_beta + \
                    self.config.train.lam_fm * fm_loss_enc + \
                    self.config.train.lam_recon * recon_loss_enc
                    
            # Backward pass for the encoder.
            # The graph retained by loss_disc_scaled.backward(retain_graph=True) ensured
            # that the paths for kl_div (via z_shared) and recon_loss_enc (via x_recon_shared)
            # back to the encoder are still valid.
            # The fm_loss_enc path involves the updated discriminator, which is fine.
            self.manual_backward(loss_enc) 
            encoder_optimizer.step()
            encoder_scheduler.step()

            self.log("encoder_loss", loss_enc.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True) # Added .mean().item() if loss_enc is not scalar
            self.log("train_kl_loss", kl_div.mean().item(), on_step=True, on_epoch=False, logger=True)
            self.log("train_log_det_shared", log_det_shared.mean().item(), on_step=True, on_epoch=True, logger=True)


            # --- Decoder Update ---
            decoder_optimizer.zero_grad()
            # Encoder has been updated. We need a new x_recon based on the updated encoder's z.
            # This is Forward Pass 2 for the whole model.
            x_recon_for_dec, z_mean_dec, z_log_var_dec, z_dec, \
            z_transformed_dec, log_det_dec, z_mask_dec = self.model(
                x, x_mask, disable_flow=disable_flow, pitch=pitch
            )

            # Decoder losses are calculated using x_recon_for_dec (from updated encoder)
            # and the *UPDATED* discriminator (from disc_optimizer.step() far above).
            _, _, fm_loss_dec, gen_loss_dec = self.model.disc_aug_losses(
                x, x_recon_for_dec, x_mask
            )
            recon_loss_dec = self.model.recon_loss(x, x_recon_for_dec, x_mask)
            
            loss_dec = (self.config.train.lam_gen * gen_loss_dec +
                        self.config.train.lam_fm * fm_loss_dec + 
                        self.config.train.lam_recon * recon_loss_dec)
                    
            self.manual_backward(loss_dec)
            decoder_optimizer.step()
            decoder_scheduler.step()

            self.log("decoder_loss", loss_dec.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True) # Added .mean().item()
            self.log("train_fm_loss", fm_loss_dec.mean().item(), on_step=True, on_epoch=False, logger=True) # This was fm_loss_dec
            self.log("train_gen_loss", gen_loss_dec.mean().item(), on_step=True, on_epoch=False, logger=True) # This was gen_loss_dec
            self.log("train_recon_loss", recon_loss_dec.mean().item(), on_step=True, on_epoch=False, logger=True) # This was recon_loss_dec
            self.log("train_log_det_dec", log_det_dec.mean().item(), on_step=True, on_epoch=True, logger=True)

            # Log decoder gradient norm
            decoder_params = self.model.decoder.parameters()
            total_norm = 0
            for p in decoder_params:
                if p.grad is None:
                    continue
                p_norm = p.grad.detach().data.norm(2)
                total_norm += p_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.log("decoder_grad_norm", total_norm, on_step=True, on_epoch=False, logger=True)
        else: # VAE training only
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            x_recon, z_mean, z_log_var, z, \
                z_transformed, log_det, z_mask = self.model(
                x, x_mask, disable_flow=disable_flow, pitch=pitch
            )

            recon_loss = self.model.recon_loss(x, x_recon, x_mask)
            kl_loss = self.model.kl_loss(z, z_mean, z_log_var, z_mask, z_transformed, log_det)
            loss = self.config.train.lam_recon * recon_loss + self.kl_beta * kl_loss

            self.manual_backward(loss)
            decoder_optimizer.step()
            encoder_optimizer.step()
            decoder_scheduler.step()
            encoder_scheduler.step()
            disc_scheduler.step() # need to update lr anyway
            self.log("vae_loss", loss.mean().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)


        # Log learning rate
        lr = self.optimizers()[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True) 
        
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.device.type == "cuda"
        assert self.config.model.decoder.pitch_conditioning
        (x, pitch, x_mask) = batch

        x = common.normalize(self.config, x) # Input to model is normalized log mel spectrogram

        disable_flow = self.current_epoch < self.config.train.flow_start

        x_recon, z_mean, z_log_var, z, z_transformed, log_det, z_mask = self.model(x, x_mask,
            disable_flow=disable_flow, pitch=pitch)
        real_loss, fake_loss, fm_loss, gen_loss = self.model.disc_losses(x, x_recon.detach(), x_mask)
        recon_loss = self.model.recon_loss(x, x_recon, x_mask)
        kl_div = self.model.kl_loss(z, z_mean, z_log_var, z_mask, z_transformed, log_det) 
        loss = kl_div * self.kl_beta + (
            self.config.train.lam_fm * fm_loss + 
            self.config.train.lam_gen * gen_loss + 
            self.config.train.lam_recon * recon_loss)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_real_loss", real_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_fake_loss", fake_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_fm_loss", fm_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_kl_loss", kl_div, on_step=False, on_epoch=True, logger=True)
        self.log("val_gen_loss", gen_loss, on_step=False, on_epoch=True, logger=True)

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

            # First: Denormalize log mel specs
            x = common.denormalize(self.config, x)
            x_recon = common.denormalize(self.config, x_recon)
            
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
        encoder_optimizer = torch.optim.AdamW(self.model.encoder.parameters(),
             lr=self.config.train.lr, weight_decay=self.config.train.get('weight_decay', 0.01))
        decoder_optimizer = torch.optim.AdamW(
            list(self.model.decoder.parameters()) + list(self.model.flows.parameters()),
             lr=self.config.train.lr, weight_decay=self.config.train.get('weight_decay', 0.01))
        disc_optimizer = torch.optim.AdamW(self.model.discriminator.parameters(),
             lr=self.config.train.lr, weight_decay=self.config.train.get('weight_decay', 0.01))
        optimizers = [disc_optimizer, encoder_optimizer, decoder_optimizer]

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


        print(f"Configuring LR Schedulers: warmup_steps={self.config.train.warmup_steps}, total_steps={total_steps}, base_lr={self.config.train.lr}, min_lr={self.config.train.min_lr}")

        return optimizers, [
            LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_schedule_with_warmup(
                    step,
                    warmup_steps=self.config.train.warmup_steps,
                    total_steps=total_steps, # Use the calculated/retrieved total_steps
                    base_lr=1.0, # Lambda applies multiplier to the base LR in optimizer
                    min_lr=self.config.train.min_lr / self.config.train.lr, # min_lr ratio relative to base_lr
                ),
            ) for optimizer in optimizers
        ]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vaegan.yaml", help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--init_weights", type=str, default=None, help="Init weights (non-strict loading)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max_epochs from config")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (cpu, gpu, tpu, auto)")
    parser.add_argument("--devices", default="auto", help="Devices to use (e.g., 1, [0, 1], auto)")
    parser.add_argument("--version", type=int, default=None, help="For Tensorboard log directory")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # --- Optionally override config values from args ---
    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs # Assuming max_epochs is under train section

    train_dataset = MelF0Dataset(config.train.filelist, config, True)
    val_dataset = MelF0Dataset(config.val.filelist, config, False)
    collator = MelF0Collator(config, config.train.mel_pad_value, config.model.sampling_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, 
        collate_fn=collator, num_workers=config.train.num_workers, pin_memory=config.train.pin_memory,
        drop_last=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, 
        collate_fn=collator, num_workers=config.train.num_workers, pin_memory=config.train.pin_memory,
        drop_last=True, persistent_workers=True)

    # --- Instantiate model ---
    # If resuming, Lightning will load state dict later.
    # If the checkpoint contains hyperparameters (saved via self.save_hyperparameters),
    # Lightning can potentially load the model using those, but it's often safer
    # to initialize with the current config and let load_from_checkpoint handle state.
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
        warn_dict = {
                k: v for k, v in ckpt.items()
                if k not in model_state_dict or (k in model_state_dict) and (
                    v.shape != model_state_dict[k].shape)
        }
        print(f"Weights not loaded: {list(warn_dict.keys())}")
        result = model.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded initial weights from {args.init_weights}")

    logger = pl.loggers.TensorBoardLogger(
        config.train.get("log_dir", "logs"), name=config.train.get("exp_name", "mel_vae"),
        version=args.version
    )

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
        #devices=[0],
        #num_sanity_val_steps=0, # temporarily disable sanity checks
    )

    # --- Start Training (or resume) ---
    # Pass the model and dataloaders.
    # If resuming, ckpt_path inside trainer.fit handles loading.
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)