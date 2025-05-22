# Stage 1. KL Div from teacher
from svc_helper.svc.rvc.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
import torch
from copy import deepcopy
import pytorch_lightning as pl
from omegaconf import OmegaConf
from dataset import FeatureDataset, FeatureCollator

class TrainingModule(pl.LightningModule):
    def __init__(self, teacher_enc, student_enc, config : OmegaConf):
        super().__init__()
        self.teacher_enc = teacher_enc
        self.student_enc = student_enc
        self.config = config

    def step(self, batch, batch_idx):
        x = batch
        rvc_feat = x['rvc_feat']
        whisp_feat = x['whisp_feat']
        pitch = x['pitch']
        pitch_fine = x['pitch_fine']
        lens = x['lengths']

        teacher_m, teacher_logs, _ = self.teacher_enc(
            rvc_feat, pitch, lens)
        student_m, student_logs, _ = self.student_enc(
            whisp_feat, pitch, lens)

        def kl_divergence(student_m, student_logs, teacher_m, teacher_logs):
            # Convert log-variance to variance
            student_var = torch.exp(student_logs)
            teacher_var = torch.exp(teacher_logs)

            kl = 0.5 * (
                teacher_logs - student_logs +
                (student_var + (student_m - teacher_m) ** 2) / teacher_var -
                1
            )
            
            # Sum over latent dimensions, then mean over batch
            return kl.sum(dim=1).mean()

        loss = kl_divergence(student_m, student_logs, teacher_m, teacher_logs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.student_enc.parameters(), lr=self.config.train.lr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--from_rvc", type=str, default='tests/RarityTitan.pth')
    parser.add_argument("--init_weight", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    state = torch.load(args.from_rvc)
    model = SynthesizerTrnMs768NSFsid(*state['config'], is_half=True)
    del model.enc_q
    model.load_state_dict(state['weight'])
    teacher_enc = model.enc_p
    student_enc = deepcopy(teacher_enc)

    training_module = TrainingModule(teacher_enc, student_enc, config)

    if args.init_weight is not None:
        ckpt = torch.load(args.init_weight, weights_only=False)
        if 'state_dict' in ckpt: # lightning checkpoint
            training_module.load_state_dict(ckpt['state_dict'])

    train_dataset = FeatureDataset(config, is_train=True)
    val_dataset = FeatureDataset(config, is_train=False)
    collator = FeatureCollator()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=collator,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=collator,
    )


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{config.exp_name}',
        filename='best-checkpoint',
        save_top_k=2,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epochs,
        accelerator='auto',
        precision='16-mixed',
        callbacks=[checkpoint_callback],
    )
    trainer.fit(training_module, train_dataloader, val_dataloader)

    # Plan:
    # - kl loss between student and teacher distributions

    # TODO dataset and setup