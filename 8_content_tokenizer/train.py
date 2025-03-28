from accelerate import Accelerator
from model import TokenConvertModel, count_parameters
from dataset import MyDataset, train_val_split, collate_fn, collate_fn_multispk
from torch.optim import AdamW
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import subsample_features
import torch
import torch.nn.functional as F
import json
import os
import re

class Trainer:
    def __init__(self, filelist: str, config: OmegaConf):
        self.model = TokenConvertModel()
        self.config = config
        self.optimizer = AdamW(self.model.parameters(), lr=config.train.learning_rate)
        self.dataset = MyDataset(filelist, config=config)

        self.train_dataset, self.val_dataset = train_val_split(self.dataset, 
            val_split=config.train.val_split, random_seed=config.train.random_seed)

        self.is_multispk = (config.model.n_speakers > 1 or 
            config.train.override_sid is not None)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=collate_fn_multispk if self.is_multispk else collate_fn)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=config.train.batch_size, shuffle=False,
            collate_fn=collate_fn_multispk if self.is_multispk else collate_fn)

        accelerator = Accelerator(
            mixed_precision="fp16",
        )
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        )
        self.device = accelerator.device
        self.accelerator = accelerator 
        self.exp_name = config.exp_name
        self.ckpt_dir = f"checkpoints/{self.exp_name}"

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def step(self, batch, train=True):
        if train:
            self.optimizer.zero_grad()

        if self.is_multispk:
            wav_path, embed, feat, embed_mask, feat_mask, spk_ids = batch
        else:
            _, embed, feat, embed_mask, feat_mask = batch

        with self.accelerator.autocast():
            subsampled_feat = subsample_features(feat, config.train.feat_summary_subsample)
            summary = self.model.summarize(subsampled_feat)
            if self.config.train.override_sid is not None:
                spk_ids = torch.ones(feat.size(0), dtype=torch.long).to(self.device) * self.config.train.override_sid
            feat_pred = self.model(embed, embed_mask, 
                sid=torch.tensor(spk_ids).to(self.device) if self.is_multispk else None,
                summary=summary)
            loss = F.mse_loss(feat_pred, feat)
            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()

        return loss

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        if self.config.train.freeze_bottom_layers is not None:
            for i in range(self.config.train.freeze_bottom_layers):
                for param in self.model.conformers[i].parameters():
                    param.requires_grad = False
    
        for batch in tqdm(self.train_dataloader, desc=f"train Epoch {self.epoch}"):
            loss = self.step(batch, train=True)
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_dataloader)
    
    def val_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        for batch in tqdm(self.val_dataloader, desc=f"val Epoch {self.epoch}"):
            loss = self.step(batch, train=False)
            epoch_loss += loss.item()
        return epoch_loss / len(self.val_dataloader)

    def save_name(self, epoch):
        return f"{self.exp_name}_{epoch}.pth"

    def save(self, path):
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "spk_mapping": self.dataset.spk_id_mapping if self.is_multispk else {}}, path)

    def load(self, path):    
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]
        return state

    def train(self, autoload=False, pretrained_ckpt=None):
        if pretrained_ckpt is not None:
            state = torch.load(pretrained_ckpt, weights_only=True)
            self.model.load_state_dict(state["model"], strict=False)
            self.epoch = 0
            print(f"Loaded pretrained model {pretrained_ckpt}")
        elif autoload:
            files = os.listdir(self.ckpt_dir)
            regex = f"{self.exp_name}_([0-9]+).pth"
            matches = [x for x in files if re.match(regex, x)]
            if matches:
                latest = max(matches, key=lambda x: int(x.split("_")[
                    1].removesuffix(".pth")))
                self.load(os.path.join(self.ckpt_dir, latest))
                print(f"Loaded {latest}, epoch {self.epoch}")

        if not hasattr(self, "epoch"):
            self.epoch = 0

        print(f"Training {count_parameters(self.model)} for {self.config.train.num_epochs} epochs")
        for epoch in range(self.epoch, self.config.train.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            print(f"Epoch {epoch} train loss: {train_loss:.4f} val loss: {val_loss:.4f}")
            if epoch % 5 == 0:
                self.save(os.path.join(self.ckpt_dir, self.save_name(epoch)))
        self.save(os.path.join(self.ckpt_dir, self.save_name(epoch)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--config", type=str, default='config.yaml')
    parser.add_argument("--autoload", action="store_true", default=False)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer = Trainer(args.filelist, config)
    trainer.train(autoload=args.autoload, pretrained_ckpt=args.pretrained_ckpt)