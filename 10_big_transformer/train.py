import os
import re
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm
from model import MyModel
from dataset import MyDataset, train_val_split, Collator
from commons import count_parameters
import torch

from torch.optim import AdamW

class Trainer:
    def __init__(self, filelist: str, config: OmegaConf):
        self.model = MyModel(config)
        self.optimizer = AdamW(self.model.parameters(), lr=config.train.learning_rate)

        self.dataset = MyDataset(config=config, filelist_path=filelist)
        self.train_dataset, self.val_dataset = train_val_split(self.dataset, 
            val_split=config.train.val_split, random_seed=config.train.random_seed)

        self.collator = Collator(config)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=self.collator.collate)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=config.train.batch_size, shuffle=False,
            collate_fn=self.collator.collate)

        accelerator = Accelerator(
            mixed_precision="fp16", gradient_accumulation_steps=config.train.gradient_accumulation_steps
        )
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        )
        self.device = accelerator.device
        self.accelerator = accelerator 
        self.exp_name = config.train.exp_name

        self.ckpt_dir = f"checkpoints/{self.exp_name}"
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.epoch = 0
        self.config = config

        self.batch_loss = None

    def step(self, batch, train=True):
        if train:
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()

        with self.accelerator.autocast():
            decoded = self.model(**batch)
            loss = decoded.loss

            if train:
                self.accelerator.backward(loss)
                self.optimizer.step()
        return loss

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc=f"train Epoch {self.epoch}")
        for batch in pbar:
            loss = self.step(batch, train=True)
            epoch_loss += loss.item()
            self.batch_loss = loss
            pbar.set_postfix({"batch loss": self.batch_loss.item()})
        return epoch_loss / len(self.train_dataloader)

    def val_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        pbar = tqdm(self.val_dataloader, desc=f"val Epoch {self.epoch}")
        for batch in pbar:
            loss = self.step(batch, train=False)
            epoch_loss += loss.item()
            self.batch_loss = loss
            pbar.set_postfix({"batch loss": self.batch_loss.item()})
        return epoch_loss / len(self.val_dataloader)

    def save_name(self):
        return f"epoch_{self.epoch}.pt"

    def save(self, save_path):
        torch.save({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_yaml(self.config),
        }, save_path)

        self.keep_n_checkpoints(self.config.train.keep_n_checkpoints)

        print(f"Saved {save_path}")

    def keep_n_checkpoints(self, n):
        files = os.listdir(self.ckpt_dir)
        regex = f"epoch_([0-9]+).pt"
        matches = [x for x in files if re.match(regex, x) and x.endswith(".pt")]
        if len(matches) > n:
            oldest = min(matches, key=lambda x: int(x.split("_")[-1].removesuffix(".pt")))
            os.remove(os.path.join(self.ckpt_dir, oldest))
            print(f"Removed {oldest}")

    def load(self, path):
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state["model"], strict=False)
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
            regex = f"epoch_([0-9]+).pt"
            matches = [x for x in files if re.match(regex, x) and x.endswith(".pt")]
            if matches:
                latest = max(matches, key=lambda x: int(x.split("_")[
                    -1].removesuffix(".pt")))
                self.load(os.path.join(self.ckpt_dir, latest))
                print(f"Loaded {latest}, epoch {self.epoch}")
            else:
                print(f"Autoload was set but no checkpoints found in {self.ckpt_dir}")

        if not hasattr(self, "epoch"):
            self.epoch = 0

        print(f"Training {count_parameters(self.model)} for {self.config.train.num_epochs} epochs")
        for epoch in range(self.epoch, self.config.train.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            print(f"Epoch {epoch} train loss: {train_loss:.4f} val loss: {val_loss:.4f}")
            if epoch % self.config.train.epoch_save_interval == 0:
                self.save(os.path.join(self.ckpt_dir, self.save_name()))
        self.save(os.path.join(self.ckpt_dir, self.save_name()))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--config", type=str, default='configs/common.yaml')
    parser.add_argument("--autoload", action="store_true", default=False)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer = Trainer(args.filelist, config)
    try:
        trainer.train(autoload=args.autoload, pretrained_ckpt=args.pretrained_ckpt)
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint")
        trainer.save(os.path.join(trainer.ckpt_dir, trainer.save_name()))