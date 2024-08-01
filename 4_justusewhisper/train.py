from torch.utils.data import DataLoader
import lightning as L
from model import LitFeatureConvertor
from dataset import SpeechFeatureDataset, collate_fn

if __name__ == '__main__':
    train_dataset = SpeechFeatureDataset()
    val_dataset = SpeechFeatureDataset(split='test')
    train_loader = DataLoader(train_dataset,
        num_workers=4, batch_size=4, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset,
        num_workers=2, batch_size=4, collate_fn=collate_fn, shuffle=False)

    lmodel = LitFeatureConvertor(
        config={
            'train': {
                'lr': 3e-4
            },
            'model': {
                'in_emb_size': 1280,
                'out_emb_size': 768,
                'hidden_dim': 384,
                'dropout': 0.0,
                'n_layers': 6
            }
        }
    )
    trainer = L.Trainer(max_steps=20000)
    trainer.fit(model=lmodel, train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path='last')