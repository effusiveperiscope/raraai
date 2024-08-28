from torch.utils.data import DataLoader
import lightning as L
from train_module import LitModel
from dataset import SpeechFeatureDataset, collate_fn

if __name__ == '__main__':
    config={
        'dataset': {
            'data_path': 'dataset_unconditional_50.parquet'
        },
        'train': {
            'lr': 3e-4
        },
        'model': {
            'bottleneck_size': 32
        }
    }

    train_dataset = SpeechFeatureDataset(
        data_path=config['dataset']['data_path'],
        split='train'
    )
    val_dataset = SpeechFeatureDataset(
        data_path=config['dataset']['data_path'],
        split='test')
    train_loader = DataLoader(train_dataset,
        num_workers=0, batch_size=2, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset,
        num_workers=0, batch_size=2, collate_fn=collate_fn, shuffle=False)

    lmodel = LitModel(
        config=config
    )
    trainer = L.Trainer(max_steps=20000)
    trainer.fit(model=lmodel, train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path='last')