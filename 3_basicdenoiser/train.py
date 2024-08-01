from torch.utils.data import DataLoader
from model import FeatureDenoiser, LitFeatureDenoiser
from dataset import SpeechFeatureDataset, collate_fn
from huggingface_hub import hf_hub_download
import torch.distributed as dist

import lightning as L

if __name__ == '__main__':
    train_dataset = SpeechFeatureDataset()
    val_dataset = SpeechFeatureDataset(split='test')
    train_loader = DataLoader(train_dataset,
        num_workers=4, batch_size=4, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn)

    rvc_model = None

    if not dist.is_initialized() or dist.get_rank() == 0:
        # Mostly to suppress xformers warnings
        from svc_helper.svc.rvc import RVCModel
        rvc_model = RVCModel()
        test_model_path = hf_hub_download(repo_id='therealvul/RVCv2', 
            filename='RarityS1/Rarity.pth')
        test_index_path = hf_hub_download(repo_id='therealvul/RVCv2', 
            filename='RarityS1/added_IVF1866_Flat_nprobe_1_Rarity_v2.index')
        rvc_model.load_model(model_path = test_model_path,
            index_path = test_index_path)

    lmodel = LitFeatureDenoiser(config={
        'model': {
            'speech_emb_size': 768,
            'hidden_dim': 384,
            'dropout': 0.0,
            'n_layers': 6,
        },
    },)
    trainer = L.Trainer(max_steps=20000)
    trainer.fit(model=lmodel, train_dataloaders=train_loader,
        val_dataloaders=val_loader, ckpt_path='last')