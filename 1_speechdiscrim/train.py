import torch
import torch.nn as nn
import torch.optim as optim
from model import SpeechClassifier1
from dataset import (SpeechClassDataset0, BalancedSampler, AvoidSampler,
    collate_fn)
from ckpt import load_checkpoint, save_checkpoint, search_checkpoint
from metrics import (macro_precision, macro_recall, make_histogram,
    make_confusion)
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import logging
from logging import info
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train():
    torch.manual_seed(0)
    np.random.seed(0)

    conf = OmegaConf.load('config.yaml')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    draw_interval = 2
    log_interval = 1
    save_interval = 2
    upperb = 16
    model_name = 'full'
    data_folder = 'dataset_over5min.parquet'
    # model_name = 'test_2000'
    # data_folder = 'dataset_2000.parquet'
    warmstart_ckpt = None

    train_dataset = SpeechClassDataset0(split='train', data_path=data_folder)
    val_dataset = SpeechClassDataset0(split='test', data_path=data_folder)

    model = SpeechClassifier1(
        hidden_dim=conf['model']['hidden_dim'],
        n_speakers=train_dataset.n_speakers)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # file stuff
    default_ckpt_folder = model_name
    if not os.path.exists(default_ckpt_folder):
        os.makedirs(default_ckpt_folder)
    val_dataset.dump_to_file(os.path.join(default_ckpt_folder, 'val_list.txt'))
    logger = logging.getLogger()
    logfile_handler = logging.FileHandler(os.path.join(default_ckpt_folder,
         'train_log.txt'))
    logger.addHandler(logfile_handler)

    # Unfortunately things start to break on Windows with num_workers > 0
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, 
        num_workers=16,
        collate_fn=collate_fn,
        sampler=BalancedSampler(train_dataset.classes(),
            avoid_idxs=set(), alpha=0.8))
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, 
        num_workers=4,
        collate_fn=collate_fn,
        sampler=AvoidSampler(val_dataset.classes(),
            avoid_idxs=set()))

    def trainable_params(model):
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        return sum([np.prod(p.size()) for p in model_params])

    n_epochs = 1000

    if warmstart_ckpt is None:
        search_ckpt = search_checkpoint(default_ckpt_folder)
        if search_ckpt:
            info(f"resuming training from {search_ckpt}")
            start_epoch, best_acc, _, model = load_checkpoint(conf, search_ckpt)
        else:
            start_epoch = 0
            best_acc = 0.0
    else:
        info(f"warmstart from {warmstart_ckpt}")
        _, _, _, model = load_checkpoint(conf, warmstart_ckpt, strict=False)
        start_epoch = 0
        best_acc = 0.0

    model = model.to(device)
    info(f"Model with {trainable_params(model)} trainable params")
    info(f"Beginning training with {train_dataset.n_speakers} classes")
    info(f"(Random guess is {1.0/train_dataset.n_speakers})")
    for epoch in range(start_epoch, n_epochs):
        total_loss = torch.Tensor([0])

        model.train()
        pred_list = []
        gt_list = []
        for speech_features, gt_label in tqdm(train_dataloader):

            speech_features = speech_features.to(device)
            gt_label = gt_label.to(device)

            pred_label = model(speech_features)
            loss = loss_fn(pred_label, gt_label)

            pred_idx = pred_label.argmax(dim=1)
            pred_list += pred_idx.tolist()
            gt_list += gt_label.tolist()

            with torch.no_grad():
                total_loss += loss.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ((epoch) % draw_interval) == 0 and epoch > 0:
            make_histogram(gt_list, epoch, 'train_gt',
                n_speakers=train_dataset.n_speakers,
                by_epoch=False,
                ckpt_folder=default_ckpt_folder)
            make_histogram(pred_list, epoch, 'train_pred',
                n_speakers=train_dataset.n_speakers,
                ckpt_folder=default_ckpt_folder)

        total_loss /= len(train_dataloader)

        model.eval()
        val_loss = torch.Tensor([0])
        num_correct = torch.Tensor([0])
        pred_list = []
        gt_list = []
        with torch.no_grad():
            for speech_features, gt_label in val_dataloader:

                speech_features = speech_features.to(device)
                gt_label = gt_label.to(device)

                pred_label = model(speech_features)
                loss = loss_fn(pred_label, gt_label)

                pred_idx = pred_label.argmax(dim=1)
                num_correct += torch.sum(pred_idx == gt_label).cpu()
                val_loss += loss.cpu()

                pred_list += pred_idx.tolist()
                gt_list += gt_label.tolist()

            val_loss /= len(val_dataloader)
            acc = num_correct / (len(val_dataloader)*val_dataloader.batch_size)
            if (acc > best_acc):
                best_acc = acc.item()

            if ((epoch) % log_interval) == 0:
                info(f'========== Epoch {epoch} ==========')
                info(f'Total train loss: {total_loss.item()}')
                info(f'Total val loss: {val_loss.item()}')
                info(f'Val acc: {acc.item()} | {num_correct.item()} / '
                    f'{len(val_dataloader)*val_dataloader.batch_size}')
                info(f'Precision: {macro_precision(gt_list, pred_list)} '
                    f'Recall: {macro_recall(gt_list, pred_list)}')
            
            if ((epoch) % draw_interval) == 0 and epoch > 0:
                make_histogram(gt_list, epoch, 'val_gt',
                    n_speakers=train_dataset.n_speakers,
                    by_epoch=False, 
                    ckpt_folder=default_ckpt_folder)
                make_histogram(pred_list, epoch, 'val_pred',
                    n_speakers=train_dataset.n_speakers,
                    ckpt_folder=default_ckpt_folder)
                make_confusion(gt_list, pred_list, 'val',
                    epoch, train_dataset.label_encoder,
                    default_ckpt_folder)

            if ((epoch) % save_interval) == 0 and epoch > 0:
                save_checkpoint(epoch, model, optimizer, best_acc,
                    train_dataset.label_encoder,
                    ckpt_folder=default_ckpt_folder)

if __name__ == '__main__':
    train()