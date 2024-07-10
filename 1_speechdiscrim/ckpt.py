from model import SpeechClassifier1
import torch
import os
import re

def search_checkpoint(folder, pattern=r'model_(\d+).pt'):
    if not (os.path.exists(folder) and os.path.isdir(folder)):
        return None
    ckpts = []
    for f in os.listdir(folder):
        result = re.match(pattern=pattern, string=f)
        if not result:
            continue
        ckpts.append({'ckpt': os.path.join(folder, result.group()),
            'epoch': int(result.group(1))})
    if len(ckpts):
        ckpts = sorted(ckpts, key= lambda x: x['epoch'])
        return ckpts[-1]['ckpt']
    else:
        return None

def load_checkpoint(conf, filename, optimizer=None, strict=True):
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'],
            strict=strict)

    model = SpeechClassifier1(
        hidden_dim=conf['model']['hidden_dim'],
         n_speakers=len(checkpoint['label_encoder']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    return (checkpoint['epoch'], checkpoint['best_accuracy'],
        checkpoint['label_encoder'], model)

def save_checkpoint(epoch, model, optimizer, best_accuracy,
    label_encoder, ckpt_folder = ''):
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'label_encoder': label_encoder.classes_
    }
    filename = f'model_{epoch}.pt'
    print(f'Saving checkpoint to {filename}')
    torch.save(checkpoint, os.path.join(ckpt_folder, filename))
    pass