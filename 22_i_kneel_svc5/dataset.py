import datatia as dt
import torch
import torch.nn.functional as F
from einops import rearrange

def interp2(tensor):
    tensor = tensor.squeeze(0)
    tensor = rearrange(tensor, "t d -> 1 d t")
    tensor = F.interpolate(tensor, scale_factor=2)
    tensor = rearrange(tensor, "1 d t -> t d")
    return tensor

def process_row(row):
    row['whisper'] = interp2(row['whisper'])
    row['hubert'] = interp2(row['hubert'])
    return row

def drop_len_check(row):
    if row['whisper'].shape[1] < 16:
        return True

def dataset(filelist, is_train : bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor, dim=torch.Size([-1, 1280]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='hubert', datatype=torch.Tensor, dim=torch.Size([-1, 256]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='f0', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='f0_inharm', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='f0_subharm', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='f0_confidence', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='spec', datatype=torch.Tensor, dim=torch.Size([-1, 100]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='spk', datatype=torch.Tensor, dim=torch.Size([256]), keep_in_memory=False),
            dt.FieldSpec(name='wave', datatype=torch.Tensor, dim=torch.Size([-1]), provide_length=True, keep_in_memory=False),
        ],
        actions=[
            dt.LiveMapRow(operation=process_row),
            dt.RandomSubsample(fields=['whisper', 'hubert', 'f0', 'f0_inharm', 'f0_subharm', 'f0_confidence', 'spec', 'wave'], length=
                int(48000 / 480 * 8),
                frame_multiples=[1, 1, 1, 1, 1, 1, 1, 480],
                dims=[0, 0, 0, 0, 0, 0, 0, 0],),
            dt.PadGroup(fields=['whisper', 'hubert', 'f0', 'f0_inharm', 'f0_subharm', 'f0_confidence', 'spec'], 
            dims = [0, 0, 0, 0, 0, 0, 0], values = [0, 0, 0, 0, 0, 0, 0], to_length=[257, 257, 257, 257, 257, 257, 257]), # fft_size // 2 + 1
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ],
        is_train=is_train
    )


def dataset_f0(filelist, is_train : bool = True):
    lines = []
    with open(filelist, encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line.split('|')[2])
    return dt.Dataset(
        filelist=lines,
        field_specs = [
            dt.FieldSpec(name='f0', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
        ],
        actions=[
            dt.PadGroup(fields=['f0'], dims=[0], values=[0], to_multiple=[2])
        ],
        is_train=is_train
    )

if __name__ == '__main__':
    dataset_f0('data/applejack_sing/train.txt')