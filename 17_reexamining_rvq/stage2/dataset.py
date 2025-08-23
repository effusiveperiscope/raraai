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
    return row

def dataset(filelist, is_train : bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor, dim=torch.Size([-1, 1280]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='f0', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='spec', datatype=torch.Tensor, dim=torch.Size([-1, 100]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='spk', datatype=torch.Tensor, dim=torch.Size([256]), keep_in_memory=False),
            dt.FieldSpec(name='wave', datatype=torch.Tensor, dim=torch.Size([-1]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='sid', datatype=int),
        ],
        actions=[
            dt.LiveMapRow(operation=process_row),
            dt.RandomSubsample(fields=['whisper', 'f0', 'spec', 'wave'], length=
                int(48000 / 480 * 4), # 4 seconds
                frame_multiples=[1, 1, 1, 480],
                dims=[0, 0, 0, 0],),
            dt.PadGroup(fields=['whisper', 'f0', 'spec'], 
            dims = [0, 0, 0], values = [0, 0, 0]),
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ],
        is_train=is_train
    )

if __name__ == '__main__':
    dummy_filelist = [
        'data/test/test_2_01.wav.whisper|data/test/test_2_01.wav.f0|data/test/test_2_01.wav.spec|data/test/test_2_01.wav.spk|data/test/test_2_01.wav.wave|0',
    ]
    ds = dataset(dummy_filelist, is_train=True)
    loader = ds.loader()

    for batch in loader:
        print([(k,x.shape) for k,x in batch.items()])