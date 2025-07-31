import datatia as dt
import torch
import torch.nn.functional as F
from einops import rearrange

def interp2(tensor):
    tensor = rearrange(tensor, "t d -> 1 d t")
    tensor = F.interpolate(tensor, scale_factor=2)
    tensor = rearrange(tensor, "1 d t -> t d")
    return tensor

def interp_row(row):
    row['whisper'] = interp2(row['whisper'])
    row['hubert'] = interp2(row['hubert'])
    return row

def dataset(filelist, is_train : bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor, dim=torch.Size([-1, 1280]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='hubert', datatype=torch.Tensor, dim=torch.Size([-1, 256]), keep_in_memory=False),
            dt.FieldSpec(name='f0', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='spec', datatype=torch.Tensor, dim=torch.Size([-1, 100]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='spk', datatype=torch.Tensor, dim=torch.Size([256]), keep_in_memory=False),
            dt.FieldSpec(name='wave', datatype=torch.Tensor, dim=torch.Size([-1]), provide_length=True, keep_in_memory=False),
        ],
        actions=[
            dt.LiveMapRow(operation=interp_row),
            dt.PadGroup(fields=['whisper', 'hubert', 'f0', 'spec'], 
            dims = [0, 0, 0, 0], values = [0, 0, 0, 0]),
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
        ],
        is_train=is_train
    )

if __name__ == '__main__':
    dummy_filelist = [
        'test/test.whisper|test/test.hubert|test/test.f0|test/test.spec|test/test.spk|test/test.wave',
    ]
    ds = dataset(dummy_filelist, is_train=True)
    loader = ds.loader()

    for batch in loader:
        print([(k,x.shape) for k,x in batch.items()])