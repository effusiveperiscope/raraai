import datatia as dt
import torch
import torch.nn.functional as F
from einops import rearrange

# All voice conversion models require a 2x upsampling of input speech features
def interp2(tensor):
    tensor = tensor.squeeze(0)
    tensor = rearrange(tensor, "t d -> 1 d t")
    tensor = F.interpolate(tensor, scale_factor=2)
    tensor = rearrange(tensor, "1 d t -> t d")
    return tensor

def interp_row(row):
    row['whisper'] = interp2(row['whisper'])
    return row

def dataset(filelist, is_train: bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor,
                dim=torch.Size([-1, 1024]), keep_in_memory=False)
        ],
        actions=[
            dt.LiveMapRow(operation=interp_row),
            dt.RandomSubsample(fields=['whisper'], dims=[0], length=800), 
            dt.PadGroup(fields=['whisper'], dims=[0], values=[0])
        ]
    )

if __name__ == '__main__':
    dummy_filelist = [
        'test/test.whisper',
    ]
    ds = dataset(dummy_filelist, is_train=True)
    loader = ds.loader()

    for batch in loader:
        print([(k,x.shape) for k,x in batch.items()])