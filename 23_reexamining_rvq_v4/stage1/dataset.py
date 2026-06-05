import datatia as dt
import torch
import torch.nn.functional as F
import os
from pathlib import Path
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

    filelines = []
    with open(filelist) as f:
        filelines = f.readlines()

    def relpath(fileline):
        # Assume the filelist path is correct, and the data is in the parent of the filelist
        fileline = fileline.strip()
        parent = Path(filelist).parent 
        name = Path(fileline).name
        return str(parent / name)
        
    if len(filelines) > 0 and not os.path.exists(filelines[0]):
        print(f"Checking for relative path resolution of data @ {Path(filelist).parent} ...")
        # Check for relative paths
        if os.path.exists(relpath(filelines[0])):
            print("Found")
            filelines = [relpath(x) for x in filelines]
        else:
            raise ValueError(f"Could not find data files @ {Path(filelist).parent}")

    return dt.Dataset(
        filelist=filelines,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor,
                dim=torch.Size([-1, 1024]), keep_in_memory=False, provide_length=True)
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