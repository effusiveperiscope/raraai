import datatia as dt
import torch
import torch.nn.functional as F
from einops import rearrange


def squeezeop(row):
    row['wave'] = row['wave'].squeeze()
    return row

def dataset(filelist, is_train: bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='wave', datatype=torch.Tensor,
                dim=torch.Size([-1]), keep_in_memory=False,
                provide_length=True)
        ],
        actions=[
            dt.LiveMapRow(operation=squeezeop),
            dt.RandomSubsample(fields=['wave'], dims=[0], length=16000*3), # 3 seconds
            dt.PadGroup(fields=['wave'], dims=[0], values=[0])
        ]
    )