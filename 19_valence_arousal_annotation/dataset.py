import datatia as dt
import torch

def dataset(filelist, is_train: bool):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='whisper', datatype=torch.Tensor,
                dim=torch.Size([-1, 1024]), keep_in_memory=False),
            dt.FieldSpec(name='valence', datatype=int),
            dt.FieldSpec(name='arousal', datatype=int),
        ],
        actions=[
            dt.PadGroup(fields=['whisper'], dims=[0], values=[0])
        ]
    )

if __name__ == '__main__':
    data = dataset('data/test/filelist.txt', is_train=True)