import datatia as dt
import torch

def dataset_ac(filelist, is_train : bool = True):
    return dt.Dataset(
        filelist=filelist,
        field_specs=[
            dt.FieldSpec(name='wave', datatype=torch.Tensor, dim=torch.Size([-1]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='spec', datatype=torch.Tensor, dim=torch.Size([-1, 769]), provide_length=True, keep_in_memory=False),
            dt.FieldSpec(name='f0', datatype=torch.Tensor, dim=torch.Size([-1]), keep_in_memory=False),
            dt.FieldSpec(name='sid', datatype=int)
        ],
        actions=[
            dt.PadGroup(fields=['spec', 'f0'], 
                dims = [0], values = [0], to_length=[257]), # fft_size // 2 + 1
            dt.PadGroup(fields=['wave'], dims=[0], values=[0]),
            dt.RandomSubsample(fields=['f0', 'spec', 'wave'], length=
                int(48000 / 480 * 8),
                frame_multiples=[1, 1, 480],
                dims=[0, 0, 0],),
        ],
        is_train=is_train
    )

if __name__ == '__main__':
    ds = dataset_ac('data/test/train.txt')
    for d in ds:
        for field, value in d.items():
            if type(value) == torch.Tensor:
                print(field, value.shape)