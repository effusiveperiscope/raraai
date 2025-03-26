import torch
# randomly subsample feat along the sequence dimension
def subsample_features(feat, subsample_frac):
    feat_seq_len = feat.shape[1]
    max_subsample_len = int(feat_seq_len * subsample_frac)
    subsample_len = torch.randint(1, max_subsample_len + 1, (1,)).item()
    start_idx = torch.randint(0, feat_seq_len - subsample_len + 1, (1,)).item()
    return feat[:, start_idx:start_idx + subsample_len]