import torch
sid_avgs1 = 'data/mlp_base/sid_avgs.pt'
sid_avgs2 = 'data/sing_tiny/sid_avgs.pt'

sid_avgs1 = torch.load(sid_avgs1)
offset_sid_avgs1 = len(sid_avgs1)
sid_avgs2 = torch.load(sid_avgs2)
sid_avgs2 = {str(int(sid) + offset_sid_avgs1)
    : emb for sid, emb in sid_avgs2.items()}

new_sid_avgs = sid_avgs1 | sid_avgs2
print(list(new_sid_avgs.keys()))
torch.save(new_sid_avgs, 'data/sid_avgs.pt')