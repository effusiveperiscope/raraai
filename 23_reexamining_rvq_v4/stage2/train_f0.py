from modeling.pitch import F0FM
from omegaconf import OmegaConf
import pytorch_lightning as pl

class TrainModule(pl.LightningModule):
    def __init__(self, net: F0FM, config: OmegaConf):
        super().__init__()
        self.net = net
        self.config = config

    def step(self, batch):
        # TODO we need to introduce a non-random version of the loss for validation
        # X we need to make preprocess.py output a filelist that just has the F0 files (?)
        # TODO actually change the preprocessed dataset for rarity
        # TODO add train_f0.txt , val_f0.txt to the config
        # TODO change preprocess.py to include the speaker embedding (?)
        # TODO what do f0_target and f0 mean?! Something seems to be wrong with our model
        f0_target = batch['f0']
        pass