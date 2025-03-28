from model import TokenConvertModel
import os
import torch

class Converters:
    def __init__(self, config):
        self.converters_dir = config.converters_dir
        self.converters = {}
        if not os.path.exists(self.converters_dir):
            os.makedirs(self.converters_dir)

        self.preload()

    def preload(self):
        files = os.listdir(self.converters_dir)
        for file in files:
            if file.endswith(".pth"):
                basename = file.removesuffix(".pth")
                abspath = os.path.abspath(os.path.join(self.converters_dir, file))
                converter_data = torch.load(abspath, weights_only=True)
                spk_mapping = converter_data["spk_mapping"]
                for spk, id in spk_mapping.items():
                    key = f"{spk} [{basename}]"
                    self.converters[key] = {
                        'path': abspath,
                        'id': id
                    }

    def load(self, key: str, model: TokenConvertModel) -> int:
        if not key in self.converters:
            raise Exception(f"Converter {key} not found")
        state = torch.load(self.converters[key]["path"], weights_only=True)
        model.load_state_dict(state["model"])
        return self.converters[key]["id"]