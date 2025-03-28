import os
from huggingface_hub import list_repo_files, hf_hub_download
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import asyncio

def find_models_hf(repo_name: str):
    repo_files = list_repo_files(repo_name)
    model_map = {}

    for file in repo_files:
        # .pth -> RVC model file
        if file.endswith(".pth"):
            model_name = os.path.dirname(file)
            if not model_name in model_map:
                model_map[model_name] = {}

            model_map[model_name]["rvc_weight"] = file
            model_map[model_name]["name"] = model_name
            model_map[model_name]["repo_name"] = repo_name

        # .index -> Retrieval feature index
        if file.endswith(".index"):
            model_name = os.path.dirname(file)
            if not model_name in model_map:
                model_map[model_name] = {}

            model_map[model_name]["feature_index"] = file
            model_map[model_name]["name"] = model_name
            model_map[model_name]["repo_name"] = repo_name

    return model_map

async def dl_wrapper(repo_name: str, path: str):
    return await asyncio.to_thread(hf_hub_download, repo_id=repo_name, filename=path)

class ModelPathWorker(QRunnable):
    class Emitters(QObject):
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

    def __init__(self, model_data: dict):
        super().__init__()
        self.model_data = model_data
        self.repo_name = model_data['repo_name']
        self.emitters = ModelPathWorker.Emitters()

    def run(self):
        try:
            rvc_path, index_path = asyncio.run(self.download_paths())
        except Exception as e:
            self.emitters.error.emit(str(e))
            return
        self.emitters.finished.emit({
            "rvc_local_path": rvc_path, 
            "index_local_path": index_path,
            "name": self.model_data["name"]})

    async def download_paths(self):
        rvc_weight = self.model_data["rvc_weight"]
        feature_index = self.model_data["feature_index"]

        # Run both downloads concurrently
        rvc_task = dl_wrapper(self.repo_name, rvc_weight)
        index_task = dl_wrapper(self.repo_name, feature_index)

        return await asyncio.gather(rvc_task, index_task)  # Wait for both to finish