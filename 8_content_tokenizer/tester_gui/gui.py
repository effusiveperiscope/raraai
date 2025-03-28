from functools import partial
from typing import Optional
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
import numpy as np
from tester_gui.file import FileButtonWithPreview
from tester_gui.audio import AudioPreviewWidget, AudioRecorder
from omegaconf import OmegaConf
from tester_gui.hf import ModelPathWorker, find_models_hf
from tester_gui.field import FieldWidget
from tester_gui.converters import Converters
from tester_gui.utils import runpdb
from model import TokenConvertModel
from preprocess import ContentTokenizer
from svc_helper.svc.rvc import RVCModel
import torch
import re
import os
import soundfile as sf

class Gui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RVC With Content Tokenizer Tester")
        self.setGeometry(100, 100, 800, 900)

        self.config = OmegaConf.load("gui_config.yaml")

        if not os.path.exists(self.config.record_dir):
            os.makedirs(self.config.record_dir)
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        self.initSystem()
        self.initUI()

    def initSystem(self):
        self.content_tokenizer = ContentTokenizer()
        self.model = TokenConvertModel()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.rvc_model = RVCModel()
        self.thread_pool = QThreadPool()
        self.converters = Converters(self.config)
        self.sid = None

    def initUI(self):
        self.main = QFrame(self)
        self.setCentralWidget(self.main)

        self.main_layout = QHBoxLayout(self.main)

        self.left_frame = QGroupBox()
        self.left_frame.setTitle("Model And Conversion")
        self.left_frame.setFixedWidth(400)
        self.modelAndConversion(self.left_frame)    

        self.right_frame = QGroupBox()
        self.right_frame.setTitle("Recording")
        self.recordFrame(self.right_frame)

        self.main_layout.addWidget(self.left_frame)
        self.main_layout.addWidget(self.right_frame)

        self.show()

    def modelAndConversion(self, frame):
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(QLabel("RVC model selection: "))
        self.modelCombo(layout)

        layout.addWidget(QLabel("Converter model selection: "))
        self.convertersCombo(layout)

        self.audio_in = FileButtonWithPreview()
        self.audio_ref = FileButtonWithPreview("Style reference")

        self.in_preview = AudioPreviewWidget()
        def updateInPreview(l: list):
            if len(l) == 0:
                return
            self.in_preview.from_file(l[0])
        self.audio_in.fileSelected.connect(updateInPreview)

        self.ref_preview = AudioPreviewWidget()
        def updateRefPreview(l: list):
            if len(l) == 0:
                return
            self.ref_preview.from_file(l[0])
        self.audio_ref.fileSelected.connect(updateRefPreview)

        layout.addWidget(QLabel("Input Audio: "))
        layout.addWidget(self.audio_in)
        layout.addWidget(self.in_preview)
        layout.addWidget(QLabel("Style reference: "))
        layout.addWidget(self.audio_ref)
        layout.addWidget(self.ref_preview)

        self.transpose = FieldWidget(
            label=QLabel("Transpose: "),
            field=QLineEdit("12"),
            validator=QIntValidator(-36, 36, self)
        )
        layout.addWidget(self.transpose)
        self.index_ratio = FieldWidget(
            label=QLabel("Index ratio: "),
            field=QLineEdit("0.0"),
            validator=QDoubleValidator(0, 1, 2, self)
        )
        layout.addWidget(self.index_ratio)
        self.rms_mix = FieldWidget(
            label=QLabel("RMS mix: "),
            field=QLineEdit("1.0"),
            validator=QDoubleValidator(0, 1, 2, self)
        )
        layout.addWidget(self.rms_mix)
        self.protect_uv_ratio = FieldWidget( # Protects unvoiced segments from indexing
            label=QLabel("Protect UV ratio: "),
            field=QLineEdit("0.33"),
            validator=QDoubleValidator(0, 1, 2, self)
        )
        layout.addWidget(self.protect_uv_ratio)
        self.use_Vevo_conversion = QCheckBox("Use Vevo conversion")
        layout.addWidget(self.use_Vevo_conversion)

        self.convert_button = QPushButton("Convert")
        layout.addWidget(self.convert_button)
        self.convert_button.clicked.connect(self.convert)

        self.output_preview = AudioPreviewWidget()
        layout.addWidget(self.output_preview)

    def modelCombo(self, layout: QLayout):
        modelCombo = QComboBox()
        model_map = {}
        for repo in self.config.hf_search_repos:
            repo_map = find_models_hf(repo)
            if self.config.hf_regex_filters.get(repo) is not None:
                repo_map = {k:v for k,v in repo_map.items() if re.search(self.config.hf_regex_filters[repo], k)}
            model_map.update(repo_map)
        self.model_map = model_map
        
        self.models_list = list(model_map.keys())
        for model_name in self.models_list:
            modelCombo.addItem(model_name)
        layout.addWidget(modelCombo)

        self.model_status = QLabel("Model status: ")
        layout.addWidget(self.model_status)

        modelCombo.currentIndexChanged.connect(self.modelChange)
        self.model_combo = modelCombo

    def modelPathWorkerFinished(self, data: dict):
        rvc_path = data["rvc_local_path"]
        index_path = data.get("index_local_path") # Can be None

        self.rvc_model.load_model(
            model_path=rvc_path,
            index_path=index_path)
        self.model_status.setText(f"Model status: Loaded {data['name']}")
        self.model_combo.setEnabled(True)

    def modelPathWorkerError(self, error: str):
        self.model_status.setText(f"Model status: {error}")
        self.model_combo.setEnabled(True)

    def modelChange(self, index):
        assert hasattr(self, 'models_list')
        assert hasattr(self, 'model_map')
        assert index < len(self.models_list)

        model_name = self.models_list[index]
        model_data = self.model_map[model_name]
        self.model_data = model_data

        self.model_path_worker = ModelPathWorker(model_data)
        emitter = self.model_path_worker.emitters
        emitter.finished.connect(self.modelPathWorkerFinished)
        emitter.error.connect(self.modelPathWorkerError)
        self.thread_pool.start(self.model_path_worker)
        self.model_status.setText(f"Model status: Loading {model_name}...")
        self.model_combo.setEnabled(False)

    def convertersChange(self, index):
        assert hasattr(self, 'convertersKeys')
        assert index < len(self.convertersKeys)
        self.converters_status.setText(f"Converter status: Loading {self.convertersKeys[index]}")
        self.sid = self.converters.load(self.convertersKeys[index], self.model)
        self.converters_status.setText(f"Converter status: Loaded {self.convertersKeys[index]}")

    def convertersCombo(self, layout: QLayout):
        convertersCombo = QComboBox()
        self.convertersKeys = list(self.converters.converters.keys())
        for converter in self.convertersKeys:
            convertersCombo.addItem(converter)

        layout.addWidget(convertersCombo)

        self.converters_status = QLabel("Converter status: ")
        layout.addWidget(self.converters_status)

        convertersCombo.currentIndexChanged.connect(self.convertersChange)
        self.converters_combo = convertersCombo

    def feature_override(self, padded_audio, sid: Optional[int]):
        with torch.no_grad():
            embed, _, masks = self.content_tokenizer.extract_hubert_codes(
                padded_audio.unsqueeze(0).to(self.device)
                .to(torch.float32))
            feats = self.model(embed, masks, sid=torch.tensor(sid).to(
                self.device if sid is not None else None).unsqueeze(0))
        return feats

    def opt_name(self, in_file_name, postfix='', extension="wav"):
        basename = os.path.basename(in_file_name)
        name = self.model_data['name']
        transpose = int(self.transpose.field.text())
        return f"{basename}_{name}_{transpose}{postfix}.{extension}"

    def convertFinished(self, data: list, params: dict):
        file = params['input_path']
        opt = np.array(data, dtype=np.int16)
        i = 0
        opt_name = os.path.join(
            self.config.output_dir, self.opt_name(file))
        while os.path.exists(opt_name):
            i += 1
            opt_name = os.path.join(
                self.config.output_dir, self.opt_name(file, f"_{i}"))
        sf.write(opt_name, opt, self.rvc_model.output_sample_rate())
        self.output_preview.from_file(opt_name)
        self.convert_button.setEnabled(True)

    def convertError(self, error: str):
        print(error)
        self.convert_button.setEnabled(True)
        
    def convert(self):
        hooks = {}
        if self.use_Vevo_conversion.isChecked():
            hooks['feature_override'] = partial(self.feature_override,
            sid = self.sid)
        for file in self.audio_in.file.files:
            params = {
                'input_path': file,
                'transpose': int(self.transpose.field.text()),
                'index_rate': float(self.index_ratio.field.text()),
                'rms_mix_rate': float(self.rms_mix.field.text()),
                'protect': float(self.protect_uv_ratio.field.text()),
                'extra_hooks': hooks
            }
            task = ConversionTask(self.rvc_model, params)
            task.emitters.finished.connect(self.convertFinished)
            task.emitters.error.connect(self.convertError)
            self.thread_pool.start(task)
        self.convert_button.setEnabled(False)

    def push_audio(self, file_path):
        self.audio_in.setFilesManually([file_path])

    def recordFrame(self, frame):
        layout = QVBoxLayout(frame)
        layout.setAlignment(Qt.AlignTop)

        self.recorder = AudioRecorder(
            push_fn=self.push_audio,
            record_dir=self.config.record_dir
        )
        layout.addWidget(self.recorder)

class ConversionTask(QRunnable):
    class Emitters(QObject):
        finished = pyqtSignal(list, dict)
        error = pyqtSignal(str)
        
    def __init__(self, rvc_model: RVCModel, params: dict):
        super().__init__()
        self.rvc_model = rvc_model
        self.params = params
        self.emitters = self.Emitters()

    def run(self):
        try:
            opt = self.rvc_model.infer_file(**self.params)
            self.emitters.finished.emit(opt.tolist(), self.params)
        except Exception as e:
            self.emitters.error.emit(str(e))

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gui = Gui()
    sys.exit(app.exec_())