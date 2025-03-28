from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *

class FileButton(QPushButton):
    fileDropped = pyqtSignal(list)
    def __init__(self, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            clean_files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                clean_files.append(url.toLocalFile())
            self.fileDropped.emit(clean_files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class SimpleFileButton(QPushButton):
    fileSelected = pyqtSignal(list)
    def __init__(self, label = "Files to Convert"):
        super().__init__(label)
        self.setAcceptDrops(True)
        self.clicked.connect(self.file_dialog)
        self.files = []

    def file_dialog(self):
        self.files = QFileDialog.getOpenFileNames(
            self, "Files to process")[0]
        self.fileSelected.emit(self.files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.files = []
            for url in event.mimeData().urls():
                if not url.toLocalFile():
                    continue
                self.files.append(url.toLocalFile())
            self.fileSelected.emit(self.files)
            event.acceptProposedAction()
        else:
            event.ignore()
        pass

class FileButtonWithPreview(QWidget):
    fileSelected = pyqtSignal(list)

    def __init__(self, label = "Files to Convert", preview = False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.file = SimpleFileButton(label)
        
        self.label = QLabel("Files: ")

        self.layout.addWidget(self.file)
        self.layout.addWidget(self.label)

        self.file.fileSelected.connect(self.fileSelected)
        self.file.fileSelected.connect(self.updateLabel)

    def setFilesManually(self, files):
        self.file.files = files
        self.file.fileSelected.emit(self.file.files)

    def updateLabel(self):
        self.label.setText(f"Files: {self.file.files}")
