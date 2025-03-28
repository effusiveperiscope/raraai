from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class FieldWidget(QFrame):
    def __init__(self, label: QLabel, field, validator=None):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(label)
        self.field = field
        field.setAlignment(Qt.AlignRight)
        field.sizeHint = lambda: QSize(60, 32)
        field.setSizePolicy(QSizePolicy.Maximum,
            QSizePolicy.Preferred)
        self.layout.addWidget(field)

        if validator is not None:
            field.setValidator(validator)

    def setEnabled(self, val):
        self.field.setEnabled(val)