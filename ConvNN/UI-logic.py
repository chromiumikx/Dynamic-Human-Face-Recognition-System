#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from ConvNN.mainfrwk import *

from  ConvNN.cam_faces_whl import *
from ConvNN.para_config import *
from ConvNN.io_whl import *
from ConvNN.detection_whl import *


class MainWindow(Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.btn_ok_collect.clicked.connect(self.collect)

    def collect(self):
        collect_user_data()
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
