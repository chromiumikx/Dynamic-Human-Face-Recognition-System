#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from mainfrwk import *

class MainWindow(Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)


class Example(Ui_Form):
    
    def __init__(self):
        super().__init__()

    def showColor(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.setStyleSheet(self.style+"#window{background:%s}" % col.name())
    def showDialog(self):
        text, ok = QInputDialog.getText(self, '对话框', 
            '请输入你的名字:')
        
        if ok:
            self.linet1.setText(str(text))
    def showFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.readline()
                self.linet1.setText(data) 
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
