# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QWidget,QPushButton,
    QLineEdit,QHBoxLayout, QVBoxLayout,QColorDialog,QInputDialog,QFileDialog)
from PyQt5.QtGui import QCursor,QColor


class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        
        self.style = """ 
                QPushButton{background-color:grey;color:white;} 
                #window{ background:pink; }
                #test{ background-color:black;color:white; }
            """
        Form.setStyleSheet(self.style)
        Form.setWindowFlags(Qt.FramelessWindowHint)
        Form.setWindowOpacity(0.9)
        
        self.pushButton = QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(370, 10, 21, 21))
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:/Program Files/锐捷网络/Ruijie Supplicant/resource/skin/btn_del_list_normal.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setFlat(True)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(340, 10, 21, 21))
        self.pushButton_2.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("D:/Program Files/锐捷网络/Ruijie Supplicant/resource/skin/btn_system_max_normal.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(310, 10, 21, 21))
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton.clicked.connect(self.close)
        self.pushButton_3.clicked.connect(self.showMinimized)
        self.pushButton_2.clicked.connect(self.showMaximized)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    #重写三个方法使我们的Example窗口支持拖动,上面参数window就是拖动对象
    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            self.m_drag=True
            self.m_DragPosition=event.globalPos()-self.pos()
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_drag:
            self.move(QMouseEvent.globalPos()-self.m_DragPosition)
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_drag=False
        self.setCursor(QCursor(Qt.ArrowCursor))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_3.setText(_translate("Form", "-"))

