# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainfrwk.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor,QColor

class Ui_Form(QtWidgets.QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(600, 400)

        self.style = """ 
                QPushButton{background-color:grey;color:white;} 
                #window{ background:pink; }
                #test{ background-color:black;color:white; }
            """
        Form.setStyleSheet(self.style)
        Form.setWindowFlags(Qt.FramelessWindowHint)
        Form.setWindowOpacity(0.95)
        
        self.bnt_exit = QtWidgets.QPushButton(Form)
        self.bnt_exit.setGeometry(QtCore.QRect(570, 10, 21, 21))
        self.bnt_exit.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:/Program Files/锐捷网络/Ruijie Supplicant/resource/skin/btn_del_list_normal.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bnt_exit.setIcon(icon)
        self.bnt_exit.setFlat(True)
        self.bnt_exit.setObjectName("bnt_exit")
        self.bnt_max = QtWidgets.QPushButton(Form)
        self.bnt_max.setGeometry(QtCore.QRect(540, 10, 21, 21))
        self.bnt_max.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("D:/Program Files/锐捷网络/Ruijie Supplicant/resource/skin/btn_system_max_normal.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bnt_max.setIcon(icon1)
        self.bnt_max.setFlat(True)
        self.bnt_max.setObjectName("bnt_max")
        self.bnt_min = QtWidgets.QPushButton(Form)
        self.bnt_min.setGeometry(QtCore.QRect(510, 10, 21, 21))
        self.bnt_min.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("D:/Program Files/锐捷网络/Ruijie Supplicant/resource/skin/btn_system_min_normal.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bnt_min.setIcon(icon2)
        self.bnt_min.setFlat(True)
        self.bnt_min.setObjectName("bnt_min")
        self.data = QtWidgets.QGroupBox(Form)
        self.data.setGeometry(QtCore.QRect(30, 20, 241, 151))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(12)
        self.data.setFont(font)
        self.data.setAutoFillBackground(True)
        self.data.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.data.setFlat(True)
        self.data.setCheckable(False)
        self.data.setObjectName("data")
        self._label_1 = QtWidgets.QLabel(self.data)
        self._label_1.setGeometry(QtCore.QRect(10, 30, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_1.setFont(font)
        self._label_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self._label_1.setMidLineWidth(0)
        self._label_1.setTextFormat(QtCore.Qt.AutoText)
        self._label_1.setWordWrap(False)
        self._label_1.setObjectName("_label_1")
        self._label_2 = QtWidgets.QLabel(self.data)
        self._label_2.setGeometry(QtCore.QRect(10, 70, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_2.setFont(font)
        self._label_2.setObjectName("_label_2")
        self.textEdit_username_1 = QtWidgets.QTextEdit(self.data)
        self.textEdit_username_1.setGeometry(QtCore.QRect(93, 30, 141, 31))
        self.textEdit_username_1.setObjectName("textEdit_username_1")
        self.textEdit_userid_1 = QtWidgets.QTextEdit(self.data)
        self.textEdit_userid_1.setGeometry(QtCore.QRect(93, 70, 141, 31))
        self.textEdit_userid_1.setObjectName("textEdit_userid_1")
        self.btn_ok_collect = QtWidgets.QPushButton(self.data)
        self.btn_ok_collect.setGeometry(QtCore.QRect(130, 110, 101, 31))
        self.btn_ok_collect.setObjectName("btn_ok_collect")
        self.recognition = QtWidgets.QGroupBox(Form)
        self.recognition.setGeometry(QtCore.QRect(300, 50, 281, 311))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.recognition.setFont(font)
        self.recognition.setAutoFillBackground(True)
        self.recognition.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.recognition.setFlat(True)
        self.recognition.setCheckable(False)
        self.recognition.setObjectName("recognition")
        self._label_3 = QtWidgets.QLabel(self.recognition)
        self._label_3.setGeometry(QtCore.QRect(10, 30, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_3.setFont(font)
        self._label_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self._label_3.setMidLineWidth(0)
        self._label_3.setTextFormat(QtCore.Qt.AutoText)
        self._label_3.setWordWrap(False)
        self._label_3.setObjectName("_label_3")
        self._label_4 = QtWidgets.QLabel(self.recognition)
        self._label_4.setGeometry(QtCore.QRect(10, 70, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_4.setFont(font)
        self._label_4.setObjectName("_label_4")
        self.textEdit_username_2 = QtWidgets.QTextEdit(self.recognition)
        self.textEdit_username_2.setGeometry(QtCore.QRect(93, 30, 181, 31))
        self.textEdit_username_2.setObjectName("textEdit_username_2")
        self.textEdit_userid_2 = QtWidgets.QTextEdit(self.recognition)
        self.textEdit_userid_2.setGeometry(QtCore.QRect(93, 70, 181, 31))
        self.textEdit_userid_2.setObjectName("textEdit_userid_2")
        self.btn_ok_re = QtWidgets.QPushButton(self.recognition)
        self.btn_ok_re.setGeometry(QtCore.QRect(170, 110, 101, 31))
        self.btn_ok_re.setObjectName("btn_ok_re")
        self.train = QtWidgets.QGroupBox(Form)
        self.train.setGeometry(QtCore.QRect(30, 190, 241, 191))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.train.setFont(font)
        self.train.setAutoFillBackground(True)
        self.train.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.train.setFlat(True)
        self.train.setCheckable(False)
        self.train.setObjectName("train")
        self._label_5 = QtWidgets.QLabel(self.train)
        self._label_5.setGeometry(QtCore.QRect(10, 30, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_5.setFont(font)
        self._label_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self._label_5.setMidLineWidth(0)
        self._label_5.setTextFormat(QtCore.Qt.AutoText)
        self._label_5.setWordWrap(False)
        self._label_5.setObjectName("_label_5")
        self._label_6 = QtWidgets.QLabel(self.train)
        self._label_6.setGeometry(QtCore.QRect(10, 70, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self._label_6.setFont(font)
        self._label_6.setObjectName("_label_6")
        self.textEdit_username_3 = QtWidgets.QTextEdit(self.train)
        self.textEdit_username_3.setGeometry(QtCore.QRect(93, 30, 141, 31))
        self.textEdit_username_3.setObjectName("textEdit_username_3")
        self.textEdit_userid_3 = QtWidgets.QTextEdit(self.train)
        self.textEdit_userid_3.setGeometry(QtCore.QRect(93, 70, 141, 31))
        self.textEdit_userid_3.setObjectName("textEdit_userid_3")
        self.btn_ok_train = QtWidgets.QPushButton(self.train)
        self.btn_ok_train.setGeometry(QtCore.QRect(130, 150, 101, 31))
        self.btn_ok_train.setObjectName("btn_ok_train")
        self.textEdit_none = QtWidgets.QTextEdit(self.train)
        self.textEdit_none.setGeometry(QtCore.QRect(93, 110, 141, 31))
        self.textEdit_none.setObjectName("textEdit_none")
        self.bnt_exit.raise_()
        self.bnt_max.raise_()
        self.bnt_min.raise_()
        self.data.raise_()
        self.recognition.raise_()
        self.train.raise_()


        # 绑定默认按钮
        self.bnt_exit.clicked.connect(self.close)
        self.bnt_min.clicked.connect(self.showMinimized)
        self.bnt_max.clicked.connect(self.showMaximized)

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
        self.data.setTitle(_translate("Form", "数据收集模块"))
        self._label_1.setText(_translate("Form", "User Name"))
        self._label_2.setText(_translate("Form", "User ID"))
        self.btn_ok_collect.setText(_translate("Form", "开始"))
        self.recognition.setTitle(_translate("Form", "检测识别模块"))
        self._label_3.setText(_translate("Form", "User Name"))
        self._label_4.setText(_translate("Form", "User ID"))
        self.btn_ok_re.setText(_translate("Form", "开始"))
        self.train.setTitle(_translate("Form", "模型训练模块"))
        self._label_5.setText(_translate("Form", "User Name"))
        self._label_6.setText(_translate("Form", "User ID"))
        self.btn_ok_train.setText(_translate("Form", "开始"))

