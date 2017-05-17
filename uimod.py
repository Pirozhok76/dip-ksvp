from PyQt5 import QtCore, QtGui, uic

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QDesktopWidget, QAction, qApp, QFileDialog

from PyQt5.QtGui import QIcon


import math

import sys


from calc import calcul
from MainDia import MainDia





class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        self.initUI()

    def initUI(self):



        self.wnd = uic.loadUi("./UIs/dippr.ui", self) #загрузка файла интерфейса wnd = window

        # QtCore.QTimer.singleShot(3000, self.showMainDia)

        self.wnd.setWindowIcon(QIcon("./resources/icon.png")) #иконка окна

        appearance = self.palette()
        appearance.setColor(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                            QtGui.QColor("white"))

        self.setPalette(appearance)

        self.setAutoFillBackground(True)

        self.saveAction = self.wnd.action

        self.saveAction.setShortcut('Ctrl+S')

        self.saveAction.setStatusTip('Сохранить результаты в файл')

        self.saveAction.triggered.connect(self.showDialog)

        self.exitAction = self.wnd.action_3

        self.wnd.btn1.clicked.connect(self.runMW)

        self.wnd.btn2.clicked.connect(self.showMainDia)


        # self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Выход из приложения')  # подсказка на статус-баре

        self.exitAction.triggered.connect(qApp.quit)
        # self.wnd.action.setEnabled(False)

        self.show()

    def showMainDia(self):
        self.show_MainDia = MainDia(self)
        self.show_MainDia.InitUI()

    def showDialog(self):  #запись результатов в файл
        fname = QFileDialog.getSaveFileName(self, 'Сохранить файл', '/results.txt')[0]
        f = open(fname, 'w')
        # f.write()
        f.close()

    def runMW(self):
        self.calc_part(self.wnd.textEdit)
        # pass

    def calc_part(self, textEdit):
        if self.wnd.radioButton_1.isChecked():
            eps = calcul.epsi[0]
            # sqreps = math.tan(eps)

        elif self.wnd.radioButton_2.isChecked():
            eps = calcul.epsi[1]
            # sqreps = math.tan(eps)

        elif self.wnd.radioButton_3.isChecked():
            eps = calcul.epsi[2]
            # sqreps = math.tan(eps)

        r2 = self.wnd.dblSpinBox_1.value()

        # res1 = calcul.peripheral(r2, eps)
        res2 = calcul.axial(eps, r2)

        textEdit.append(str(res2))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())