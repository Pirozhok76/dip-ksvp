from PyQt5 import QtCore, QtGui, uic

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QDesktopWidget, QAction, qApp, QFileDialog

from PyQt5.QtGui import QIcon

import math

import sys


from calc import calcul


class MainDia(QMainWindow):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        # self.InitUI()

    def InitUI(self):
        appearance = self.palette()
        appearance.setColor(QtGui.QPalette.Normal, QtGui.QPalette.Window,
                            QtGui.QColor("white"))

        self.setPalette(appearance)

        self.setAutoFillBackground(True)

        self.wnd2 = uic.loadUi("./UIs/MainDia.ui", self)

        self.show()

    def runDia(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainDia()
    sys.exit(app.exec_())