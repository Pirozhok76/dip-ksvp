from PyQt5 import QtCore, QtGui, uic

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, \
    QDesktopWidget, QAction, qApp, QFileDialog

from PyQt5.QtGui import QIcon

import sys


from calc import calcul


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        self.initUI()

    def initUI(self):

        self.wnd = uic.loadUi('dippr.ui', self) #загрузка файла интерфейса wnd = window

        self.wnd.setWindowIcon(QIcon('icon.png')) #иконка окна

        self.saveAction = self.wnd.action

        self.saveAction.setShortcut('Ctrl+S')

        self.saveAction.setStatusTip('Сохранить результаты в файл')

        self.saveAction.triggered.connect(self.showDialog)

        self.exitAction = self.wnd.action_2
        # self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Выход из приложения')  # подсказка на статус-баре

        # self.wnd.action.setEnabled(False)

        self.show()

    def showDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Сохранить файл', '/results.txt')[0]
        f = open(fname, 'w')
        # f.write(self.wnd.textEdit.toPlainText() + self.wnd.textEdit_2.toPlainText())
        f.close()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())