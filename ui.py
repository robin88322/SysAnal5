from sys import argv
import time
#from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.uic import loadUiType
from PyQt5 import uic
import random
#from logic import *


form_class, base_class = loadUiType('data/main.ui')
class AppWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.widgetList = []    
        self.show()
    def closeEvent(self, event):
        event.accept()

app = 0
app = QApplication(argv)
w = AppWindow()
w.show()

def input():
    #filename = QFileDialog.directory(w, "Choose catalog", ".", QFileDialog.ReadOnly)
    filename = QFileDialog.getExistingDirectory(w, "Select Directory")
    filename = filename + "/"
    w.line_input.setText(filename)

def upload(): # тут треба ебнути вигрузку з екселя і порахувати початкові ймовірності і передіти 
# в три таблички data, experts, probability
    print("lioha pidor")
    for i in range(8):
        for j in range(7):
            print(i)
            w.data.setItem(i , j, QTableWidgetItem(str(random.randint(1, 10))))  
    for i in range(7):
        w.experts.setItem(0 , i, QTableWidgetItem("1"))

def scenario():
    # file = w.line_input.text()
    # eta_def = w.doubleSpinBox.value()
    res = []
    # func_type = w.comboBox.currentText()
    # res, matrix = get_results(file, eta_def, func_type)
    # w.lineEdit_s1.setText(res[0])
    w.L.setText("L: " + str(random.random()))
    w.eps.setText("eps: " + str(random.random()))
    w.D.setText("D: " + str(random.random()))
    
    print(w.monte.value())
 
#w.clearButton.clicked.connect(clear_function) 

w.select_input.clicked.connect(input)
w.build_scenario.clicked.connect(scenario)
w.upload.clicked.connect(upload)

app.exec_()
