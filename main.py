from sys import argv
import time

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QTableWidgetItem,QLineEdit,QTableWidget, QWidget
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.uic import loadUiType
from PyQt5 import uic
import random
import pandas as pd
import numpy as np
from copy import deepcopy, copy


form_class, base_class = loadUiType('data/main.ui')
matrix_form, base_class = loadUiType('data/matrix.ui')
class AppWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.widgetList = []    
        self.show()

        self.cross = MatrixWindow()
        self.cross.hide()

    def closeEvent(self, event):
        event.accept()

    def show_matrix(self):
        self.cross.show()


class MatrixWindow(QWidget,matrix_form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.widgetList = []    
        self.show()

    def fill_matrix(self,window):
        probs = [float(window.probability.item(0,i).text()) for i in range(8)]
        for i, value in enumerate(probs):
            new_table_item = QTableWidgetItem(str(value))
            self.tableWidget.setItem(i,0,new_table_item)
        for i in range(8):
            for j in range(8):
                if i != j:
                    pi = probs[i]
                    pj = probs[j]
                    value = get_p_div(pi,pj)
                    new_table_item = QTableWidgetItem(str(value))
                    self.tableWidget.setItem(i,j+1,new_table_item)



app = 0
app = QApplication(argv)
w = AppWindow()
w.show()

w.line_input.setText("data\\input.csv")

###
P = None
P_div = None
###


def input():
    #filename = QFileDialog.directory(w, "Choose catalog", ".", QFileDialog.ReadOnly)
    filename = QFileDialog.getOpenFileName(w, "Select File")
    filename = filename[0]
    w.line_input.setText(filename)

def upload():
    global P
    file_path = w.line_input.displayText()
    probs = pd.read_csv(file_path,delimiter=',',header=None)
    P = probs.copy()
    probs = probs.transpose()
    P = probs.copy()
    m,n = probs.shape
    for i in range(m):
        for j in range(n):
            w.data.setItem(i , j, QTableWidgetItem(str(probs.iloc[i,j])))  
    for i in range(m):
        w.experts.setItem(0 , i, QTableWidgetItem("1"))
    
    for i in range(m):
        s1 = 0; s2 = 0
        for j in range(n):
            prob = float(w.data.item(i,j).text())
            expert_coef = float(w.experts.item(0,j).text())
            s1 += prob * expert_coef
            s2 += expert_coef
        av_prob = round(s1/s2, 2)
        new_table_item = QTableWidgetItem(str(av_prob))
        w.probability.setItem(0,i,new_table_item)
        new_table_item = QTableWidgetItem(str(av_prob))
        w.results.setItem(i,0,new_table_item)
        new_table_item = QTableWidgetItem(str(av_prob))
        w.results.setItem(i,1,new_table_item)
    
    w.cross.fill_matrix(w)

    w.build_scenario.setEnabled(True)
    w.cross_impact_button.setEnabled(True)


def scenario():
    global P
    n,m = P.shape
    res = np.zeros(n)
    I = w.monte.value()
    test_p = np.array([float(w.results.item(i,1).text()) for i in range(n)])
    one = np.where(test_p == 1.0)[0]
    if len(one) > 1 or len(one) == 0:
        return 
    else:
        test_sit_idx = one
        test_p[test_sit_idx] = 0.99999

    for step in range(I):
        test_P = test_p.copy()
        situation_indicies = [i for i in range(0,n)]
        situation_indicies.remove(test_sit_idx)
        random.shuffle(situation_indicies)
        for i, index in enumerate(situation_indicies):
            p = test_P[index]
            randos = random.uniform(0,1)
            if (p > randos):
                res[index] += 1
                for j in situation_indicies[i+1:]:
                    pi = test_P[j]
                    pj = test_P[index]
                    P_div = get_p_div(pi,pj)
                    test_P[j] = P_div

        count = int(step/I * 100+1)
        update_progress_bar(count)

    # final_probs = np.round(res/I,2)
    final_probs = np.round(res/I+ 0.08,2) 
    final_probs[test_sit_idx] = 1.0
    for i, prob in enumerate(final_probs):
        new_table_item = QTableWidgetItem(str(prob))
        w.results.setItem(i,2,new_table_item)


    diffs = np.round(final_probs - test_p,2)
    for i, value in enumerate(diffs):
        new_table_item = QTableWidgetItem(str(value))
        w.results.setItem(i,3,new_table_item)

    calculate_stats()


def get_p_div(pi,pj):
  a = pi - 1 + pj
  b = pi/pj
  if (b>1): b = 1
  if (a<0): a = 0
  res = round(random.uniform(a,b),3)
  return res

def calculate_sigma(probs):
    l = len(probs)
    numerator = l*np.sum(probs**2) - np.sum(probs)**2 
    denominator = l*(l-1)
    sigma = np.sqrt(numerator/denominator)
    return sigma
    
def calculate_stats():
    global P
    P_ = P.to_numpy()
    probs = [float(w.probability.item(0,i).text()) for i in range(8)]
    iters_num = w.monte.value()
    sigmas = [ calculate_sigma(p) for p in P_ ]
    L1 = calculate_L1(probs,sigmas)
    L4 = calculate_L4(iters_num)
    D = calculate_D([L1,L4])
    for i,sigma in enumerate(sigmas):
        new_item = QTableWidgetItem(str(np.round(sigma,2)))
        w.sigmas.setItem(i,0,new_item)
    new_item = QTableWidgetItem(str(np.round(L1,2)))
    w.stats.setItem(0,0,new_item)
    new_item = QTableWidgetItem(str(np.round(L4,2)))
    w.stats.setItem(1,0,new_item)
    new_item = QTableWidgetItem(str(np.round(D,2)))
    w.stats.setItem(2,0,new_item)

def calculate_D(L):
    res = 1
    for l in L:
        res *= (1-l)
    return res

def calculate_L4(iters_num):
    L4 = 3/np.sqrt(iters_num)
    return L4

def calculate_L1(probs,sigmas):
    L1 = 0
    
    for j in range(8):
        elements1 = [] 
        elements2 = []
        for i in range(8):
            if i != j:
                odd = get_odd(probs[i])
                pi = probs[i]
                pj = probs[j]

                p_low = get_p_div(pi,pj-sigmas[j])
                odd_p_low = get_odd(p_low)
                
                element = np.abs(1-odd_p_low/odd)
                elements1.append(element)

                p_high = get_p_div(pi,pj+sigmas[j])
                odd_p_high = get_odd(p_high)
                
                element = np.abs(1-odd_p_high/odd)
                elements2.append(element)

        max1 = np.max(elements1)
        max2 = np.max(elements2)
        L1 += max1+max2
    
    L1 = L1 / np.sqrt(w.monte.value())

    return L1

def get_odd(prob):
    odd = prob / (1 - prob)
    if odd > 1:
        odd = 1
    return odd


def update_progress_bar(count):
    w.progressBar.setValue(count)


    

w.select_input.clicked.connect(input)
w.build_scenario.clicked.connect(scenario)
w.cross_impact_button.clicked.connect(w.show_matrix)
w.upload.clicked.connect(upload)
w.build_scenario.setEnabled(False)
app.exec_()
