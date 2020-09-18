import tkinter as tk
from tkinter import ttk
import requests
import datetime
from datetime import datetime
import quandl
import sys


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout, Activation
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.animation as animation
from matplotlib import style
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 


BIG_FONT = ("Verdana", 12)

import urllib
import json

import pandas as pd
import numpy as np



class BlockCharts(tk.Tk):
    def __init__(self, *args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        
        tk.Tk.wm_title(self, "BlockCharts") #title of app
        
        container = tk.Frame(self)
        
        container.pack(expand=True, fill = "both", side = "top")
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        
        self.frames = {}
        
        frame_list = [StartPage, PageOne, PageTwo, GraphPage, PageFour] #contains all pages
        
        for F in (frame_list):
            
            frame = F(container,self)
            
            self.frames[F] = frame
            
            frame.grid(row=0, column = 0, sticky = "nsew")
        
        self.show_frame(StartPage)
        
    def show_frame(self,container):
        frame = self.frames[container]
        frame.tkraise()
        
        
class StartPage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="Main Menu",font=BIG_FONT)        
        label.pack(pady=10,padx=10)
        
        button_p1 = ttk.Button(self,text="Display Bitcoin/USD Prices", command = lambda: controller.show_frame(PageOne))
        button_p1.pack()
        
        button_p2 = ttk.Button(self,text="Display Ethereum/USD Prices", command = lambda: controller.show_frame(PageTwo))
        button_p2.pack()
        
        button_graph = ttk.Button(self,text="Display XRP/USD Prices", command = lambda: controller.show_frame(GraphPage))
        button_graph.pack()
        
        button_p4 = ttk.Button(self,text="Display LiteCoin/USD Prices", command = lambda: controller.show_frame(PageFour))
        button_p4.pack()
        
        button_exit = ttk.Button(self,text="Exit", command = lambda: controller.quit())
        button_exit.pack()
        
        
class PageOne(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="Bitcoin/USD",font=BIG_FONT)        
        label.pack(pady=10,padx=10)
        
        button_start = ttk.Button(self,text="Back to Main Menu", command = lambda: controller.show_frame(StartPage))
        button_start.pack()
        
        
        look_back = 60
        forward_days = 1
        num_periods = 20
        
        #data
        
        today = datetime.today().strftime('%Y-%m-%d')
        df = yf.download("BTC-USD", start="2014-01-01", end=today)
        df = df["Close"]
        
        array = df.values.reshape(df.shape[0],1)
        scl = MinMaxScaler()
        array = scl.fit_transform(array)
        
        division = len(array) - num_periods*forward_days
        array_test = array[division-look_back:]
        array_train = array[:division]
        
        def processData(data, look_back, forward_days,jump=1):
            X,Y = [],[]
            for i in range(0,len(data) -look_back -forward_days +1, jump):
                X.append(data[i:(i+look_back)])
                Y.append(data[(i+look_back):(i+look_back+forward_days)])
            return np.array(X),np.array(Y)
        
        X_test,Y_test = processData(array_test,look_back,forward_days,forward_days)
        Y_test = np.array([list(a.ravel()) for a in Y_test])
        
        X,y = processData(array_train,look_back,forward_days)
        y = np.array([list(a.ravel()) for a in y])
        
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
        
        best_model = load_model('LSTM_Crypto.h5')
        Xt = best_model.predict(X_test)
        
        
        fig = matplotlib.figure.Figure(figsize = (15,10))      
        a = fig.add_subplot(111)
        
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Xt.reshape(-1,1)), color = 'blue')
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Y_test.reshape(-1,1)), color = 'red')
        a.set_xlabel('Days')
        a.set_ylabel('USD')
        pred_label = mpatches.Patch(color='blue', label='predictions')
        actual_label = mpatches.Patch(color='red', label='actual prices')
        a.legend(handles=[pred_label,actual_label])
        
        canvas = FigureCanvasTkAgg(fig,self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand = True)

        
        
class PageTwo(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="Ethereum/USD",font=BIG_FONT)        
        label.pack(pady=10,padx=10)
        
        button_start = ttk.Button(self,text="Back to Main Menu", command = lambda: controller.show_frame(StartPage))
        button_start.pack()
        
        
        look_back = 60
        forward_days = 1
        num_periods = 20
        
        #data
        
        today = datetime.today().strftime('%Y-%m-%d')
        df = yf.download("ETH-USD", start="2014-01-01", end=today)
        df = df["Close"]
        
        array = df.values.reshape(df.shape[0],1)
        scl = MinMaxScaler()
        array = scl.fit_transform(array)
        
        division = len(array) - num_periods*forward_days
        array_test = array[division-look_back:]
        array_train = array[:division]
        
        def processData(data, look_back, forward_days,jump=1):
            X,Y = [],[]
            for i in range(0,len(data) -look_back -forward_days +1, jump):
                X.append(data[i:(i+look_back)])
                Y.append(data[(i+look_back):(i+look_back+forward_days)])
            return np.array(X),np.array(Y)
        
        X_test,Y_test = processData(array_test,look_back,forward_days,forward_days)
        Y_test = np.array([list(a.ravel()) for a in Y_test])
        
        X,y = processData(array_train,look_back,forward_days)
        y = np.array([list(a.ravel()) for a in y])
        
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
        
        best_model = load_model('LSTM_Crypto.h5')
        Xt = best_model.predict(X_test)
        
        
        fig = matplotlib.figure.Figure(figsize = (15,10))      
        a = fig.add_subplot(111)
        
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Xt.reshape(-1,1)), color = 'blue')
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Y_test.reshape(-1,1)), color = 'red')
        a.set_xlabel('Days')
        a.set_ylabel('USD')
        pred_label = mpatches.Patch(color='blue', label='predictions')
        actual_label = mpatches.Patch(color='red', label='actual prices')
        a.legend(handles=[pred_label,actual_label])

        
        canvas = FigureCanvasTkAgg(fig,self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand = True)
        
class GraphPage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="XRP/USD",font=BIG_FONT)        
        label.pack(pady=10,padx=10)
        
        button_start = ttk.Button(self,text="Back to Main Menu", command = lambda: controller.show_frame(StartPage))
        button_start.pack()
        
        
        look_back = 60
        forward_days = 1
        num_periods = 20
        
        #data
        
        today = datetime.today().strftime('%Y-%m-%d')
        df = yf.download("XRP-USD", start="2014-01-01", end=today)
        df = df["Close"]
        
        array = df.values.reshape(df.shape[0],1)
        scl = MinMaxScaler()
        array = scl.fit_transform(array)
        
        division = len(array) - num_periods*forward_days
        array_test = array[division-look_back:]
        array_train = array[:division]
        
        def processData(data, look_back, forward_days,jump=1):
            X,Y = [],[]
            for i in range(0,len(data) -look_back -forward_days +1, jump):
                X.append(data[i:(i+look_back)])
                Y.append(data[(i+look_back):(i+look_back+forward_days)])
            return np.array(X),np.array(Y)
        
        X_test,Y_test = processData(array_test,look_back,forward_days,forward_days)
        Y_test = np.array([list(a.ravel()) for a in Y_test])
        
        X,y = processData(array_train,look_back,forward_days)
        y = np.array([list(a.ravel()) for a in y])
        
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
        
        best_model = load_model('LSTM_Crypto.h5')
        Xt = best_model.predict(X_test)
        
        
        fig = matplotlib.figure.Figure(figsize = (15,10))      
        a = fig.add_subplot(111)
        
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Xt.reshape(-1,1)), color = 'blue')
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Y_test.reshape(-1,1)), color = 'red')
        a.set_xlabel('Days')
        a.set_ylabel('USD')
        pred_label = mpatches.Patch(color='blue', label='predictions')
        actual_label = mpatches.Patch(color='red', label='actual prices')
        a.legend(handles=[pred_label,actual_label])
        
        canvas = FigureCanvasTkAgg(fig,self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand = True)
        
class PageFour(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self,text="LiteCoin/USD",font=BIG_FONT)        
        label.pack(pady=10,padx=10)
        
        button_start = ttk.Button(self,text="Back to Main Menu", command = lambda: controller.show_frame(StartPage))
        button_start.pack()
        
        
        look_back = 60
        forward_days = 1
        num_periods = 20
        
        #data
        
        today = datetime.today().strftime('%Y-%m-%d')
        df = yf.download("LTC-USD", start="2014-01-01", end=today)
        df = df["Close"]
        
        array = df.values.reshape(df.shape[0],1)
        scl = MinMaxScaler()
        array = scl.fit_transform(array)
        
        division = len(array) - num_periods*forward_days
        array_test = array[division-look_back:]
        array_train = array[:division]
        
        def processData(data, look_back, forward_days,jump=1):
            X,Y = [],[]
            for i in range(0,len(data) -look_back -forward_days +1, jump):
                X.append(data[i:(i+look_back)])
                Y.append(data[(i+look_back):(i+look_back+forward_days)])
            return np.array(X),np.array(Y)
        
        X_test,Y_test = processData(array_test,look_back,forward_days,forward_days)
        Y_test = np.array([list(a.ravel()) for a in Y_test])
        
        X,y = processData(array_train,look_back,forward_days)
        y = np.array([list(a.ravel()) for a in y])
        
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
        
        best_model = load_model('LSTM_Crypto.h5')
        Xt = best_model.predict(X_test)
        
        
        fig = matplotlib.figure.Figure(figsize = (15,10))      
        a = fig.add_subplot(111)
        
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Xt.reshape(-1,1)), color = 'blue')
        a.plot(np.arange(0, 20, 1),scl.inverse_transform(Y_test.reshape(-1,1)), color = 'red')
        a.set_xlabel('Days')
        a.set_ylabel('USD')
        pred_label = mpatches.Patch(color='blue', label='predictions')
        actual_label = mpatches.Patch(color='red', label='actual prices')
        a.legend(handles=[pred_label,actual_label])
        
        canvas = FigureCanvasTkAgg(fig,self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill = tk.BOTH, expand = True)
        
        
app = BlockCharts()
app.geometry("1280x720")
app.mainloop()