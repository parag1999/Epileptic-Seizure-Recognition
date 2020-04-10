# -*- coding: utf-8 -*-

import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout ,Flatten , Dense
from keras.callbacks import ModelCheckpoint
#from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Scipy signal spectogram
#f, t, Sxx = signal.spectrogram(dataset["164"]["x"], 178)
#plt.pcolormesh(t, f, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.savefig('foo_scipy_signal_spectogram.png')


def order_data(path):
    a = pd.read_csv(path)
    a = a.rename(columns={"Unnamed":"id"})
    c = a.copy()
    d = {}
    dataset = {}
    for idx,row in a.iterrows():    
        c.at[idx,"id"] = re.sub('(\.V1\.)','.',row["id"])
    for idx,row in c.iterrows():
        time, uid = row["id"].split('.')
        time = int(time[1:])
        if uid in d:
             d[uid].append((time,row[1:]))
        else:
            d[uid] =[(time,row[1:])]
    d_copy = d.copy()
    for uid in d_copy:
        d[uid].sort(key = lambda x:x[0])
    for uid in d:
        dataset[uid] = {
                'x':[],
                'y':int(d[uid][0][1]["y"])
                }
        for time in d[uid]:
            partial_x = time[1][:-1].to_list()
            dataset[uid]['x'].extend(partial_x)
    for uid in dataset:
        dataset[uid]["x"] = np.array(dataset[uid]["x"])
    
    return dataset

def get_spectrogram(data):
    rate = 178
    nfft= 178
    noverlap= 177
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())


def preprocess_data():
    path = os.getcwd()+'/Epileptic Seizure Recognition.csv'
    dataset = order_data(path)
    X = []
    Y = []
    for uid in dataset:
        spectrogram = get_spectrogram(dataset[uid]["x"])
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        if dataset[uid]["y"]>1:
            y = 0
        else:
            y=1    
        X.append(normgram)
        Y.append(y)     
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y
    

if __name__ == "__main__":
    X, Y = preprocess_data()
    imheight, imwidth = (36, 54)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    input_shape = (imheight, imwidth, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint('model_train.h5',save_best_only=True)])
    
