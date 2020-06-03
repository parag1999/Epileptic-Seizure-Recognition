import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


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
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft, cmap="viridis")
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


def preprocess_data(path):
    dataset = order_data(path)
    X = []
    Y = []
    for uid in dataset:
        spectrogram = get_spectrogram(dataset[uid]["x"])
        graygram = rgb2gray(spectrogram)
        normgram = normalize_gray(graygram)
        if dataset[uid]["y"]>1:
            y = 0
        else:
            y=1    
        X.append(normgram)
        Y.append(y)     
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y
