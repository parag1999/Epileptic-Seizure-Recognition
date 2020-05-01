
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from lspopt import spectrogram_lspopt
from scipy.signal import spectrogram

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


if __name__ == "__main__":
    path = os.getcwd()+'/Epileptic Seizure Recognition.csv'
    dataset = order_data(path)
    sf = 178
    data = dataset["164"]["x"]
    fig = plt.figure()
    
    f, t, Sxx = spectrogram_lspopt(data, sf, c_parameter=20.0)
    #plt.subplot(4,2,2)
    plt.title("lspopt spec")
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    #plt.subplot(4,2,3)
    plt.title("matplotlib spec")
    plt.specgram(data, Fs=sf, cmap="viridis")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    f, t, Sxx = spectrogram(data, sf)
    #plt.subplot(4,2,4)
    plt.title("scipy spec")
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
#    plt.tight_layout()
#    plt.show()
    sf = 178
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    #pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft, cmap="viridis")
    f, t, Sxx = spectrogram_lspopt(data, sf, c_parameter=20.0)
    plt.pcolormesh(t, f, Sxx)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    
    