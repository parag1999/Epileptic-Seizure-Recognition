from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout ,Flatten , Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from util import preprocess_data
import os

def construct_model(input_shape):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

    

if __name__ == "__main__":
    os.chdir('..')
    path = os.getcwd()+'dataset/Epileptic Seizure Recognition.csv'
    X, Y = preprocess_data(path)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    
    imheight, imwidth = (36, 54)
    input_shape = (imheight, imwidth, 1)

    model = construct_model(input_shape)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=[ModelCheckpoint('model_binary.h5',save_best_only=True)])
    
