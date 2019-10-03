import keras
import matplotlib.pyplot as plt 
import numpy as np
import random
import tensorflow as tf  
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def main() :
    dataset = pd.read_csv('data/winequality-red.csv',sep=';')

    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    output = ['quality']
    X = dataset[features].values
    Y = dataset[output].values

    X/8.0
    Y/8.0

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

    input_layer = keras.layers.Dense(200, activation=tf.nn.sigmoid)
    secret_layer = keras.layers.Dense(180, activation=tf.nn.sigmoid)
    secret_layer2 = keras.layers.Dense(180, activation=tf.nn.sigmoid)
    output_layer = keras.layers.Dense(9, activation=tf.nn.softmax)

    model = keras.Sequential([input_layer,secret_layer,secret_layer2, output_layer])
    model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])    
    print(y_train)

    model.fit(X_train, y_train, epochs=5)
   
    loss, accuracy = model.evaluate(X_test,y_test)
    print('Network accuracy:', accuracy)
    print('Network loss:', loss)

if __name__ == '__main__':
    main()

