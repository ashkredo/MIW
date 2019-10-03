import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
# from scipy.misc import toimage
import numpy as np
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,8):
            pyplot.subplot2grid((4,8),(i,j))
            pyplot.imshow(X[k])
            k = k+1
    # show the plot
    pyplot.show()
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(x_train, y_train, test_size=0.70, random_state=42)
# print(X_train_sm.shape)
# print(y_train_sm.shape)
# print(y_train_sm.shape[0])
# print(type( y_train_sm))

klass_01 = 0
klass_02 = 1

## remove all classes except  'dog' and  'ship' from train
indexes = np.array([], np.int)
for i in range(y_train_sm.shape[0]):
    if y_train_sm[i] != klass_01 and y_train_sm[i] != klass_02:
        indexes = np.append(indexes, i)
print('indexes: ', indexes)

y_train_sm = np.delete(y_train_sm, indexes, 0)
X_train_sm = np.delete(X_train_sm, indexes , 0)
##
print('X_train_sm: ',X_train_sm.shape)
print('y_train_sm: ', y_train_sm.shape)
print('y_train_sm: ', y_train_sm)
## remove all classes except  'dog' and  'ship' from test
indexes_test = np.array([], np.int)
for i in range(y_test_sm.shape[0]):
    if y_test_sm[i] != klass_01 and y_test_sm[i] != klass_02:
        indexes_test = np.append(indexes_test, i)
y_test_sm = np.delete(y_test_sm, indexes_test, 0)
X_test_sm = np.delete(X_test_sm, indexes_test , 0)
##
print('X_test_sm: ',X_test_sm.shape)
print('y_test_sm: ', y_test_sm.shape)

X_train_sm = X_train_sm.astype('float32') /255
y_train_sm = y_train_sm.astype('float32') /255

X_test_sm = X_test_sm.astype('float32') /255
y_test_sm = y_test_sm.astype('float32') /255

# X_train_subset = x_train[(y_train == 5) | (y_train == 8)]
# y_train_subset = y_train[(y_train == 5) | (y_train == 8)]
# print(y_train)
# print(y_train_subset)


#show_imgs(X_train_sm[:32])

print('X_train_sm.shape[1:]:', X_train_sm.shape[1:])

batch_size = 32
num_classes = 2
epochs = 1

####

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes , activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
y_train_sm = to_categorical(y_train_sm, num_classes= num_classes , dtype = 'int')
y_test_sm = to_categorical(y_test_sm, num_classes= num_classes , dtype = 'int')
print('y_train_sm : ', y_train_sm)
print('y_train_sm.shape :', y_train_sm.shape)

print('y_test_sm : ', y_test_sm)
print('y_test_sm.shape :', y_test_sm.shape)
model.fit(X_train_sm, y_train_sm, epochs = epochs, batch_size = batch_size)


#####


# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# y_train_sm = to_categorical(y_train_sm, num_classes= num_classes , dtype = 'int')
# y_test_sm = to_categorical(y_test_sm, num_classes= num_classes , dtype = 'int')

# model.fit(X_train_sm, y_train_sm,
#               batch_size=batch_size,
#               epochs=epochs,
#               shuffle=True)


# # # #


test_loss, test_acc = model.evaluate(X_test_sm, y_test_sm)
print('test_loss = ', test_loss)
print('test_acc = ', test_acc)
