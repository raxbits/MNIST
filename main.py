import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD

class_count = 10  # 0-9 digits
epochs_count = 100
batchS = 128

activation_set={1:'sigmoid',2:'relu',3:'tanh'}
optimizer_set={1:'Adam',2:'SGD'}

'''Format Data'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''Process Dataset'''
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_test /= 255
X_train /= 255
y_test = keras.utils.to_categorical(y_test, class_count)
y_train = keras.utils.to_categorical(y_train, class_count)


'''Model Setup'''
model = Sequential()
model.add(Dense((64), activation=activation_set[1], input_shape=(784,))) #Uses Sigmoid by default
model.add(Dense((10), activation='softmax'))

solver=SGD(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=solver, metrics=['mae', 'accuracy'])

'''Model Overview'''
model.summary()

'''Train + Validation'''
model.fit(X_train, y_train, batch_size=batchS, epochs=epochs_count, verbose=1, validation_data=(X_test, y_test))
