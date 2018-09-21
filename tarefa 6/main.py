import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
# Building the model
xor = Sequential()
# Add required layers
xor.add(Dense(8, input_dim=X.shape[1], activation="tanh"))
xor.add(Dense(1, activation="sigmoid"))

sgd = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

xor.compile(loss="binary_crossentropy", optimizer=sgd, metrics = ["accuracy"])
# Uncomment this line to print the model architecture
xor.summary()
# Fitting the model
history = xor.fit(X, y, epochs=50, verbose=0)
# Scoring the model
score = xor.evaluate(X, y)

print("\nAccuracy: ", score[-1])
print("\nPredictions: ", xor.predict_proba(X))
