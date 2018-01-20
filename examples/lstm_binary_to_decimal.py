"""Seq2Seq model that converts variable length number to binary"""

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Activation
import numpy as np
import math
from generate_data import generate_binary_data
from character_table import CharacterTable

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#Hyperparameters
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 1  # Number of epochs to train for.
HIDDEN_SIZE = 128
LAYERS = 1
NUM_SAMPLES = 10000  # Number of samples to train on.
RNN = LSTM
NR_ITERATIONS = 100

BINARY_LEN = math.ceil(math.log(NUM_SAMPLES, 2))
DECIMAL_LEN = math.ceil(math.log(NUM_SAMPLES, 10))

#---DATA---
#Generate data
DATA = generate_binary_data(limit=NUM_SAMPLES)
CHARS = "0123456789"
CHAR_TABLE = CharacterTable(CHARS)

#Padding and to string
X, Y = [], []
for x, y in DATA:
    X.append("0" * (BINARY_LEN-len(str(x))) + str(x))
    Y.append("0" * (DECIMAL_LEN-len(str(y))) + str(y))

#Vectorization
#For every sample, have n rows which can be each encoded with a character
X_ENCODED = np.zeros((NUM_SAMPLES, BINARY_LEN, len(CHARS)), dtype=np.bool)
Y_ENCODED = np.zeros((NUM_SAMPLES, DECIMAL_LEN, len(CHARS)), dtype=np.bool)
for i, x in enumerate(X):
    X_ENCODED[i] = CHAR_TABLE.encode(x, BINARY_LEN)
for i, y in enumerate(Y):
    Y_ENCODED[i] = CHAR_TABLE.encode(y, DECIMAL_LEN)

#Shuffle and split data
INDICES = np.arange(NUM_SAMPLES)
np.random.shuffle(INDICES)
X_ENCODED = X_ENCODED[INDICES]
Y_ENCODED = Y_ENCODED[INDICES]

SPLIT_INDEX = len(X) - len(Y) // 10
(X_TRAIN, X_VAL) = X_ENCODED[:SPLIT_INDEX], X_ENCODED[SPLIT_INDEX:]
(Y_TRAIN, Y_VAL) = Y_ENCODED[:SPLIT_INDEX], Y_ENCODED[SPLIT_INDEX:]

print('Training Data:')
print(X_TRAIN.shape)
print(Y_TRAIN.shape)

print('Validation Data:')
print(X_VAL.shape)
print(Y_VAL.shape)


#---MODEL---
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(BINARY_LEN, len(CHARS))))
model.add(RepeatVector(DECIMAL_LEN))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(TimeDistributed(Dense(len(CHARS))))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(NR_ITERATIONS):

    model.fit(X_TRAIN, Y_TRAIN,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_VAL, Y_VAL))
    
    if iteration%10 == 0:
        print()
        print('-' * 50)
        print('Iteration', iteration)
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(100):
            ind = np.random.randint(0, len(X_VAL))
            rowx, rowy = X_VAL[np.array([ind])], Y_VAL[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = CHAR_TABLE.decode(rowx[0])
            correct = CHAR_TABLE.decode(rowy[0])
            guess = CHAR_TABLE.decode(preds[0], calc_argmax=False)
            print('Q', q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)