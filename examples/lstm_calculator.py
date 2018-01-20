"""Making a full calculator with the help of seq2seq
Expansion upon 'lstm_addition.py' and 'addition_rnn.py'"""

from keras.models import Sequential
from keras import layers
import numpy as np
from character_table import CharacterTable
from generate_data import generate_data
import math

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#__PARAMETERS__
RNN = layers.LSTM
HIDDEN_SIZE = 64
LAYERS = 1
N_EPOCH = 1
N_BATCH = 1
LIMIT = 100
N_EXAMPLES = 20000 #TODO use number that can be evenly divided by number of operations
LENGTH = math.ceil(math.log10(LIMIT*LIMIT)) #TODO fix length
SUM_LENGTH = LENGTH*2+1 #two numbers and an operation sign
ITERATIONS = 10

#__ENCODING__
SIGNS = "+-*/"
CHARS = "0123456789 " + SIGNS
CHAR_TABLE = CharacterTable(CHARS)

VECTORIZED_INPUT = np.zeros((N_EXAMPLES, SUM_LENGTH, len(CHARS)), dtype=np.bool)
VECTORIZED_TARGET = np.zeros((N_EXAMPLES, LENGTH, len(CHARS)), dtype=np.bool)
for j,sign in enumerate(SIGNS):
    INDEX_SIGN = j*N_EXAMPLES // len(SIGNS)
    #__DATA__
    INPUT, TARGET = [], []
    X, Y = generate_data(sign, n_examples=N_EXAMPLES // len(SIGNS), limit=LIMIT)

    #__PADDING__
    #TODO remove duplicates
    for x in X: INPUT.append(sign.join([str(n) for n in x]).rjust(LENGTH))
    for y in Y: TARGET.append(str(y).rjust(LENGTH))

    #__VECTORIZE__
    for i, calculation in enumerate(INPUT):
        VECTORIZED_INPUT[i + INDEX_SIGN] = CHAR_TABLE.encode(calculation, SUM_LENGTH)
    for i, answer in enumerate(TARGET):
        VECTORIZED_TARGET[i + INDEX_SIGN] = CHAR_TABLE.encode(answer, LENGTH)

#__SHUFFLE__
INDICES = np.arange(N_EXAMPLES)
np.random.shuffle(INDICES)
VECTORIZED_INPUT = VECTORIZED_INPUT[INDICES]
VECTORIZED_TARGET = VECTORIZED_TARGET[INDICES]

#__SPLIT__
SPLIT = N_EXAMPLES - N_EXAMPLES // 10
(INPUT_TRAIN, INPUT_VAL) = VECTORIZED_INPUT[:SPLIT], VECTORIZED_INPUT[SPLIT:]
(TARGET_TRAIN, TARGET_VAL) = VECTORIZED_TARGET[:SPLIT], VECTORIZED_TARGET[SPLIT:]

#__MODEL__
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(SUM_LENGTH, len(CHARS))))
model.add(layers.RepeatVector(LENGTH))
for _ in range(LAYERS): 
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(CHARS))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print('Training Data:')
print(INPUT_TRAIN.shape)
print(TARGET_TRAIN.shape)

print('Validation Data:')
print(INPUT_VAL.shape)
print(TARGET_VAL.shape)

#__TRAIN__
for iteration in range(ITERATIONS):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(INPUT_TRAIN, TARGET_TRAIN, 
    batch_size=N_BATCH,
    epochs=N_EPOCH,
    validation_data=(INPUT_VAL, TARGET_VAL))

    for i in range(100):
        index = np.random.randint(0, len(INPUT_VAL))
        row_x, row_y = INPUT_VAL[np.array([index])], TARGET_VAL[np.array([index])]

        calculation = CHAR_TABLE.decode(row_x[0])
        target = CHAR_TABLE.decode(row_y[0])
        predictions = model.predict_classes(row_x, verbose=0)
        prediction = CHAR_TABLE.decode(predictions[0], calc_argmax=False)
        
        print('Q', calculation, end=' ')
        print('T', target, end=' ')
        if target == prediction: print(colors.ok + '☑' + colors.close, end=' ')
        else: print(colors.fail + '☒' + colors.close, end=' ')
        print(prediction)



