"""Making a full calculator with the help of seq2seq
Expansion upon 'lstm_addition.py' and 'addition_rnn.py'"""

from keras.models import Sequential, model_from_json
from keras import layers
import numpy as np
from character_table import CharacterTable
from generate_data import generate_data
import math
from os.path import isfile

from hyperopt import Trials, tpe, STATUS_OK, fmin, hp

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#__ENCODING__
SIGNS = "+"
CHARS = "0123456789 " + SIGNS
CHAR_TABLE = CharacterTable(CHARS)
#__PARAMETERS__
PARAMETERS = {
    "RNN":hp.choice("RNN",[layers.LSTM]),
    "HIDDEN_SIZE":hp.choice("HIDDEN_SIZE",[64]),
    "LAYERS":hp.choice("LAYERS", [1]),
    "N_EPOCH":3,
    "N_BATCH":1,
    "LIMIT":100,
    "N_EXAMPLES":10000,
    "ITERATIONS":10,
    "CHAR_TABLE":CHAR_TABLE,
    "CHARS":CHARS,
    "SIGNS":SIGNS
}
PARAMETERS["LENGTH"] = math.ceil(math.log10(PARAMETERS["LIMIT"]*PARAMETERS["LIMIT"]))
PARAMETERS["SUM_LENGTH"] = PARAMETERS["LENGTH"]*2+1

def data(param):
    VECTORIZED_INPUT = np.zeros((param["N_EXAMPLES"], param["SUM_LENGTH"], len(param["CHARS"])), dtype=np.bool)
    VECTORIZED_TARGET = np.zeros((param["N_EXAMPLES"], param["LENGTH"], len(param["CHARS"])), dtype=np.bool)

    for j,sign in enumerate(param["SIGNS"]):
        INDEX_SIGN = j*param["N_EXAMPLES"] // len(param["SIGNS"])
        #__DATA__
        INPUT, TARGET = [], []
        X, Y = generate_data(sign, n_examples=param["N_EXAMPLES"] // len(param["SIGNS"]), limit=param["LIMIT"])

        #__PADDING__
        #TODO remove duplicates
        for x in X: INPUT.append(sign.join([str(n) for n in x]).rjust(param["LENGTH"]))
        for y in Y: TARGET.append(str(y).rjust(param["LENGTH"]))

        #__VECTORIZE__
        for i, calculation in enumerate(INPUT):
            VECTORIZED_INPUT[i + INDEX_SIGN] = param["CHAR_TABLE"].encode(calculation, param["SUM_LENGTH"])
        for i, answer in enumerate(TARGET):
            VECTORIZED_TARGET[i + INDEX_SIGN] = param["CHAR_TABLE"].encode(answer, param["LENGTH"])

    #__SHUFFLE__
    INDICES = np.arange(param["N_EXAMPLES"])
    np.random.shuffle(INDICES)
    VECTORIZED_INPUT = VECTORIZED_INPUT[INDICES]
    VECTORIZED_TARGET = VECTORIZED_TARGET[INDICES]

    #__SPLIT__
    SPLIT = param["N_EXAMPLES"] - param["N_EXAMPLES"] // 10
    (INPUT_TRAIN, INPUT_VAL) = VECTORIZED_INPUT[:SPLIT], VECTORIZED_INPUT[SPLIT:]
    (TARGET_TRAIN, TARGET_VAL) = VECTORIZED_TARGET[:SPLIT], VECTORIZED_TARGET[SPLIT:]

    print('Training Data:')
    print(INPUT_TRAIN.shape)
    print(TARGET_TRAIN.shape)

    print('Validation Data:')
    print(INPUT_VAL.shape)
    print(TARGET_VAL.shape)

    return INPUT_TRAIN, TARGET_TRAIN, INPUT_VAL, TARGET_VAL

def write_to_disk(model):
    model_json = model.to_json()
    with open("model_lstm_calculator.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_lstm_calculator.h5")
    print("saved model to disk")

def read_from_disk():
    json_file = open("model_lstm_calculator.json", "r")
    loaded_model_json = json_file.read()
    json_file.close
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_lstm_calculator.h5")
    print("loaded model from disk")
    return loaded_model
        
def create_model(param):
    #__MODEL__
    model = Sequential()
    model.add(param["RNN"](param["HIDDEN_SIZE"], input_shape=(param["SUM_LENGTH"], len(param["CHARS"]))))
    model.add(layers.RepeatVector(param["LENGTH"])) #split input into sequences of length
    for _ in range(param["LAYERS"]): 
        model.add(param["RNN"](param["HIDDEN_SIZE"], return_sequences=True))


    model.add(layers.TimeDistributed(layers.Dense(len(param["CHARS"])))) #for loop over output
    model.add(layers.Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    #__TRAIN__
    for iteration in range(param["ITERATIONS"]):

        # if isfile("model_lstm_calculator.json"):
        #     model = read_from_disk()
        #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print()
        print('-' * 50)
        print('Iteration', iteration)

        model.fit(INPUT_TRAIN, TARGET_TRAIN, 
        batch_size=param["N_BATCH"],
        epochs=param["N_EPOCH"],
        validation_data=(INPUT_VAL, TARGET_VAL))

        for i in range(10):
            index = np.random.randint(0, len(INPUT_VAL))
            row_x, row_y = INPUT_VAL[np.array([index])], TARGET_VAL[np.array([index])]

            calculation = param["CHAR_TABLE"].decode(row_x[0])
            target = param["CHAR_TABLE"].decode(row_y[0])
            predictions = model.predict_classes(row_x, verbose=0)
            prediction = param["CHAR_TABLE"].decode(predictions[0], calc_argmax=False)
            
            print('Q', calculation, end=' ')
            print('T', target, end=' ')
            if target == prediction: print(colors.ok + '☑' + colors.close, end=' ')
            else: print(colors.fail + '☒' + colors.close, end=' ')
            print(prediction)

        # write_to_disk(model)

    score, acc = model.evaluate(INPUT_VAL, TARGET_VAL, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

INPUT_TRAIN, TARGET_TRAIN, INPUT_VAL, TARGET_VAL = data(PARAMETERS)
best = fmin(create_model, PARAMETERS, algo=tpe.suggest, max_evals=5, trials=Trials())
print("best: {}".format(best))