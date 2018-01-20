"""Following tutorial from:
https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/"""

#%%
"""One-to-One LSTM for sequence prediction
Split the sequence up in input-output pairs and predict the sequence one step at a time."""
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
# prepare sequence
length = 5 
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(length,1,1)  # length samples, 1 time step, 1 feature
y = seq.reshape(length,1)
# define LSTM config
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('{}'.format(value))

#%%
"""Many-to-one LSTM for Sequence Prediction (without TimeDistributed)
Use and LSTM to output the sequence all at once"""
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)  # 1 samples, length time step, 1 feature
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length,1)))
model.add(Dense(length))
model.compile(optimizer="adam", loss="mean_squared_error")
print(model.summary())
# train LSTM
model.fit(X, y, epochs=500, batch_size=1, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('{}'.format(value))

#%%
"""Many-to-Many LSTM for Sequence Prediction (with TimeDistributed)
Use and LSTM to output the sequence all at once
TimeDistributed does two important things:
-Allows the problem to be framed and learned as defined, a one-to-one mapping of output and input
-Simplifies the network, and requires less weigths"""
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)  # 1 samples, length time step, 1 feature
y = seq.reshape(1, length, 1)  # 1 samples, length time step, 1 feature
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True)) # set return sequences to true so it returns a sequence of five outputs, one for each time step in the input data, instead of single output value as in the previous example
model.add(TimeDistributed(Dense(1))) # use a TimeDistributed layer to wrap a fully connected Dense layer with one output
# the single output value in the output layer is important. It highlights that we intend to output one time step from the sequence for each time step in the input. It just so happens we will process 'length' time steps of the input sequence at a time
# the TimeDistributed wrap applies the same dense layer (same w) to the LSTM outputs for one time step at a time
model.compile(loss="mean_squared_error", optimizer="adam")
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
    print('{}'.format(value))