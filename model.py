import pandas as pd
import numpy as np
import random
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Activation
from keras.layers import LSTM
from sklearn.utils import shuffle
import tensorflow as tf
from keras import losses
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to Convert Data For model
def inversetrans(pdt):
    ok = []
    for i in pdt:
        ok.append(i*(3169780000000 - 19717445559) + 19717445559)
    return ok



def series_to_supervised(data1, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data1) is list else data1.shape[1]
    df = DataFrame(data1)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Converting the Data for Model


def forecast(data1):
    random.seed(42)
    values = data1.values
    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(
        reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    values = reframed.values

    n_train = 82
    train = values[:n_train]
    test = values[n_train:]
    Xtrain, Ytrain = train[:, :-1], train[:, -1]
    Xtest, Ytest = test[:, :-1], test[:, -1]

    Xtrain = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
    Xtest = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))

    model = Sequential()

    model.add(LSTM(units=50, activation='tanh', input_shape=(Xtrain.shape[1],Xtrain.shape[2]), return_sequences=True, kernel_initializer='normal',kernel_regularizer='l2'))
    model.add(Dropout(0.4))

    # adding a first LSTM layer and some dropout regularisation
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(Dropout(0.4))

    # adding a second LSTM layer and some dropout regularisation
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(Dropout(0.4))

    # adding a third LSTM layer and some dropout regularisation
    model.add(LSTM(units=50, activation='tanh', return_sequences=True))
    model.add(Dropout(0.4))

    # adding a fourth LSTM layer and some dropout regularisation
    model.add(LSTM(units=50, activation='tanh',))
    model.add(Dropout(0.4))

    # adding output layer
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy'])

    # Fitting the Rnn ti the testing set
    model.fit(Xtrain, Ytrain, epochs=100,
              batch_size=20, verbose=0, shuffle=False)

    # Forecasting

    predicted = model.predict(Xtest)
    XtestRe = Xtest.reshape(Xtest.shape[0], Xtest.shape[2])
    predicted = np.concatenate((predicted, XtestRe[:, 1:]), axis=1)
    predicted = inversetrans(predicted)
    latest = predicted[0]
    return latest


if __name__ == "__main__":
    df = pd.read_csv('Oil and gas for models.csv')
    data1 = df.iloc[:, 1:]
    print("Here >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(forecast(data1))
    # print(inversetrans([0,0.5,1]))
