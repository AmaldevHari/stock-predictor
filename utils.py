#The model used here is a sequence or regression model LSTM RNN (Long Short Term Memory Recuurent Neural Network)
# this model can be used to predict stock prices
import time
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data
import logging

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}

plt.rcParams.update(params)


# get stock price information
def retreive_data(filename= None, stock_name= None, max_attempt=3):
    #retrives stcok data given a @stock_name
    #this function will try maximum of @max_attempts if preceeding
    #attempts fail
    if stock_name:
        for attempt in range(max_attempt):
            try:
                df = pdr.get_data_yahoo(stock_name) #get the stock from Yahoo
                # Only get the adjusted close.

                df = df.reindex(index=pd.date_range(df.index.min(),
                                                        df.index.max(),
                                                        freq='D')).fillna(method='ffill')
                # we need time series to have values spaced every sampling time (tau)
                # stock market data is not evenly spaced therefore reindex them to have
                # a price at every day (sampling rate is 1/day). The filling method is ffill
                # this fills the places without data with the data from the previous sample
                return df

            except Exception as e:
                print(f"fatal error: retrying...")

    elif filename:
        for attempt in range(max_attempt):
            try:
                df = pd.read_csv(filename) #get the stock from Yahoo
                # Only get the adjusted close.

                #df = df.reindex(index=pd.date_range(df["Date"][0],df["Date"][-1],freq='D')).fillna(method='ffill')
                # we need time series to have values spaced every sampling time (tau)
                # stock market data is not evenly spaced therefore reindex them to have
                # a price at every day (sampling rate is 1/day). The filling method is ffill
                # this fills the places without data with the data from the previous sample
                return df

            except Exception as e:
                print(f"fatal error: retrying...")

    else:
        print("Fatal error: stock name not specified")
    return None



def prepare_dataset(time_series, scale=True, split_ratio=0.9):
    #returns the dataset (train, test set) given the full timeseries data
    # This is done by first Normalizing the data (specified by scale flag)
    #follwoed by splitting the time_series into train and test sets
    scaler = None

    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1)) #normalize
        time_series = np.array(time_series).reshape(-1, 1)# make timeseries column vector
        time_series = scaler.fit_transform(time_series)

    train_size = int(len(time_series) * split_ratio)

    train = time_series[0:train_size]
    test = time_series[train_size:len(time_series)]

    return train, test, scaler


# Get stacked LSTM model for regression modeling
def get_model(type= 'seq', layer_units=[100, 100], dropouts=[0.2, 0.2], window_size=50, activation='linear',\
              verbose= True, hidden_units=4, input_shape=(1, 1), loss= "mse", optimizer= "rmsprop"):
    if type== 'reg':
        # build the regression model
        # inputs are layer_units (the output size of each hidden layer)
        # the dropout ratio
        # the sizes of layer_units and dropouts must be same
        assert len(layer_units) == len(dropouts), "Fatal error: size of layer_units does not match size of dropouts"
        model = Sequential()  # sequential container for hidden layers
        N = len(layer_units)

        for i in range(N):
            model.add(LSTM(layer_units[i],
                           input_shape=(window_size, 1),
                           return_sequences=True if i == 0 else False))
            model.add(Dropout(dropouts[i]))

        model.add(Dense(1))
        model.add(Activation(activation))

        start = time.time()
        model.compile(loss= loss, optimizer= optimizer)
        if verbose:
            logging.info(f" Compilation Time :  {time.time() - start}")
            logging.info(model.summary())

        return model

    elif type== 'seq':

        #build sequence model
        model = Sequential() #sequential container
        model.add(LSTM(input_shape=input_shape,
                       units=hidden_units,
                       return_sequences=True
                       ))
        model.add(TimeDistributed(Dense(1)))
        start = time.time()
        model.compile(loss= loss, optimizer= optimizer)

        if verbose:
            logging.info(f" Compilation Time :  {time.time() - start}")
            logging.info(model.summary())

        return model






# Window wise prediction function
def predict_reg_multiple(model, data, window_size=6, prediction_len=3):
    prediction_list = []

    # loop for every sequence in the dataset
    for window in range(int(len(data) / prediction_len)):
        _seq = data[window * prediction_len]
        predicted = []
        # loop till required prediction length is achieved
        for j in range(prediction_len):
            predicted.append(model.predict(_seq[np.newaxis, :, :])[0, 0])
            _seq = _seq[1:]
            _seq = np.insert(_seq, [window_size - 1], predicted[-1], axis=0)
        prediction_list.append(predicted)
    return prediction_list


# Plot window wise
def plot_reg_results(predicted_data, true_data, prediction_len=3):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)

    # plot actual data
    ax.plot(true_data,
            label='True Data',
            c='black', alpha=0.3)

    # plot flattened data
    plt.plot(np.array(predicted_data).flatten(),
             label='Prediction_full',
             c='g', linestyle='--')

    # plot each window in the prediction list
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction', c='black')

    plt.title("Forecast Plot with Prediction Window={}".format(prediction_len))
    plt.show()
