import math
import sys
import warnings
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
sns.set_style('whitegrid')
sns.set_context('talk')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}

plt.rcParams.update(params)

# specify to ignore warning messages
warnings.filterwarnings("ignore")

from utils import retreive_data
from utils import get_model
from utils import prepare_dataset
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import mean_squared_error

def get_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--stock", type= str, default= None, help= "stock data to use")
    parser.add_argument("--filename", type= str, default= "btc.csv", help= "filename with data")
    parser.add_argument("--verbose", action= "store_true", default= False, help="verbose flag")
    parser.add_argument("--split_ratio", type= float, default= 0.7, help= "split raio for train-test set")
    parser.add_argument("--category", type= str, default= "Close", help= "column in the stock price data \
                                                                         fo which prediction is made")
    parser.add_argument("--epoch", type= int, default= 200, help= "number of epcohs to train the model")
    parser.add_argument("--optimizer", type= str, default= "adam", help =" optimizer for loss func")
    parser.add_argument("--loss", type= str, default= "mse", help= "loss func")

    args= parser.parse_args()
    return args

if __name__ == '__main__':

    args= get_args()

    # load data
    stock_data = retreive_data(args.stock if args.stock else args.filename)
    stock_data= stock_data[args.category if args.stock else 'Closing Price (USD)']
    print(stock_data)
    logging.info("data succesfully retreived")

    # split train and test datasets
    train, test, scaler = prepare_dataset(stock_data,
                                             scale= True,
                                             split_ratio= args.split_ratio)

    train = np.reshape(train, (1, train.shape[0], 1))
    test = np.reshape(test, (1, test.shape[0], 1))

    train_in = train[:, :-1, :] #the previous data (shifted by 1 day)
    train_out = train[:, 1:, :] # the data to be predicted using previous data

    test_in = test[:, :-1, :]
    test_out = test[:, 1:, :]
    logging.info(f"train input shape: {train_in.shape}")
    logging.info(f"train output shape: {train_out.shape}")
    logging.info(f"test input shape: {test_in.shape}")
    logging.info(f"test output shape: {test_out.shape}")

    # build RNN model
    model = None
    try:
        model = get_model(type='seq',input_shape=(train_in.shape[1], 1),
                                       verbose= args.verbose, optimizer= args.optimizer, loss= args.loss)
    except Exception as e:
        logging.info("Model initialization failed, retrying....")
        model = get_model(input_shape=(train_in.shape[1], 1),
                                       verbose= args.verbose, optimizer= args.optimizer, loss= args.loss)

    if model is None:
        logging.error("fatal error: model initialization failed, exiting...")
        sys.exit(-1)

    # train the model
    model.fit(train_in, train_out,
                       epochs= args.epoch, batch_size=1,
                       verbose=2)
    logging.info("training completed... proceeding to validation")

    pred = model.predict(train_in)
    res = math.sqrt(mean_squared_error(train_out[0], pred[0]))
    logging.info('RMS error (loss): %.2f RMSE' % (res))

    #testing, requries padding because the model is made to expect input of size same as training set
    test_in = pad_sequences(test_in,
                                maxlen=train_in.shape[1],
                                padding='post',
                                dtype='float64')

    # forecast values
    val_pred = model.predict(test_in)

    # evaluate performances
    val_error = math.sqrt(mean_squared_error(test_out[0],
                                             val_pred[0][:test_out.shape[1]]))

    # inverse transformation
    pred = scaler.inverse_transform(pred.reshape(pred.shape[1],1))
    val_pred = scaler.inverse_transform(val_pred.reshape(val_pred.shape[1],1))
    model.save(f"seq_model_{args.optimizer}.h5")

    # plot the true and forecasted values
    train_size = len(pred) + 1

    plt.plot(stock_data.index,
             stock_data.values, c='black',
             alpha=0.3, label='Original Data')
    plt.plot(stock_data.index[1:train_size],
             pred, label='Training Fit', c='g')
    plt.plot(stock_data.index[train_size + 1:],
             val_pred[:test_out.shape[1]], label='Forecasted Values')
    plt.title('Results')
    plt.legend()
    plt.show()
