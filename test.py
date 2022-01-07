# https://towardsdatascience.com/cryptocurrency-price-prediction-using-deep-learning-70cfca50dd3a

import tensorflow
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [CAD]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);


def normalise_zero_base(df):
    return df / df.iloc[0] - 1


# def normalise_min_max(df):
#     return (df - df.min()) / (data.max() - df.min())

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    start = time.time()
    target_col = 'close'
    np.random.seed(42)
    window_len = 5
    test_size = 0.2
    zero_base = True
    lstm_neurons = 100
    epochs = 10
    batch_size = 32
    loss = 'mse'
    dropout = 0.2
    optimizer = 'adam'


    def split_df(df):
        # if len(df) % 2 != 0:  # Handling `df` with `odd` number of rows
        #     df = df.iloc[:-1, :]
        df1, df2 = np.array_split(df, 2)
        return df1, df2


    # train1_splitted, train2_splitted = split_df(train_data_set)
    # test1_splitted, test2_splitted = split_df(test_data_set)
    #
    # train1,train2 = split_df(train1_splitted)
    # train3,train4 = split_df(train2_splitted)
    #
    # test1,test2 = split_df(test1_splitted)
    # test3,test4 = split_df(test2_splitted)
    #
    # hist1_splitted, hist2_splitted = split_df(hist)
    # hist1, hist2 = split_df(hist1_splitted)
    # hist3, hist4 = split_df(hist2_splitted)
    #
    # print("hist size")
    # print(hist.size)


    if rank == 0:
        endpoint = 'https://min-api.cryptocompare.com/data/histoday'
        res = requests.get(endpoint + '?fsym=BTC&tsym=CAD&limit=2000')
        hist = pd.DataFrame(json.loads(res.content)['Data'] * 10)
        # print(hist.size)
        hist = hist.set_index('time')
        hist.index = pd.to_datetime(hist.index, unit='s')

        hist.drop(["conversionType", "conversionSymbol"], axis='columns', inplace=True)
        hist.head(5)

        train_data_set, test_data_set = train_test_split(hist, test_size=0.2)

        train1,train2,train3,train4 = np.array_split(train_data_set,4)
        test1,test2,test3,test4 = np.array_split(test_data_set,4)
        hist1,hist2,hist3,hist4 = np.array_split(hist,4)

        comm.send(train2,dest=1)
        comm.send(test2,dest=1)
        comm.send(hist2,dest=1)

        comm.send(train3, dest=2)
        comm.send(test3, dest=2)
        comm.send(hist3, dest=2)

        comm.send(train4, dest=3)
        comm.send(test4, dest=3)
        comm.send(hist4, dest=3)

        train = train1
        test = test1
        hist = hist1

    elif rank == 1:
        train = comm.recv(source=0)
        test = comm.recv(source=0)
        hist = comm.recv(source=0)

        # print("Rank:")
        # print(rank)
        # print(train2)
        # print("A")
        # print(test2)
    elif rank == 2:
        train = comm.recv(source=0)
        test = comm.recv(source=0)
        hist = comm.recv(source=0)

    elif rank == 3:
        train = comm.recv(source=0)
        test = comm.recv(source=0)
        hist = comm.recv(source=0)

    # a = 0
    # for i in range(1000):
    #     a += (i ** 100)
    # end = time.time()
    # # print("The time of execution of above program is :", end - start)
    # print(f'Time to split and execute data of {rank} is : {end - start}')

    line_plot(train[target_col], test[target_col], 'training', 'test', title='')

    # train, test, X_train, X_test, y_train, y_test = prepare_data(
    #     hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)


    #print(f'rank {rank} hist size: {hist.size}')
    train, test, X_train, X_test, y_train, y_test = prepare_data(
        hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1,
        shuffle=True,use_multiprocessing=True,workers=4)



    # plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
    # plt.plot(history.history['val_loss'], 'g', linewidth=2, label='Validation loss')
    # plt.title('LSTM')
    # plt.xlabel('Epochs')
    # plt.ylabel('MSE')
    # plt.show()

    targets = test[target_col][window_len:]
    preds = model.predict(X_test).squeeze()
    #print(mean_absolute_error(preds, y_test))


    MAE = mean_squared_error(preds, y_test)
    #print(MAE)


    R2 = r2_score(y_test, preds)
    print(f'Rank:{rank} - R2:{R2}')

    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=3)

    # if rank == 1:
    #     comm.send(R2,dest=0)
    #
    # if rank == 0:
    #     R2FromRank1 = comm.recv(source=1)
    #     print(f'R2 from rank {rank} = {R2}')
    #     print(f'R2 from rank 1 = {R2FromRank1}')

    a = 0
    for i in range(1000):
        a += (i ** 100)
    end = time.time()
    # print("The time of execution of above program is :", end - start)
    print(f'Time of execution of process {rank} is : {end - start}')
