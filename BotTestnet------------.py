import ccxt
import time
import datetime
import sys  
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp
from hyperopt import STATUS_OK
from dotenv import load_dotenv
import os
from hyperopt import Trials
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from keras.regularizers import l2
from itertools import product
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from dotenv import load_dotenv, find_dotenv
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd





def load_binance_api_keys():
    dotenv_path = os.path.join(os.path.dirname(__file__), 'APIsKeyAndEmail.env')
    load_dotenv(dotenv_path)

    api_key = os.environ.get("KEY")
    api_secret = os.environ.get("SECRET")

    print("KEY:", api_key)
    print("SECRET:", api_secret)

    return api_key, api_secret

def initialize_binance(api_key, api_secret):
    binance = Client(api_key, api_secret, testnet=True)
    return binance



def fetch_data(binance):
    klines = binance.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1MINUTE, "1000 minute ago UTC")  ### editable parameters
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['price_change'] = data['close'].astype(float) - data['open'].astype(float)
    data = data[['timestamp', 'close', 'volume', 'price_change']]
    
    data = data.dropna() 
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
    data['close'] = pd.to_numeric(data['volume'], errors='coerce')
    data = data[data['volume'] > 0]
    data = data[data['close'] > 0]

    
    
    scaler = MinMaxScaler()
    data[['close', 'volume', 'price_change']] = scaler.fit_transform(data[['close', 'volume', 'price_change']])
    train, test = train_test_split(data, test_size=0.3, shuffle=False)   ### editable parameters
    train, val = train_test_split(train, test_size=0.5, shuffle=False)    ### editable parameters
    train.to_csv('train_data.csv', index=False, sep=';')
    val.to_csv('val_data.csv', index=False, sep=';')
    test.to_csv('test_data.csv', index=False, sep=';')
    
    return train, val, test



def load_csv_data():
    test_data = pd.read_csv("test_data.csv", delimiter=';', header=0, usecols=[1, 2, 3])
    X_test = test_data.values[:, :-1]
    y_test = test_data.values[:, -1]
    val_data = pd.read_csv("val_data.csv", delimiter=';', header=0, usecols=[1, 2, 3])
    X_val = val_data.values[:, :-1]
    y_val = val_data.values[:, -1]
    train_data = pd.read_csv("train_data.csv", delimiter=';', header=0, usecols=[1, 2, 3])
    X_train = train_data.values[:, :-1]
    y_train = train_data.values[:, -1]
    return X_train, y_train, X_val, y_val, X_test, y_test

def reshape_data(X_train, X_val, X_test):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_val, X_test


####Editable model structure
def create_model(params):
    model = Sequential()
    model.add(LSTM(int(params['units']), input_shape=(1, 2), return_sequences=True))
    model.add(LSTM(int(params['units'])))
    model.add(Dense(1, kernel_regularizer=l2(params['l2'])))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse') ### editable parameters(for instance mea...)
    return model


def objective(params, X_train, y_train, X_val, y_val):
    model = create_model(params)
    history = model.fit(X_train, y_train, batch_size=int(params['batch_size']), epochs=int(params['epochs']), validation_data=(X_val, y_val), verbose=1)
    val_loss = history.history['val_loss'][-1]
    y_pred = model.predict(X_val).flatten()
    corr = np.corrcoef(y_val, y_pred)[0][1]
    print("MSE: {:.5f} | Correlation: {:.5f}".format(val_loss, corr))
    return {'loss': val_loss, 'status': STATUS_OK}


def calculate_return(entry_price, exit_price, position):
    if position == 1: # Long position
        return (exit_price - entry_price) / entry_price
    elif position == -1: # Short position
        return (entry_price - exit_price) / entry_price
    else:
        return 0




def generate_signals(y_pred, threshold, stop_loss, take_profit):
    signals = []
    for i, pred in enumerate(y_pred):
        if i < len(y_pred) - 1:
            if pred > threshold:
                signals.append((1, pred - stop_loss, pred + take_profit))
            elif pred < -threshold:
                signals.append((-1, pred + stop_loss, pred - take_profit))
            else:
                signals.append((0, None, None))
        else:
            signals.append((0, None, None))
    return signals


def trading_objective(params, y_test, y_pred, binance, symbol):
    threshold = params['threshold']
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    signals = generate_signals(y_pred, threshold, stop_loss, take_profit)

    position = 0
    entry_price = None
    cumulative_return = 0

    for i in range(len(y_test)):
        signal = signals[i]
        if signal[0] != position:
            if signal[0] == 1: # Buy
                if position == -1: # Close short position
                    cumulative_return += calculate_return(entry_price, y_test[i], -1)
                entry_price = signal[1]
                position = 1
            elif signal[0] == -1: # Sell
                if position == 1: # Close long position
                    cumulative_return += calculate_return(entry_price, y_test[i], 1)
                entry_price = signal[2]
                position = -1

    # Close any remaining position at the end
    if position == 1:
        cumulative_return += calculate_return(entry_price, y_test[-1], 1)
    elif position == -1:
        cumulative_return += calculate_return(entry_price, y_test[-1], -1)

    return {'loss': -cumulative_return, 'status': STATUS_OK}


def place_order(binance, symbol, side, quantity):
    try:
        order = binance.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity)
        return True
    except BinanceAPIException as e:
        print(f"Binance API Exception: {e}")
        return False
    except BinanceOrderException as e:
        print(f"Binance Order Exception: {e}")
        return False

def execute_trading_strategy(y_test, y_pred, threshold, stop_loss, take_profit, binance, symbol):
    signals = generate_signals(y_pred, threshold, stop_loss, take_profit)
    balance = float(binance.get_asset_balance(asset='USDT')['free'])
    position = 0  
    entry_price = 0
    trade_data = []
    trade_results = []  
    quantity = 0.1  ### Quantity of asset to be exchanged (customise to suit your needs)

    trade_count = 0  
    i = 0  

    while trade_count < 100 and i < len(signals):     ### number of trades to make   ### editable parameters
        signal, stop, profit = signals[i]

        if signal == 1 and position == 0 and balance >= y_test[i] * (1 + threshold):
            position += 1
            entry_price = y_test[i] * (1 + threshold)
            balance -= entry_price
            trade_data.append(("buy", i, entry_price, "buy"))
            if place_order(binance, symbol, Client.SIDE_BUY, quantity):
                print(f"Achat au prix {entry_price}")
            trade_count += 1

        elif signal == -1 and position > 0:
            position -= 1
            exit_price = y_test[i] * (1 - threshold)
            balance += exit_price
            trade_data.append(("sell", i, exit_price, "sell"))
            if place_order(binance, symbol, Client.SIDE_SELL, quantity):
                print(f"Vente au prix {exit_price}")
            trade_count += 1

        elif position > 0 and (y_test[i] <= entry_price * (1 - stop_loss) or y_test[i] >= entry_price * (1 + take_profit)):
            position -= 1
            exit_price = y_test[i]
            trade_result = calculate_return(entry_price, exit_price, 1)  # Calculate the trade result
            result_string = "stop_loss" if y_test[i] <= entry_price * (1 - stop_loss) else "take_profit"
            balance += exit_price
            trade_data.append(("sell", i, exit_price, result_string))
            trade_results.append((entry_price, exit_price, trade_result))
            if place_order(binance, symbol, Client.SIDE_SELL, quantity):
                print(f"Vente au prix {exit_price} ({result_string})")
            trade_count += 1

        i += 1

    if position > 0:
        balance += y_test[-1] * position
        trade_data.append(("sell", len(y_test) - 1, y_test[-1], "sell"))
        trade_result = calculate_return(entry_price, y_test[-1], 1)
        trade_results.append((entry_price, y_test[-1], trade_result))


    for i, result in enumerate(trade_results):
        print(f"Trade {i+1}: Entrée à {result[0]}, sortie à {result[1]}, résultat de {result[2]:.2%}")

    print(f"Solde final: {balance}")
    return balance





def plot_trades(y_test, trade_data):
    plt.figure(figsize=(20, 10))
    plt.plot(y_test, label="price")
    for trade in trade_data:
        trade_type, index, price, reason = trade
        if trade_type == "buy":
            plt.scatter(index, price, c="green", label="buy" if index == 0 else None)
        elif trade_type == "sell":
            plt.scatter(index, price, c="red", label="sell" if index == 0 else None)
    plt.legend()

def send_email(subject, body, mse, corr, best_hyperparams, to_email, from_email, password):
    body = f"{body}\nMean squared error: {mse}\nCorrelation: {corr}\nBest hyperparameters: {best_hyperparams}"
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print("Error sending the e-mail :", e)
        return False

def main():
    while True:
        api_key, api_secret = load_binance_api_keys()
        binance = initialize_binance(api_key, api_secret)
        train, val, test = fetch_data(binance)
        X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data()
        X_train, X_val, X_test = reshape_data(X_train, X_val, X_test)
        optimizer_choices = ['adam']
        param_space = {
            'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01),   ### editable parameters
            'batch_size': hp.quniform('batch_size', 32, 512, 32),         ### editable parameters
            'epochs': hp.quniform('epochs', 10, 100, 10),                 ### editable parameters
            'l2': hp.loguniform('l2', -10, -4),                           ### editable parameters
            'optimizer': hp.choice('optimizer', optimizer_choices),       ### editable parameters
            'units': hp.quniform('units', 32, 512, 32),                   ### editable parameters
            'unit': hp.quniform('unit', 32, 512, 32),                     ### editable parameters
            'dropout': hp.uniform('dropout', 0, 0.5)}                     ### editable parameters
    
        trials = Trials()
        best = fmin(lambda p: objective(p, X_train, y_train, X_val, y_val), param_space, algo=tpe.suggest, max_evals=10, trials=trials)
        best['optimizer'] = optimizer_choices[best['optimizer']]
        print("Best hyperparameters:", best)

        model = create_model(best)
        history = model.fit(X_train, y_train, batch_size=int(best['batch_size']), epochs=int(best['epochs']), validation_data=(X_val, y_val))

        y_pred = model.predict(X_test)
        corr = np.corrcoef(y_test, y_pred.flatten())[0][1]
        print("Final model correlation: ", corr)

        mse = mean_squared_error(y_test, y_pred)
        print("Mean squared error:", mse)

        plt.plot(y_test, label='Données de test')
        plt.plot(y_pred, label='Prédictions')
        plt.legend()

        now2 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
        # Créer le répertoire "ResultGraph" s'il n'existe pas déjà
        graph_dir = "ResultGraph"
        os.makedirs(graph_dir, exist_ok=True)
    
        # Construire le chemin relatif pour enregistrer le graphique
        graph_filename = "result_graph_pred_{}.png".format(now2)
        graph_filepath = os.path.join(graph_dir, graph_filename)
    
        # Enregistrer le graphique
        plt.savefig(graph_filepath)

# Recherche des meilleurs paramètres de trading
        trading_param_space = {
            'threshold': hp.uniform('threshold', 0, 0.05),          ### editable parameters
            'stop_loss': hp.uniform('stop_loss', 0, 0.01),          ### editable parameters
            'take_profit': hp.uniform('take_profit', 0, 0.01)}      ### editable parameters
        symbol='ETHUSDT'
        trading_trials = Trials()
        trading_best = fmin(lambda p: trading_objective(p, y_test, y_pred, binance, symbol), trading_param_space, algo=tpe.suggest, max_evals=200, trials=trading_trials, verbose=1)
        print("Best trading parameters : ", trading_best)

        solde_final = execute_trading_strategy(y_test, y_pred.flatten(), trading_best['threshold'], trading_best['stop_loss'], trading_best['take_profit'], binance, "ETHUSDT")

        subject = "Model performance report"
        body = "Final balance: {:.2f}".format(solde_final)
        to_email = os.environ.get("TOEMAIL")
        from_email = os.environ.get("FROMEMAIL")
        password = os.environ.get("EMAILPASSWORD") 
        if send_email(subject, body, mse, corr, best, to_email, from_email, password):
            print("E-mail sent with success")
        else:
            print("E-mail failed to be sent")

        answer = input("Do you want to restart the program ? (y/n) ")
        if answer.lower() != "y":
            break
if __name__ == '__main__':
    main()