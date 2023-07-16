####streamlit one page
import streamlit as st
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
from sklearn.metrics import mean_absolute_error, r2_score



# Définir le thème personnalisé
# Vous pouvez personnaliser les couleurs en fonction de vos préférences


# Appliquer les options de personnalisation
st.set_page_config(
    page_title="Python DP TradingBot",
    page_icon=":📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_binance_api_keys():
    #dotenv_path = os.path.join(os.path.dirname(__file__), 'APIsKeyAndEmail.env')
    #load_dotenv(dotenv_path)
    api_key = ("QrRxR5wAtOzsAhzfC3RqSjrzZwwTxuuv8ls1E5kT0EsYANfqpClImv0qibwd1w8R") #os.environ.get("KEY")
    api_secret = ("7jIgk6Pwp4WCGANWTi4hsThKjMITHKi1hLarlEssXs39oXhbf8T6P1XTVoWGeI0I") #os.environ.get("SECRET")   
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
    quantity = 0.1 ### Percentage of the total balance to be exchanged (customise to suit your needs)

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
                print(f"Buying at price {entry_price}")
            trade_count += 1

        elif signal == -1 and position > 0:
            position -= 1
            exit_price = y_test[i] * (1 - threshold)
            balance += exit_price
            trade_data.append(("sell", i, exit_price, "sell"))
            if place_order(binance, symbol, Client.SIDE_SELL, quantity):
                print(f"Selling at price {exit_price}")
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
                print(f"Selling at price {exit_price} ({result_string})")
            trade_count += 1

        i += 1

    if position > 0:
        balance += y_test[-1] * position
        trade_data.append(("sell", len(y_test) - 1, y_test[-1], "sell"))
        trade_result = calculate_return(entry_price, y_test[-1], 1)
        trade_results.append((entry_price, y_test[-1], trade_result))


    for i, result in enumerate(trade_results):
        print(f"Trade {i+1}: Entry at {result[0]}, Exit at {result[1]}, Result of {result[2]:.2%}")

    print(f"Final balance: {balance}")
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




def validate_email(email):
    # Vérifie si l'adresse e-mail est valide
    # Vous pouvez personnaliser cette fonction en fonction de vos critères de validation
    if "@" in email:
        return True
    return False







def main():
    import streamlit as st
    st.title("Deep learning Tradingbot")
    
    st.header('Python program that executes ETH buy and sell orders on the Binance test platform using signals generated by a Deep Learning program.')
    
    st.write('''Define the hyperparameter windows you wish to use to launch the program and click on the "Start the Magic" button''')
    
    api_key, api_secret = load_binance_api_keys()
    binance = initialize_binance(api_key, api_secret)
    st.header('Email Formular')
    
    email_streamlit_key = st.empty()
    email_streamlit = email_streamlit_key.text_input("Entrez votre adresse e-mail", key="email_input")
    if st.button("Valider"):
        if validate_email(email_streamlit):
            st.success("Adresse e-mail valide !")
        else:
            st.error("Adresse e-mail invalide. Veuillez réessayer.")
    st.header('Models Hyperparameters Search')
    
    col_slider_1, col_slider_2, col_slider_3, col_slider_4 = st.columns(4)





    with col_slider_1 :
        # Définir les valeurs minimale et maximale du slider
        valeur_min_learning_rate= 0.001
        valeur_max_learning_rate = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_learning_rate = st.slider("Learning rate", valeur_min_learning_rate, valeur_max_learning_rate, (valeur_min_learning_rate, valeur_max_learning_rate), step=0.001, key="2")
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_learning_rate = valeurs_learning_rate[0]
        new_valeur_max_learning_rate = valeurs_learning_rate[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_learning_rate)
        st.write(new_valeur_max_learning_rate)
        st.write(" "*10)

    with col_slider_2 : 
        valeur_min_batch_size= 32
        valeur_max_batch_size = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_batch_size = st.slider("Batch size", valeur_min_batch_size, valeur_max_batch_size, (valeur_min_batch_size, valeur_max_batch_size), step=1, key=3)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_batch_size = valeurs_batch_size[0]
        new_valeur_max_batch_size = valeurs_batch_size[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_batch_size)
        st.write(new_valeur_max_batch_size)
        st.write(" "*10)

    with col_slider_3 :
        valeur_min_epochs= 10
        valeur_max_epochs = 100
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_epochs = st.slider("Epochs", valeur_min_epochs, valeur_max_epochs, (valeur_min_epochs, valeur_max_epochs), step=1, key=4)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_epochs = valeurs_epochs[0]
        new_valeur_max_epochs = valeurs_epochs[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_epochs)
        st.write(new_valeur_max_epochs)
        st.write(" "*10)

    with col_slider_4 :
        # Définir les valeurs minimale et maximale du slider
        valeur_min_l2= -10
        valeur_max_l2 = -4
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_l2 = st.slider("l2", valeur_min_l2, valeur_max_l2, (valeur_min_l2, valeur_max_l2), step=1, key=5)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_l2 = valeurs_l2[0]
        new_valeur_max_l2 = valeurs_l2[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_l2)
        st.write(new_valeur_max_l2)
    col_selectbox_5, col_slider_6, col_slider_7, col_slider_8 = st.columns(4)
    with col_selectbox_5 :
        optimizer = st.selectbox(
            'Optimizer',
            ('Adam','SGD', 'Adamax', 'Adagrad'))
        st.write(optimizer)
        #optimizer = st.selectbox('Optimizer', 'adam', 'SGD')
        #st.write (optimizer)
    with col_slider_6 : 
        # Définir les valeurs minimale et maximale du slider
        valeur_min_units= 32
        valeur_max_units = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_units = st.slider("Units", valeur_min_units, valeur_max_units, (valeur_min_units, valeur_max_units), step=1, key=6)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_units = valeurs_units[0]
        new_valeur_max_units = valeurs_units[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_units)
        st.write(new_valeur_max_units)
    with col_slider_7 : 
        # Définir les valeurs minimale et maximale du slider
        valeur_min_unit= 32
        valeur_max_unit = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_unit = st.slider("Unit", valeur_min_unit, valeur_max_unit, (valeur_min_unit, valeur_max_unit), step=1, key=7)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_unit = valeurs_unit[0]
        new_valeur_max_unit = valeurs_unit[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_unit)
        st.write(new_valeur_max_unit)
    with col_slider_8: 
            # Définir les valeurs minimale et maximale du slider
        valeur_min_dropout= 0.001
        valeur_max_dropout = 0.5
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_dropout = st.slider("Dropout", valeur_min_dropout, valeur_max_dropout, (valeur_min_dropout, valeur_max_dropout), step=0.1, key=8 )
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_dropout = valeurs_dropout[0]
        new_valeur_max_dropout = valeurs_dropout[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_dropout)
        st.write(new_valeur_max_dropout)
    
    st.header('Trading Parameters')
    col_thresold, col_stop_loss, col_take_profit = st.columns(3)
    with col_thresold:
        valeur_min_thresold= 32
        valeur_max_thresold = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_thresold = st.slider("Thresold value", valeur_min_thresold, valeur_max_thresold, (valeur_min_thresold, valeur_max_thresold), step=1, key=9)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_thresold = valeurs_thresold[0]
        new_valeur_max_thresold = valeurs_thresold[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_thresold)
        st.write(new_valeur_max_thresold)
    with col_stop_loss:
        valeur_min_stop_loss= 32
        valeur_max_stop_loss = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_stop_loss = st.slider("Stop Loss Value", valeur_min_stop_loss, valeur_max_stop_loss, (valeur_min_stop_loss, valeur_max_stop_loss), step=1, key=10)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_stop_loss = valeurs_stop_loss[0]
        new_valeur_max_stop_loss = valeurs_stop_loss[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_stop_loss)
        st.write(new_valeur_max_stop_loss)
    with col_take_profit:
        valeur_min_take_profit= 32
        valeur_max_take_profit = 512
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_take_profit = st.slider("Take profit value", valeur_min_take_profit, valeur_max_take_profit, (valeur_min_take_profit, valeur_max_take_profit), step=1, key=11)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_take_profit = valeurs_take_profit[0]
        new_valeur_max_take_profit = valeurs_take_profit[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_take_profit)
        st.write(new_valeur_max_take_profit)



    st_param_model = st.button ('Start Magic')
    if st_param_model :
        st.write("Le bouton a été cliqué !")
        train, val, test = fetch_data(binance)
        # Créer deux colonnes pour afficher les ensembles de données
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("Ensemble de formation"):
                st.dataframe(train.style.set_properties(**{'background-color': 'lightblue', 'color': 'black'}))
        with col2:
            with st.expander("Ensemble de validation"):
                st.dataframe(val.style.set_properties(**{'background-color': 'lightgreen', 'color': 'black'}))
        with col3:
            with st.expander("Ensemble de test"):
                st.dataframe(test.style.set_properties(**{'background-color': 'lightyellow', 'color': 'black'}))
        X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data()
        X_train, X_val, X_test = reshape_data(X_train, X_val, X_test)
        st.title("Recherche du meilleur model .......")
        param_space = {
            'learning_rate': hp.uniform('learning_rate', new_valeur_min_learning_rate, new_valeur_max_learning_rate),
            'batch_size': hp.uniform('batch_size',new_valeur_min_batch_size, new_valeur_max_batch_size),
            'epochs': hp.uniform('epochs',new_valeur_min_epochs, new_valeur_max_epochs),
            'l2': hp.loguniform('l2', -10, 0),
            'optimizer': hp.choice('optimizer', optimizer),            
            'units': hp.uniform('units', new_valeur_min_units, new_valeur_max_units),
            'unit': hp.uniform('unit', new_valeur_min_unit, new_valeur_max_unit),
            'dropout': hp.uniform('dropout', new_valeur_min_dropout, new_valeur_max_dropout),
        }
        trials = Trials()
        max_evals = 5
        for i in range(1, max_evals + 1):
            best = fmin(
                fn=lambda p: objective(p, X_train, y_train, X_val, y_val),
                space=param_space,
                algo=tpe.suggest,
                max_evals=i,
                trials=trials,
                verbose = 1,
        )
            progress_bar = st.progress(i / max_evals)
            st.text(f"Iteration{i}/{max_evals}")
        optimisation_complete = True

        if optimisation_complete: 
            best['optimizer'] = optimizer[best['optimizer']]
            st.title("Utilisation du meilleurs model trouvé1")
            print("Best hyperparameters:", best)
            model = create_model(best)
            history = model.fit(X_train, y_train, batch_size=int(best['batch_size']), epochs=int(best['epochs']), validation_data=(X_val, y_val))
            y_pred = model.predict(X_test)
            st.title("Utilisation du meilleurs model 2")

            corr = np.corrcoef(y_test, y_pred.flatten())[0][1]
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            errors = np.abs(y_test - y_pred)
            st.title("Utilisation du meilleurs model 3")

            trading_param_space = {
                        'threshold': hp.uniform('threshold', new_valeur_min_thresold, new_valeur_max_thresold),         
                        'stop_loss': hp.uniform('stop_loss', new_valeur_min_stop_loss, new_valeur_max_stop_loss),      
                        'take_profit': hp.uniform('take_profit', new_valeur_min_take_profit, new_valeur_max_take_profit)}  
            symbol='ETHUSDT'
            trading_trials = Trials()
            st.title("Utilisation du meilleurs model 4")

            max_eval = 200
            for i in range(1, max_eval + 1):
                trading_best = fmin(
                    fn=lambda p: trading_objective(p, y_test, y_pred, binance, symbol),
                    space=trading_param_space,
                    algo=tpe.suggest,
                    max_eval=i,
                    trials=trading_trials,
                    verbose = 1,
        )   
                progress_bar = st.progress(i / max_eval)
                st.text(f"Iteration{i}/{max_eval}")
            optimisation_trading_complete = True
            st.title("Utilisation du meilleurs model 5")


            if optimisation_trading_complete : 
                solde_final = execute_trading_strategy(y_test, y_pred.flatten(), trading_best['threshold'], trading_best['stop_loss'], trading_best['take_profit'], binance, "ETHUSDT")
                subject = "Model performance report"
                body = "Final balance: {:.2f}".format(solde_final)
                to_email = email_streamlit
                from_email = os.environ.get("FROMEMAIL")
                password = os.environ.get("EMAILPASSWORD") 
                if send_email(subject, body, mse, corr, best, to_email, from_email, password):
                    st.success("E-mail sent with success")
                else:
                    st.error("E-mail failed to be sent")

if __name__ == '__main__':
    main()



        ### Histogramme de Répartition des erreurs
        ##st.subheader("Histogramme de Répartition des erreurs")
        ##fig_hist = plt.figure()
        ##plt.hist(errors, bins=50)
        ##plt.xlabel('Erreur')
        ##plt.ylabel("Nombre d'occurrences")
        ##plt.title('Histogramme de Répartition des erreurs')
        ##st.pyplot(fig_hist)
        ### Graphique des prédictions
        ##st.subheader("Graphique des Prédictions")
        ##fig_pred = plt.figure()
        ##plt.plot(y_test, label='Données de test')
        ##plt.plot(y_pred, label='Prédictions')
        ##plt.title('Graphique des Prédictions')
        ##plt.legend()
        ##st.pyplot(fig_pred)