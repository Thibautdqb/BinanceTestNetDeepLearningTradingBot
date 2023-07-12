#### graphique de reslutats et données 

# print des dataframes
# données de correlations
# repartitions de erreurs 
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



def show_results():
    st.title("DATA and RESULTS")
    st.write("Contenu de la page 2")
    
    train, val, test = fetch_data()
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
                         

    # Interaction avec la page 1
    if st.button("Afficher la page 1"):
        from HYPERPARAMETERS import show_hyperparameters
        show_hyperparameters()



            