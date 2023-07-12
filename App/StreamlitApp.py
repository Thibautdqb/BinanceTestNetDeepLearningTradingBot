# main.py
import streamlit as st
from HYPERPARAMETERS import show_hyperparameters
from RESULTS import show_results

def main():
    st.sidebar.title("Deep learning Tradingbot")
    st.sidebar.header('Python program that executes ETH buy and sell orders on the Binance test platform using signals generated by a Deep Learning program.')
    st.sidebar.write('''Define the hyperparameter windows you wish to use to launch the program and click on the "Start the Magic" button''')

    app_page = st.sidebar.radio("Sélectionnez une page", ("HYPERPARAMETERS", "RESULTS"))

    if app_page == "HYPERPARAMETERS":
        show_hyperparameters()
    elif app_page == "RESULTS":
        show_results()

if __name__ == "__main__":
    main()