# main.py
import streamlit as st
from HYPERPARAMETERS import show_hyperparameters
from RESULTS import show_results

def main():
    st.sidebar.title("Menu")
    app_page = st.sidebar.radio("Sélectionnez une page", ("HYPERPARAMETERS", "RESULTS"))

    if app_page == "HYPERPARAMETERS":
        show_hyperparameters()
    elif app_page == "RESULTS":
        show_results()

if __name__ == "__main__":
    main()