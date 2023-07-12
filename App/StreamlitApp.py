import streamlit as st
from HYPERPARAMETERS import show_hyperparameters
from RESULTS import show_results

def main():
    st.sidebar.title("Menu")
    app_page = st.sidebar.radio("Sélectionnez une page", ("Page 1", "Page 2"))

    if app_page == "Page 1":
        show_hyperparameters()
    elif app_page == "Page 2":
        show_results()

if __name__ == "__main__":
    main()