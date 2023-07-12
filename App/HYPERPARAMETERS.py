
import streamlit as st
import streamlit as st


###parameters
def show_hyperparameters():
    st.title("HYPERPARAMETERS")
        
    st.write('''Define the hyperparameter windows you wish to use to launch the program and click on the "Start the Magic" button''')
    col_slider_1, col_slider_2, col_slider_3, col_slider_4 = st.columns(4)
    with col_slider_1 :
        # Définir les valeurs minimale et maximale du slider
        valeur_min_learning_rate= 0.0001
        valeur_max_learning_rate = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_learning_rate = st.slider("Learning rate", valeur_min_learning_rate, valeur_max_learning_rate, (valeur_min_learning_rate, valeur_max_learning_rate), step=0.001, key=1)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_learning_rate = valeurs_learning_rate[0]
        new_valeur_max_learning_rate = valeurs_learning_rate[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_learning_rate)
        st.write(new_valeur_max_learning_rate)
    with col_slider_2 : 
        valeur_min_batch_size= 0.0001
        valeur_max_batch_size = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_batch_size = st.slider("Batch size", valeur_min_batch_size, valeur_max_batch_size, (valeur_min_batch_size, valeur_max_batch_size), step=0.001, key=2)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_batch_size = valeurs_batch_size[0]
        new_valeur_max_batch_size = valeurs_batch_size[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_batch_size)
        st.write(new_valeur_max_batch_size)
    with col_slider_3 :
        valeur_min_epochs= 0.0001
        valeur_max_epochs = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_epochs = st.slider("Epochs", valeur_min_epochs, valeur_max_epochs, (valeur_min_epochs, valeur_max_epochs), step=0.001, key=3)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_epochs = valeurs_epochs[0]
        new_valeur_max_epochs = valeurs_epochs[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_epochs)
        st.write(new_valeur_max_epochs)
    with col_slider_4 :
        # Définir les valeurs minimale et maximale du slider
        valeur_min_l2= 0.0001
        valeur_max_l2 = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_l2 = st.slider("l2", valeur_min_l2, valeur_max_l2, (valeur_min_l2, valeur_max_l2), step=0.001, key=4)
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
        valeur_min_units= 0.0001
        valeur_max_units = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_units = st.slider("Units", valeur_min_units, valeur_max_units, (valeur_min_units, valeur_max_units), step=0.001, key=5)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_units = valeurs_units[0]
        new_valeur_max_units = valeurs_units[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_units)
        st.write(new_valeur_max_units)
    with col_slider_7 : 
        # Définir les valeurs minimale et maximale du slider
        valeur_min_unit= 0.0001
        valeur_max_unit = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_unit = st.slider("Unit", valeur_min_unit, valeur_max_unit, (valeur_min_unit, valeur_max_unit), step=0.001, key=6)
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_unit = valeurs_unit[0]
        new_valeur_max_unit = valeurs_unit[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_unit)
        st.write(new_valeur_max_unit)
    with col_slider_8: 
            # Définir les valeurs minimale et maximale du slider
        valeur_min_dropout= 0.0001
        valeur_max_dropout = 0.01
        # Utiliser le widget slider avec les valeurs minimale et maximale
        valeurs_dropout = st.slider("Dropout", valeur_min_dropout, valeur_max_dropout, (valeur_min_dropout, valeur_max_dropout), step=0.001, key=7 )
        # Obtenir les valeurs sélectionnées à partir du tuple retourné par le slider
        new_valeur_min_dropout = valeurs_dropout[0]
        new_valeur_max_dropout = valeurs_dropout[1]
        # Afficher les valeurs sélectionnées
        st.write(new_valeur_min_dropout)
        st.write(new_valeur_max_dropout)

    # Interaction avec la page 2
    if st.button("Afficher la page 2"):
        from RESULTS import show_results
        show_results()