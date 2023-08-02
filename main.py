import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder

def main():

    # Titre de la page 
    st.markdown("<h1 style='text-align: center;'>Prédiction Churn</h1>", unsafe_allow_html=True)

    # Chargement et affichage de l'image
    #image = 'assets/powerforecast.png'
    #st.image(image, caption='', use_column_width=True)

    st.write("Entrez les données du client pour la prédiction :")
    age = st.number_input("Age", min_value=18, max_value=100)
    sexe = st.selectbox("Sexe", ['F', 'M'])
    duree = st.number_input("Durée d'abonnement (en mois)", min_value=1)
    lecture = st.number_input("Nombre de lectures", min_value=0)
    taux = st.slider("Taux d'ouverture", min_value=0.0, max_value=1.0)

    encoder = LabelEncoder()
    input_client = pd.DataFrame({'Age': [age], 'Sexe': [sexe], "Durée d'abonnement (en mois)": [duree], "Nombre de lectures": [lecture], "Taux d'ouverture": [taux]})
    input_client['Sexe'] = encoder.fit_transform(input_client['Sexe'])

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    if st.button("Prédire"):
        prediction = model.predict(input_client)
        if prediction == 1 : 
            st.write("***Le client pourrait se désabonner de la newsletter***")
        else :
            st.write("***Le client pourrait continuer son abonnement***")

if __name__ == '__main__':
    main()
