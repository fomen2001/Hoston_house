# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import load_and_preprocess_data
from model import train_model, evaluate_model, predict

# 1. Chargement et préparation des données
(x_train_norm, y_train), (x_test_norm, y_test) = load_and_preprocess_data()

st.write(f"Taille des données d'entraînement : {x_train_norm.shape}")
st.write(f"Taille des données de test : {x_test_norm.shape}")

# 2. Entraînement du modèle
st.sidebar.header("Paramètres du modèle")
epochs = st.sidebar.slider("Nombre d'époques", min_value=50, max_value=500, value=100, step=50)

model, history = train_model(x_train_norm, y_train, x_test_norm, y_test, epochs=epochs)

# 3. Visualisation des courbes de perte et MAE
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

epochs_range = range(1, len(loss) + 1)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(epochs_range, loss, 'bo', label='Perte entraînement')
ax[0].plot(epochs_range, val_loss, 'b', label='Perte validation')
ax[0].set_title("Courbe de perte (MSE)")
ax[0].set_xlabel("Époques")
ax[0].set_ylabel("Perte")
ax[0].legend()

ax[1].plot(epochs_range, mae, 'ro', label='MAE entraînement')
ax[1].plot(epochs_range, val_mae, 'r', label='MAE validation')
ax[1].set_title("Courbe d'erreur moyenne absolue (MAE)")
ax[1].set_xlabel("Époques")
ax[1].set_ylabel("MAE")
ax[1].legend()

st.pyplot(fig)

# 4. Évaluation finale sur le test
loss_test, mae_test = evaluate_model(model, x_test_norm, y_test)
st.write(f"Évaluation finale (500 époques) sur test :")
st.write(f"Perte (MSE) : {loss_test:.4f}")
st.write(f"Erreur moyenne absolue (MAE) : {mae_test:.4f}")

# 5. Prédictions sur le jeu de test
predictions = predict(model, x_test_norm)
comparison_df = pd.DataFrame({
    'Prix réel': y_test,
    'Prix prédit': predictions,
    'Erreur absolue': np.abs(y_test - predictions)
})

st.write("Comparaison entre prix réels et prédits :")
st.write(comparison_df.head())

# 6. Formulaire d'entrée pour la prédiction
st.sidebar.header("Entrez les caractéristiques de la maison")

# Entrées de l'utilisateur pour prédiction
features = []
feature_names = [
    "CRIM (Taux de criminalité par habitant)",
    "ZN (Proportion de terres résidentielles à faible densité)",
    "INDUS (Proportion de zones commerciales)",
    "CHAS (Proximité de la rivière Charles)",
    "NOX (Concentration de dioxyde d'azote)",
    "RM (Nombre moyen de pièces par logement)",
    "AGE (Proportion de logements construits avant 1940)",
    "DIS (Distance à divers centres d'emploi)",
    "RAD (Accessibilité aux autoroutes radiales)",
    "TAX (Taux de taxe sur la propriété)",
    "PTRATIO (Ratio élèves/enseignant)",
    "B (Proportion de résidents noirs)",
    "LSTAT (Pourcentage de population à statut social inférieur)"
]

# Demander à l'utilisateur de saisir les valeurs des caractéristiques
for feature in feature_names:
    feature_value = st.sidebar.number_input(feature, min_value=0.0, format="%.2f")
    features.append(feature_value)

# Convertir les entrées en un tableau NumPy
user_input = np.array(features).reshape(1, -1)

# Normalisation des entrées de l'utilisateur
mean = x_train_norm.mean(axis=0)
std = x_train_norm.std(axis=0)
user_input_norm = (user_input - mean) / std

# 7. Prédiction en temps réel
if st.sidebar.button("Faire la prédiction"):
    prediction = model.predict(user_input_norm)
    st.write(f"Le prix estimé de la maison est : ${prediction[0][0]:.2f}")
