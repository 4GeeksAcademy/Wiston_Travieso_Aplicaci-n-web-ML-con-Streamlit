import streamlit as st
import pandas as pd
import pickle

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de ML", layout="centered")

# 2. Carga del modelo (Usa cache para que no se recargue en cada interacción)
@st.cache_resource
def load_model():
    with open("../models/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 3. Interfaz de usuario
st.title("🤖 Mi Modelo de Predicción")
st.markdown("Ingresa los datos para obtener una respuesta en tiempo real.")

# Crear columnas para organizar los inputs
col1, col2 = st.columns(2)

with col1:
    valor_a = st.number_input("Variable A", min_value=0.0)
    categoria = st.selectbox("Categoría", ["Opción 1", "Opción 2"])

with col2:
    valor_b = st.slider("Variable B", 0, 100, 50)

# 4. Lógica de predicción
if st.button("Realizar Predicción"):
    # Aquí transformas los inputs en el formato que espera tu modelo (ej. DataFrame)
    datos_entrada = pd.DataFrame([[valor_a, valor_b]]) 
    prediccion = model.predict(datos_entrada)
    
    st.success(f"El resultado de la predicción es: {prediccion[0]}")