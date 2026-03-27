import streamlit as st
import pandas as pd
import pickle
from pickle import load

import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración y Estética
st.set_page_config(page_title="Proveedor Ideal", page_icon="📊")

st.title("🛒 Optimizador de Compras")
st.markdown("""
Esta herramienta predice el **Proveedor más probable** para una compra basándose en el histórico de transacciones del equipo.
""")

# 2. Función para cargar el modelo y los encoders
@st.cache_resource
def load_assets():
    # Ajustamos las rutas al entorno de Render (usualmente en la raíz o /src)
    model = joblib.load('models/model.pkl')
    le_item = joblib.load('models/le_item.pkl')
    le_cat = joblib.load('models/le_cat.pkl')
    le_target = joblib.load('models/le_target.pkl')
    return model, le_item, le_cat, le_target

try:
    model, le_item, le_cat, le_target = load_assets()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Asegúrate de que la carpeta 'models' esté en el repo.")
    st.stop()

# 3. Formulario de entrada
with st.sidebar:
    st.header("Parámetros de Compra")
    
    # Selectores basados en las clases que aprendió el Encoder
    item = st.selectbox("Producto (ItemName)", le_item.classes_)
    category = st.selectbox("Categoría", le_cat.classes_)
    
    quantity = st.number_input("Cantidad", min_value=1, value=10)
    unit_price = st.number_input("Precio Unitario", min_value=0.1, value=50.0)
    
    total_cost = quantity * unit_price
    st.info(f"Costo Total Estimado: ${total_cost:,.2f}")

# 4. Predicción
if st.button("Sugerir Proveedor", type="primary"):
    # Transformar las entradas usando los encoders cargados
    item_encoded = le_item.transform([item])[0]
    cat_encoded = le_cat.transform([category])[0]
       
    input_df = pd.DataFrame([[
        item_encoded, 
        cat_encoded, 
    ]], columns=['ItemName', 'Category'])

    input_data = input_df.values 
    
    try:
        # Usamos input_data (los valores puros) para saltar la validación de nombres
        prediction_numeric = model.predict(input_data)[0]
        prediction_name = le_target.inverse_transform([prediction_numeric])[0]
        
        # Mostrar resultado
        st.success(f"### Proveedor Recomendado: **{prediction_name}**")
        
        # Confianza
        probs = model.predict_proba(input_data)
        confianza = max(probs[0]) * 100
        st.info(f"Nivel de confianza: {confianza:.2f}%")
        
    except Exception as e:
        st.error(f"Error en la predicción: {e}")

st.divider()
st.caption("Proyecto de Data Science - Wiston Travieso")