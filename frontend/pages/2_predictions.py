"""
Página de predicciones de demanda.

Este módulo define la página que muestra las predicciones de demanda
futura, tendencias y cajeros prioritarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Agregar ruta para importación de módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar módulos necesarios
from api.data.db_connector import create_db_connection, load_atm_data
from api.data.simulation import simulate_current_atm_status, generate_sample_carriers
from api.models.prediction import predict_cash_demand, get_priority_atms, get_demand_by_day
from api.utils.helpers import format_currency, get_exchange_rate
from frontend.components.charts import demand_forecast_chart
from frontend.components.maps import atm_status_map

# Configuración de la página
st.set_page_config(
    page_title="Predicciones - ATM Optimizer",
    page_icon="💰",
    layout="wide"
)

# Título
st.title("Predicciones de Demanda")
st.markdown("Análisis predictivo de la demanda futura de efectivo")

# Inicialización de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_prediction_data():
    """Carga o simula datos necesarios para predicciones"""
    try:
        # Intentar cargar datos reales
        engine = create_db_connection()
        atms_df, carriers_df, restrictions_df = load_atm_data(engine)
        
        # Si no hay datos, generar datos de ejemplo
        if len(atms_df) == 0:
            st.warning("No se encontraron datos en la base de datos. Usando datos simulados.")
            atms_df = pd.DataFrame()  # Los datos se generarán en simulate_current_atm_status
            carriers_df = generate_sample_carriers(3)
        
        # Simular estado actual
        current_status = simulate_current_atm_status(atms_df)
        
        # Generar predicciones
        predictions = predict_cash_demand(current_status, days_ahead=7)
        
        return current_status, predictions, carriers_df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        # Generar datos de ejemplo como fallback
        atms_df = pd.DataFrame()  # Los datos se generarán en simulate_current_atm_status
        current_status = simulate_current_atm_status(atms_df)
        predictions = predict_cash_demand(current_status, days_ahead=7)
        carriers_df = generate_sample_carriers(3)
        
        return current_status, predictions, carriers_df

# Cargar datos
current_status, predictions, carriers_df = load_prediction_data()

# Filtros en sidebar
st.sidebar.title("Filtros")

# Selector de fecha de predicción
available_dates = sorted(predictions['date'].unique())
selected_date = st.sidebar.selectbox(
    "Fecha de predicción",
    options=available_dates,
    index=0
)

# Selector de prioridad
priority_options = {
    "Todas": 1,
    "Media y Alta": 2,
    "Solo Alta": 3
}
selected_priority = st.sidebar.radio(
    "Mostrar cajeros con prioridad:",
    options=list(priority_options.keys()),
    index=1
)
min_priority = priority_options[selected_priority]

# Configuración de moneda
st.sidebar.title("Configuración")
currency = st.sidebar.radio(
    "Moneda:",
    options=["COP", "USD"],
    index=0
)

# Obtener tasa de cambio si es necesario
exchange_rate = None
if currency == "USD":
    exchange_rate = get_exchange_rate()
    st.sidebar.info(f"Tasa de cambio: 1 USD = {exchange_rate:,.2f} COP")

# Filtrar cajeros por prioridad
priority_atms = get_priority_atms(predictions, selected_date, min_priority)

# Mostrar resumen de predicciones
st.write("## Resumen de predicciones para " + selected_date)

# Métricas de resumen
col1, col2, col3 = st.columns(3)

with col1:
    total_atms_priority = len(priority_atms)
    st.metric("Cajeros requiriendo atención", total_atms_priority)

with col2:
    if len(priority_atms) > 0:
        total_demand = priority_atms['predicted_demand'].sum()
        formatted_demand = format_currency(total_demand, currency, exchange_rate)
        st.metric("Demanda total estimada", formatted_demand)
    else:
        st.metric("Demanda total estimada", "0")

with col3:
    # Promedio días hasta agotamiento
    if len(priority_atms) > 0:
        avg_days = priority_atms['days_until_empty'].mean()
        st.metric("Promedio días hasta agotamiento", f"{avg_days:.1f} días")
    else:
        st.metric("Promedio días hasta agotamiento", "N/A")

# Gráfico de demanda por día
st.write("### Demanda proyectada por día")

# Calcular demanda total por día
daily_demand = get_demand_by_day(predictions)

# Crear gráfico
fig = px.bar(
    daily_demand,
    x='date',
    y='total_demand',
    color='is_payday',
    color_discrete_map={0: '#4e73df', 1: '#1cc88a'},
    labels={'total_demand': 'Demanda Total', 'date': 'Fecha', 'is_payday': 'Día de Pago'},
    title='Demanda Total Proyectada por Día'
)

# Ajustar formato del eje Y para moneda
if currency == "USD" and exchange_rate:
    fig.update_layout(yaxis=dict(tickprefix='$', ticksuffix=' USD'))
    # Convertir valores
    fig.update_traces(y=daily_demand['total_demand'] / exchange_rate)
else:
    fig.update_layout(yaxis=dict(tickprefix='$', ticksuffix=' COP'))

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Dividir en dos columnas para mostrar mapa y tabla
col1, col2 = st.columns([2, 1])

# Columna 1: Mapa de cajeros prioritarios
with col1:
    st.write("### Cajeros prioritarios para reabastecimiento")
    
    if len(priority_atms) > 0:
        # Crear un dataframe de estado específico para cajeros prioritarios
        priority_status = pd.merge(
            current_status,
            priority_atms[['atm_id', 'priority']],
            left_on='id',
            right_on='atm_id'
        )
        
        # Mostrar mapa
        atm_status_map(priority_status, carriers_df, currency=currency, exchange_rate=exchange_rate)
    else:
        st.info("No hay cajeros prioritarios para la fecha y criterios seleccionados.")

# Columna 2: Tabla de cajeros prioritarios
with col2:
    st.write("### Detalles de cajeros prioritarios")
    
    if len(priority_atms) > 0:
        # Preparar datos para tabla
        display_cols = ['atm_id', 'current_cash', 'predicted_demand', 'days_until_empty', 'priority']
        display_df = priority_atms[display_cols].copy()
        
        # Formatear columnas monetarias
        display_df['current_cash'] = display_df['current_cash'].apply(
            lambda x: format_currency(x, currency, exchange_rate)
        )
        display_df['predicted_demand'] = display_df['predicted_demand'].apply(
            lambda x: format_currency(x, currency, exchange_rate)
        )
        
        # Mapear prioridad a texto
        priority_map = {3: 'Alta', 2: 'Media', 1: 'Baja'}
        display_df['priority'] = display_df['priority'].map(priority_map)
        
        # Renombrar columnas
        display_df.columns = ['ID Cajero', 'Efectivo Actual', 'Demanda Estimada', 'Días hasta Agotamiento', 'Prioridad']
        
        # Ordenar por días hasta agotamiento
        display_df = display_df.sort_values('Días hasta Agotamiento')
        
        # Mostrar tabla
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")

# Visualización detallada de cajero individual
st.write("## Análisis detallado por cajero")

# Selector de cajero
if len(current_status) > 0:
    atm_options = current_status[['id', 'name']].copy()
    atm_options['label'] = atm_options.apply(lambda x: f"{x['id']} - {x['name']}", axis=1)
    
    selected_atm_label = st.selectbox(
        "Seleccione un cajero para analizar:",
        options=atm_options['label'].tolist()
    )
    
    # Extraer el ID del cajero (sin convertir a int)
    selected_atm_id = selected_atm_label.split(' - ')[0]
    
    # Obtener información del cajero usando el ID original
    atm_info = current_status[current_status['id'] == selected_atm_id].iloc[0]
    
    # Mostrar pronóstico para el cajero seleccionado
    demand_forecast_chart(predictions, selected_atm_id, atm_info, currency=currency, exchange_rate=exchange_rate)
else:
    st.warning("No hay datos de cajeros disponibles.")