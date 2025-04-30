"""
Página de dashboard principal.

Este módulo define la página de dashboard que muestra el estado actual
de los cajeros, KPIs principales y visualizaciones generales.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import requests
from datetime import datetime, timedelta

# Agregar ruta para importación de módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar módulos necesarios
from api.data.db_connector import create_db_connection, load_atm_data
from api.data.simulation import simulate_current_atm_status, generate_sample_carriers
from frontend.components.maps import atm_status_map
from frontend.components.charts import kpi_metrics_cards, atm_status_summary
from api.utils.helpers import get_exchange_rate, format_currency

# Configuración de la página
st.set_page_config(
    page_title="Dashboard - ATM Optimizer",
    page_icon="💰",
    layout="wide"
)

# Título
st.title("Dashboard de Estado Actual")
st.markdown("Monitoreo en tiempo real del estado de cajeros automáticos")

# Inicialización de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data():
    """Carga o simula datos necesarios para el dashboard"""
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
        
        # En implementación real, descomentar:
        from api.utils.metrics import calculate_current_kpis
        current_kpis = calculate_current_kpis(current_status)
        
        return current_status, carriers_df, current_kpis
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        # Generar datos de ejemplo como fallback
        atms_df = pd.DataFrame()  # Los datos se generarán en simulate_current_atm_status
        carriers_df = generate_sample_carriers(3)
        current_status = simulate_current_atm_status(atms_df)
        
        # KPIs simulados
        current_kpis = {
            'disponibilidad': 87.5,
            'downtime': 12.5,
            'eficiencia_capital': 64.3,
            'dias_hasta_agotamiento': 5.7,
            'requieren_atencion_pronto': 3
        }
        
        return current_status, carriers_df, current_kpis

# Cargar datos
current_status, carriers_df, current_kpis = load_data()

# Filtros en sidebar
st.sidebar.title("Filtros")

# Filtro por estado
status_filter = st.sidebar.multiselect(
    "Estado de cajeros",
    options=["Normal", "Advertencia", "Crítico"],
    default=["Normal", "Advertencia", "Crítico"]
)

# Filtro por tipo de ubicación
if 'location_type' in current_status.columns:
    location_types = current_status['location_type'].unique().tolist()
    location_filter = st.sidebar.multiselect(
        "Tipo de ubicación",
        options=location_types,
        default=location_types
    )
else:
    location_filter = []

# En el sidebar, después de los otros filtros
st.sidebar.title("Configuración")

# Selector de moneda
currency = st.sidebar.radio(
    "Moneda:",
    options=["COP", "USD"],
    index=0  # Predeterminado COP
)

# Obtener tasa de cambio si es necesario
exchange_rate = None
if currency == "USD":
    exchange_rate = get_exchange_rate()
    st.sidebar.info(f"Tasa de cambio: 1 USD = {exchange_rate:,.2f} COP")


# Aplicar filtros
filtered_status = current_status.copy()
if status_filter:
    filtered_status = filtered_status[filtered_status['status'].isin(status_filter)]
if location_filter:
    filtered_status = filtered_status[filtered_status['location_type'].isin(location_filter)]

# Dashboard principal
st.write("## Estado actual del sistema")

# Mostrar KPIs
kpi_metrics_cards(current_kpis)

# Dividir en dos columnas
col1, col2 = st.columns([2, 1])

# Columna 1: Mapa de estado
with col1:
    atm_status_map(filtered_status, carriers_df, currency=currency, exchange_rate=exchange_rate)

# Columna 2: Resumen y estadísticas
with col2:
    # Resumen de estados
    atm_status_summary(filtered_status)
    
    # Estadísticas adicionales
    st.write("### Estadísticas Generales")
    
    # Total de cajeros
    total_atms = len(current_status)
    st.metric("Total de cajeros", total_atms)
    
    # Efectivo total en cajeros
    if 'current_cash' in current_status.columns:
        total_cash = current_status['current_cash'].sum()
        formatted_total = format_currency(total_cash, currency, exchange_rate)
        st.metric("Efectivo total en circulación", formatted_total)
    
    # Fecha y hora de actualización
    st.info(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Tabla de datos completa (expandible)
with st.expander("Ver datos completos"):
    # Seleccionar columnas relevantes
    if len(filtered_status) > 0: 
        display_cols = ['id', 'name', 'location_type', 'current_cash', 
                        'capacity', 'usage_percent', 'status', 'days_until_empty']
        display_df = filtered_status[display_cols].copy()
        
        # Formatear columnas monetarias
        display_df['current_cash'] = display_df['current_cash'].apply(
            lambda x: format_currency(x, currency, exchange_rate)
        )
        display_df['capacity'] = display_df['capacity'].apply(
            lambda x: format_currency(x, currency, exchange_rate)
        )
        
        # Renombrar columnas para mejor visualización
        display_df.columns = ['ID', 'Nombre', 'Tipo de Ubicación', 'Efectivo Actual', 
                             'Capacidad', '% Utilización', 'Estado', 'Días hasta Agotamiento']
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No hay datos para mostrar con los filtros actuales.")