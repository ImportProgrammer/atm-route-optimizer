"""
Módulo de funciones auxiliares.

Este módulo proporciona funciones de utilidad general que pueden ser
utilizadas por otros componentes del sistema.
"""

import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula distancia entre coordenadas geográficas.
    
    Args:
        lat1, lon1: Coordenadas del primer punto
        lat2, lon2: Coordenadas del segundo punto
        
    Returns:
        Distancia en kilómetros
    """
    # Radio de la Tierra en km
    R = 6371.0
    
    # Convertir de grados a radianes
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Diferencias
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Fórmula de Haversine
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_exchange_rate():
    """
    Obtiene la tasa de cambio actual COP a USD.
    
    Returns:
        Tasa de cambio (cuántos COP equivalen a 1 USD)
    """
    try:
        # Intentar obtener tasa de API externa (ejemplo con Exchange Rate API)
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        # La API devuelve cuántos COP equivalen a 1 USD
        if 'rates' in data and 'COP' in data['rates']:
            return data['rates']['COP']
        else:
            # Tasa predeterminada si falla
            return 4100  # Valor aproximado COP/USD
    except:
        # Si falla la conexión, usar valor predeterminado
        return 4100  # Valor aproximado COP/USD

def format_currency(amount, currency="COP", exchange_rate=None):
    """
    Formatea un valor monetario en la moneda especificada.
    
    Args:
        amount: Cantidad a formatear
        currency: Moneda ("COP" o "USD")
        exchange_rate: Tasa de cambio COP/USD (requerido si currency es "USD")
        
    Returns:
        Cadena formateada con el valor monetario
    """
    if currency == "USD" and exchange_rate:
        # Convertir a USD y formatear con 2 decimales
        converted = amount / exchange_rate
        return f"${converted:,.2f} USD"
    else:
        # Mantener en COP y formatear sin decimales
        return f"${amount:,.0f} COP"

def get_date_filters(default_days=7):
    """
    Crea filtros de fecha para el análisis.
    
    Args:
        default_days: Número predeterminado de días para el análisis
        
    Returns:
        start_date, end_date: Fechas de inicio y fin seleccionadas
    """
    today = datetime.now().date()
    
    # Fecha predeterminada de inicio (hace X días)
    default_start = today - timedelta(days=default_days)
    
    # Crear selectores de fecha
    start_date = st.date_input(
        "Fecha de inicio",
        value=default_start,
        max_value=today
    )
    
    end_date = st.date_input(
        "Fecha de fin",
        value=today,
        min_value=start_date,
        max_value=today
    )
    
    return start_date, end_date