"""
Módulo para la simulación de datos de cajeros automáticos.

Este módulo proporciona funciones para simular estados actuales de cajeros,
patrones de consumo y predicciones de demanda cuando no hay datos reales disponibles.

Funciones principales:
    - simulate_current_atm_status: Simula el estado actual de los cajeros
    - simulate_consumption_patterns: Simula patrones de consumo por día/hora
    - generate_synthetic_transactions: Genera transacciones sintéticas
"""

"""
Módulo para la simulación de datos de cajeros automáticos.

Este módulo proporciona funciones para simular estados actuales de cajeros,
patrones de consumo y predicciones de demanda cuando no hay datos reales disponibles.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math

def simulate_current_atm_status(atms_df):
    """
    Simula el estado actual de los cajeros basado en datos históricos de consumo.
    
    Args:
        atms_df: DataFrame con información de cajeros
        
    Returns:
        DataFrame con estado actual incluyendo:
        - Efectivo disponible
        - % de capacidad utilizada
        - Estado (Normal, Advertencia, Crítico)
    """
    # Verificar si hay datos de cajeros
    if len(atms_df) == 0:
        print("No hay datos de cajeros para simular. Generando datos de ejemplo...")
        atms_df = generate_sample_atms(15)  # Generar 15 cajeros de ejemplo
    
    # Copiar dataframe para no modificar el original
    status_df = atms_df.copy()
    
    # Simular efectivo disponible (entre 10% y 90% de capacidad)
    status_df['current_cash'] = status_df.apply(
        lambda row: max(0, row['capacity'] * random.uniform(0.1, 0.9)), 
        axis=1
    )
    
    # Calcular porcentaje de capacidad utilizada
    status_df['usage_percent'] = (status_df['current_cash'] / status_df['capacity'] * 100).round(2)
    
    # Determinar estado basado en umbral y capacidad actual
    status_df['status'] = status_df.apply(
        lambda row: 'Crítico' if row['current_cash'] < row['min_threshold'] else
                    'Advertencia' if row['current_cash'] < row['min_threshold'] * 1.5 else
                    'Normal',
        axis=1
    )
    
    # Simular último reabastecimiento (entre 1 y 14 días atrás)
    today = datetime.now().date()
    status_df['last_restock'] = status_df.apply(
        lambda row: (today - timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d'),
        axis=1
    )
    
    # Simular días estimados hasta agotamiento basado en consumo promedio
    # Consumo diario aproximado entre 3-10% de capacidad
    status_df['daily_consumption'] = status_df.apply(
        lambda row: row['capacity'] * random.uniform(0.03, 0.1),
        axis=1
    )
    
    # Calcular días hasta agotamiento
    status_df['days_until_empty'] = status_df.apply(
        lambda row: max(0, (row['current_cash'] - row['min_threshold']) / row['daily_consumption']),
        axis=1
    ).round(1)
    
    return status_df

def generate_sample_atms(num_atms=10):
    """
    Genera datos de ejemplo para cajeros cuando no hay datos reales.
    
    Args:
        num_atms: Número de cajeros a generar
        
    Returns:
        DataFrame con datos simulados de cajeros
    """
    # Coordenadas aproximadas para Bogotá, Colombia
    bogota_center_lat = 4.6486
    bogota_center_lon = -74.0821
    
    # Tipos de ubicación posibles
    location_types = ['Centro Comercial', 'Oficina', 'Vía Principal', 'Sucursal']
    
    # Generar datos
    atms_data = []
    
    for i in range(1, num_atms + 1):
        # Añadir variación aleatoria a las coordenadas
        lat_offset = random.uniform(-0.05, 0.05)
        lon_offset = random.uniform(-0.05, 0.05)
        
        # Crear cajero
        atm = {
            'id': i,
            'name': f'Cajero {i:03d}',
            'latitude': bogota_center_lat + lat_offset,
            'longitude': bogota_center_lon + lon_offset,
            'capacity': random.randint(8000, 20000) * 10000,  # Entre 80M y 200M COP
            'cash_type': 'COP',
            'location_type': random.choice(location_types),
            'min_threshold': random.randint(1000, 3000) * 10000,  # Entre 10M y 30M COP
            'max_capacity': random.randint(8000, 20000) * 10000,  # Igual a capacity
        }
        atms_data.append(atm)
    
    return pd.DataFrame(atms_data)

def generate_sample_carriers(num_carriers=3):
    """
    Genera datos de ejemplo para transportadoras cuando no hay datos reales.
    
    Args:
        num_carriers: Número de transportadoras a generar
        
    Returns:
        DataFrame con datos simulados de transportadoras
    """
    # Coordenadas en Bogotá para bases de operaciones
    bases = [
        {'lat': 4.6761, 'lon': -74.0486, 'name': 'Transportadora Norte'},
        {'lat': 4.6278, 'lon': -74.0636, 'name': 'Transportadora Centro'},
        {'lat': 4.5981, 'lon': -74.1131, 'name': 'Transportadora Sur'},
        {'lat': 4.6342, 'lon': -74.1355, 'name': 'Transportadora Oeste'},
    ]
    
    # Áreas de servicio
    service_areas = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro', 'Toda la ciudad']
    
    # Generar datos
    carriers_data = []
    
    for i in range(1, min(num_carriers + 1, len(bases) + 1)):
        base = bases[i-1]
        
        carrier = {
            'id': i,
            'name': base['name'],
            'base_latitude': base['lat'],
            'base_longitude': base['lon'],
            'capacity': random.randint(30000, 50000) * 10000,  # Entre 300M y 500M COP
            'vehicles': random.randint(2, 5),
            'service_area': service_areas[i-1] if i <= len(service_areas) else 'Toda la ciudad'
        }
        carriers_data.append(carrier)
    
    return pd.DataFrame(carriers_data)

def generate_sample_restrictions(atms_df):
    """
    Genera restricciones de ejemplo para cajeros.
    
    Args:
        atms_df: DataFrame con cajeros
        
    Returns:
        DataFrame con restricciones simuladas
    """
    restrictions_data = []
    
    # Para cada cajero, crear restricciones aleatorias
    for _, atm in atms_df.iterrows():
        atm_id = atm['id']
        
        # Asignar restricciones aleatorias a algunos cajeros
        if random.random() < 0.7:  # 70% de los cajeros tienen restricciones
            # Días con restricciones (0=lunes, 6=domingo)
            restricted_days = random.sample(range(7), random.randint(1, 3))
            
            for day in restricted_days:
                # Hora de apertura (entre 7 AM y 10 AM)
                open_hour = random.randint(7, 10)
                # Hora de cierre (entre 5 PM y 9 PM)
                close_hour = random.randint(17, 21)
                
                # Permitir reabastecimiento (algunos días no)
                restock_allowed = random.random() < 0.7
                
                restrictions_data.append({
                    'atm_id': atm_id,
                    'day_of_week': day,
                    'open_time': f'{open_hour:02d}:00:00',
                    'close_time': f'{close_hour:02d}:00:00',
                    'restock_allowed': restock_allowed
                })
    
    return pd.DataFrame(restrictions_data)