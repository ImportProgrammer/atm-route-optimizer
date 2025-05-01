"""
Módulo para el cálculo de métricas y KPIs.

Este módulo implementa funciones para calcular y evaluar métricas de negocio
relacionadas con la gestión de efectivo en cajeros automáticos.
"""

import pandas as pd
import numpy as np

def calculate_current_kpis(status_df):
    """
    Calcula KPIs basados en el estado actual de los cajeros.
    
    Args:
        status_df: DataFrame con el estado actual de los cajeros
        
    Returns:
        Diccionario con KPIs calculados
    """
    # Verificar que hay datos
    if status_df is None or len(status_df) == 0:
        # Devolver valores predeterminados si no hay datos
        return {
            'disponibilidad': 0,
            'downtime': 0,
            'eficiencia_capital': 0,
            'dias_hasta_agotamiento': 0,
            'requieren_atencion_pronto': 0
        }
    
    # Calcular KPIs
    kpis = {}
    
    # KPI 1: Disponibilidad de efectivo (% de cajeros por encima del umbral mínimo)
    total_atms = len(status_df)
    atms_above_threshold = len(status_df[status_df['current_cash'] >= status_df['min_threshold']])
    kpis['disponibilidad'] = round((atms_above_threshold / total_atms * 100) if total_atms > 0 else 0, 2)
    
    # KPI 2: Downtime por agotamiento (% de cajeros en estado crítico)
    critical_atms = len(status_df[status_df['status'] == 'Crítico'])
    kpis['downtime'] = round((critical_atms / total_atms * 100) if total_atms > 0 else 0, 2)
    
    # KPI 3: Eficiencia de capital (% promedio de utilización de capacidad)
    kpis['eficiencia_capital'] = round(status_df['usage_percent'].mean() if total_atms > 0 else 0, 2)
    
    # KPI 4: Urgencia de reabastecimiento (días promedio hasta agotamiento)
    kpis['dias_hasta_agotamiento'] = round(status_df['days_until_empty'].mean() if total_atms > 0 else 0, 2)
    
    # KPI 5: Cajeros que requieren atención en los próximos 3 días
    require_attention_soon = len(status_df[status_df['days_until_empty'] <= 3])
    kpis['requieren_atencion_pronto'] = require_attention_soon
    
    return kpis

def calculate_improved_kpis(status_df, atms_to_restock):
    """
    Calcula KPIs después de implementar las rutas optimizadas.
    
    Args:
        status_df: DataFrame con estado actual de cajeros
        atms_to_restock: DataFrame con cajeros que serán reabastecidos
        
    Returns:
        Diccionario con KPIs mejorados
    """
    # Verificar que hay datos
    if status_df is None or len(status_df) == 0:
        return calculate_current_kpis(status_df)  # Devolver KPIs actuales si no hay datos
    
    # Copiar dataframe para no modificar el original
    improved_status = status_df.copy()
    
    # Identificar cajeros que serán reabastecidos
    if len(atms_to_restock) > 0:
        for _, atm in atms_to_restock.iterrows():
            idx = improved_status[improved_status['id'] == atm['atm_id']].index
            if len(idx) > 0:
                improved_status.loc[idx, 'current_cash'] = atm.get('max_capacity', atm['current_cash'])
                improved_status.loc[idx, 'usage_percent'] = 100
                improved_status.loc[idx, 'status'] = 'Normal'
                improved_status.loc[idx, 'days_until_empty'] = improved_status.loc[idx, 'days_until_empty'] * 2

    # Calcular KPIs mejorados
    return calculate_current_kpis(improved_status)

def calculate_savings(routes, atms_to_restock, carrier, alternative_routes=None):
    """
    Calcula el ahorro estimado por la optimización de rutas.
    
    Args:
        routes: Lista de rutas optimizadas
        atms_to_restock: DataFrame con cajeros a reabastecer
        carrier: Información de la transportadora
        alternative_routes: Rutas alternativas para comparación
        
    Returns:
        Diccionario con cálculos de ahorro
    """
    if not routes or len(atms_to_restock) == 0 or carrier is None:
        return {
            'distancia_optimizada': 0,
            'distancia_no_optimizada': 0,
            'ahorro_distancia': 0,
            'ahorro_porcentaje': 0,
            'ahorro_costo': 0,
            'ahorro_mensual': 0
        }
    
    # Parámetros de costo
    costo_por_km = 5000  # COP por km (ejemplo)
    costo_por_visita = 100000  # COP por visita a cajero (ejemplo)
    dias_operacion_mes = 20  # días hábiles al mes
    
    # Calcular para rutas optimizadas
    distancia_optimizada = sum(route['distance'] for route in routes)
    visitas_optimizadas = sum(len(route['route']) - 2 for route in routes)
    
    # Si hay rutas alternativas, usar esas como comparación
    if alternative_routes and len(alternative_routes) > 0:
        distancia_no_optimizada = sum(route['distance'] for route in alternative_routes)
        visitas_no_optimizadas = sum(len(route['route']) - 2 for route in alternative_routes)
    else:
        # Simular enfoque no optimizado (cada cajero es una visita individual desde la base)
        distancia_no_optimizada = 0
        for _, atm in atms_to_restock.iterrows():
            # Distancia de ida y vuelta a cada cajero (aproximación)
            dist_ida_vuelta = haversine_distance(
                carrier['base_latitude'], carrier['base_longitude'],
                atm['latitude'], atm['longitude']
            ) * 2  # ida y vuelta
            distancia_no_optimizada += dist_ida_vuelta
        
        visitas_no_optimizadas = len(atms_to_restock)
    
    # Cálculo de costos
    costo_optimizado = (distancia_optimizada * costo_por_km) + (visitas_optimizadas * costo_por_visita)
    costo_no_optimizado = (distancia_no_optimizada * costo_por_km) + (visitas_no_optimizadas * costo_por_visita)
    
    # Cálculos de ahorro
    ahorro_distancia = distancia_no_optimizada - distancia_optimizada
    ahorro_porcentaje = (ahorro_distancia / distancia_no_optimizada * 100) if distancia_no_optimizada > 0 else 0
    ahorro_costo = costo_no_optimizado - costo_optimizado
    ahorro_mensual = ahorro_costo * dias_operacion_mes
    
    return {
        'distancia_optimizada': round(distancia_optimizada, 2),
        'distancia_no_optimizada': round(distancia_no_optimizada, 2),
        'ahorro_distancia': round(ahorro_distancia, 2),
        'ahorro_porcentaje': round(ahorro_porcentaje, 2),
        'ahorro_costo': round(ahorro_costo, 2),
        'ahorro_mensual': round(ahorro_mensual, 2)
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia de Haversine entre dos puntos en la Tierra.
    
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