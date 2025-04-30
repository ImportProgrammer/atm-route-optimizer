"""
Módulo para la predicción de demanda de efectivo.

Este módulo implementa algoritmos para predecir la demanda futura de efectivo
en cajeros automáticos basados en datos históricos, patrones y factores externos.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def predict_cash_demand(status_df, days_ahead=7):
    """
    Predice la demanda de efectivo para los próximos días.
    
    Args:
        status_df: DataFrame con estado actual de cajeros
        days_ahead: Número de días para los que se generarán predicciones
        
    Returns:
        DataFrame con predicciones diarias de demanda
    """
    predictions = []
    current_date = datetime.now().date()
    
    # Para cada cajero y día, generar predicción
    for _, atm in status_df.iterrows():
        # Obtener consumo diario base (de la simulación del estado actual)
        daily_base = atm['daily_consumption']
        
        # Para cada día futuro
        for day in range(1, days_ahead + 1):
            date = current_date + timedelta(days=day)
            
            # Factores que afectan la demanda
            # 1. Factor del día de la semana (más alto para viernes y fines de semana)
            day_of_week = date.weekday()
            if day_of_week == 4:  # Viernes
                day_factor = 1.4
            elif day_of_week in [5, 6]:  # Fin de semana
                day_factor = 1.2
            else:
                day_factor = 1.0
                
            # 2. Factor de quincena (más alto para días 15 y 30)
            if date.day in [15, 16, 30, 1]:
                pay_day_factor = 1.8
            else:
                pay_day_factor = 1.0
                
            # 3. Factor aleatorio para simular variabilidad
            random_factor = random.uniform(0.8, 1.2)
            
            # Calcular demanda final
            predicted_demand = daily_base * day_factor * pay_day_factor * random_factor
            
            # Añadir a predicciones
            predictions.append({
                'atm_id': atm['id'],
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': predicted_demand,
                'current_cash': max(0, atm['current_cash'] - (daily_base * (day-1) * random.uniform(0.9, 1.1))),
                'day_of_week': day_of_week,
                'is_payday': 1 if date.day in [15, 16, 30, 1] else 0
            })
    
    # Convertir a DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Calcular días hasta agotamiento para cada cajero y fecha
    predictions_df = pd.merge(
        predictions_df,
        status_df[['id', 'min_threshold']],
        left_on='atm_id',
        right_on='id'
    )
    
    # Calcular días hasta agotamiento (basado en efectivo actual y demanda proyectada)
    predictions_df['days_until_empty'] = predictions_df.apply(
        lambda row: max(0, (row['current_cash'] - row['min_threshold']) / row['predicted_demand']),
        axis=1
    ).round(1)
    
    # Asignar prioridad basada en días hasta agotamiento
    predictions_df['priority'] = predictions_df['days_until_empty'].apply(
        lambda x: 3 if x <= 1 else (2 if x <= 3 else 1)
    )
    
    return predictions_df

def get_priority_atms(predictions_df, date=None, min_priority=2):
    """
    Identifica cajeros prioritarios para reabastecimiento.
    
    Args:
        predictions_df: DataFrame con predicciones
        date: Fecha para la que se seleccionarán cajeros (YYYY-MM-DD)
        min_priority: Prioridad mínima para selección (1=baja, 2=media, 3=alta)
        
    Returns:
        DataFrame con cajeros seleccionados
    """
    if date is None:
        # Si no se especifica fecha, usar la primera disponible
        date = predictions_df['date'].min()
    
    # Filtrar predicciones para la fecha especificada
    day_predictions = predictions_df[predictions_df['date'] == date].copy()
    
    # Seleccionar cajeros con prioridad suficiente
    priority_atms = day_predictions[day_predictions['priority'] >= min_priority].copy()
    
    return priority_atms

def get_demand_by_day(predictions_df):
    """
    Calcula la demanda agregada por día.
    
    Args:
        predictions_df: DataFrame con predicciones
        
    Returns:
        DataFrame con demanda total por día
    """
    # Agrupar y sumar demanda por día
    daily_demand = predictions_df.groupby('date')['predicted_demand'].sum().reset_index()
    daily_demand.columns = ['date', 'total_demand']
    
    # Determinar si cada día es día de pago
    daily_demand['is_payday'] = daily_demand['date'].apply(
        lambda x: 1 if int(x.split('-')[2]) in [15, 16, 30, 1] else 0
    )
    
    return daily_demand