"""
Componentes de gráficos para la interfaz de usuario.

Este módulo proporciona componentes Streamlit reutilizables para la
visualización de datos mediante gráficos y elementos visuales.

Componentes principales:
    - demand_forecast_chart: Gráfico de pronóstico de demanda
    - kpi_metrics_cards: Tarjetas visuales para KPIs
    - comparison_chart: Gráfico para comparación de escenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def kpi_metrics_cards(kpis, comparison_kpis=None):
    """
    Muestra tarjetas de KPIs en Streamlit.
    
    Args:
        kpis: Diccionario con valores actuales de KPIs
        comparison_kpis: Diccionario opcional con valores de comparación
    """
    # Definir colores
    colors = {
        'primary': '#4e73df',
        'success': '#1cc88a',
        'warning': '#f6c23e',
        'danger': '#e74a3b',
    }
    
    # Definiciones de KPIs con sus umbrales
    kpi_definitions = {
        'disponibilidad': {
            'name': 'Disponibilidad de Efectivo',
            'unit': '%',
            'icon': '✓',
            'target': '≥ 85%',
            'is_good': lambda v: v >= 85,
            'delta_good_direction': 'up',
            'description': 'Porcentaje de cajeros con efectivo por encima del umbral mínimo'
        },
        'downtime': {
            'name': 'Downtime por Agotamiento',
            'unit': '%',
            'icon': '⚠️',
            'target': '≤ 15%',
            'is_good': lambda v: v <= 15,
            'delta_good_direction': 'down',
            'description': 'Porcentaje de cajeros en estado crítico (sin efectivo disponible)'
        },
        'eficiencia_capital': {
            'name': 'Eficiencia de Capital',
            'unit': '%',
            'icon': '💰',
            'target': '40-70%',
            'is_good': lambda v: 40 <= v <= 70,
            'delta_good_direction': 'up' if kpis.get('eficiencia_capital', 0) < 55 else 'down',
            'description': 'Utilización promedio de la capacidad de los cajeros'
        },
        'dias_hasta_agotamiento': {
            'name': 'Días hasta Agotamiento',
            'unit': ' días',
            'icon': '📅',
            'target': '≥ 5 días',
            'is_good': lambda v: v >= 5,
            'delta_good_direction': 'up',
            'description': 'Promedio de días hasta que los cajeros lleguen al umbral mínimo'
        },
        'requieren_atencion_pronto': {
            'name': 'Proyectados para Atención',
            'unit': ' cajeros',
            'icon': '🔔',
            'target': 'Mínimo',
            'is_good': lambda v: v <= 5,
            'delta_good_direction': 'down',
            'description': 'Cajeros que se agotarán en los próximos 3 días'
        }
    }
    
    # Crear columnas para los KPIs
    cols = st.columns(len(kpis))
    
    # Mostrar cada KPI
    for i, (key, value) in enumerate(kpis.items()):
        if key in kpi_definitions:
            definition = kpi_definitions[key]
            
            # Calcular delta si hay comparación
            delta = None
            if comparison_kpis and key in comparison_kpis:
                delta = value - comparison_kpis[key]
                if definition['delta_good_direction'] == 'down':
                    delta = -delta
            
            # Determinar color del valor
            value_color = colors['success'] if definition['is_good'](value) else colors['danger']
            
            # Mostrar métrica con tooltip que incluye la descripción
            with cols[i]:
                st.metric(
                    label=f"{definition['icon']} {definition['name']}",
                    value=f"{value}{definition['unit']}",
                    delta=f"{delta:+.2f}{definition['unit']}" if delta is not None else None,
                    help=f"{definition['description']} (Objetivo: {definition['target']})"
                )
                
                # Mostrar barra de progreso para valores porcentuales
                if definition['unit'] == '%':
                    if key == 'eficiencia_capital':
                        # Caso especial para eficiencia de capital (rango óptimo)
                        progress_color = value_color
                        # Usar una escala de 0 a 100
                        st.progress(value/100)
                    else:
                        # Para otros porcentajes
                        progress_value = value/100 if key != 'downtime' else 1-(value/100)
                        st.progress(progress_value)
                else: 
                    st.empty()
                
                # Mostrar descripción bajo cada KPI
                st.caption(definition['description'])

def atm_status_summary(status_df):
    """
    Muestra un resumen del estado de los cajeros.
    
    Args:
        status_df: DataFrame con el estado actual de los cajeros
    """
    if status_df is None or len(status_df) == 0:
        st.warning("No hay datos de estado para mostrar el resumen.")
        return
    
    # Contar cajeros por estado
    status_counts = status_df['status'].value_counts().reset_index()
    status_counts.columns = ['Estado', 'Cantidad']
    
    # Ordenar por criticidad
    status_order = {'Crítico': 0, 'Advertencia': 1, 'Normal': 2}
    status_counts['Orden'] = status_counts['Estado'].map(status_order)
    status_counts = status_counts.sort_values('Orden').drop('Orden', axis=1)
    
    # Crear gráfico con Plotly
    colors = {'Crítico': '#e74a3b', 'Advertencia': '#f6c23e', 'Normal': '#1cc88a'}
    
    fig = px.bar(
        status_counts, 
        x='Estado', 
        y='Cantidad',
        color='Estado',
        color_discrete_map=colors,
        text='Cantidad',
        title='Resumen de Estado de Cajeros'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Cantidad de Cajeros',
        showlegend=False,
        height=300
    )
    
    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla de cajeros críticos si existen
    critical_atms = status_df[status_df['status'] == 'Crítico'].sort_values('days_until_empty')
    
    if len(critical_atms) > 0:
        st.write("### Cajeros en Estado Crítico")
        
        # Formatear para mejor visualización
        display_df = critical_atms[['id', 'name', 'current_cash', 'min_threshold', 'days_until_empty', 'last_restock']].copy()
        display_df['current_cash'] = display_df['current_cash'].apply(lambda x: f"${x:,.0f}")
        display_df['min_threshold'] = display_df['min_threshold'].apply(lambda x: f"${x:,.0f}")
        
        # Renombrar columnas
        display_df.columns = ['ID', 'Nombre', 'Efectivo Actual', 'Umbral Mínimo', 'Días hasta Agotamiento', 'Último Reabastecimiento']
        
        st.dataframe(display_df, use_container_width=True)

def demand_forecast_chart(predictions_df, atm_id, atm_info, currency="COP", exchange_rate=None):
    """
    Visualiza la predicción de demanda y nivel de efectivo para un cajero específico.
    
    Args:
        predictions_df: DataFrame con predicciones para varios cajeros
        atm_id: ID del cajero a visualizar (puede ser string o int)
        atm_info: Información actual del cajero
        currency: Moneda a mostrar ("COP" o "USD")
        exchange_rate: Tasa de cambio COP/USD
    """
    if predictions_df is None or len(predictions_df) == 0:
        st.warning("No hay datos de predicción disponibles.")
        return
    
    # Filtrar predicciones para el cajero seleccionado
    # Manejar tanto IDs numéricos como de texto
    if isinstance(atm_id, str) and not isinstance(predictions_df['atm_id'].iloc[0], str):
        atm_predictions = predictions_df[predictions_df['atm_id'].astype(str) == atm_id].sort_values('date')
    else:
        atm_predictions = predictions_df[predictions_df['atm_id'] == atm_id].sort_values('date')
    
    if len(atm_predictions) == 0:
        st.warning(f"No hay predicciones disponibles para el cajero {atm_id}")
        return
    
    # Preparar datos para la visualización
    dates = pd.to_datetime(atm_predictions['date'])
    demands = atm_predictions['predicted_demand'].values
    
    # Simular nivel de efectivo proyectado
    cash_levels = [atm_info['current_cash']]
    for demand in demands:
        next_level = max(0, cash_levels[-1] - demand)
        cash_levels.append(next_level)
    
    cash_levels = cash_levels[:-1]  # Eliminar el último que es un día extra
    
    # Crear gráfico con Plotly
    fig = go.Figure()
    
    # Convertir valores si se usa USD
    if currency == "USD" and exchange_rate:
        demands_display = demands / exchange_rate
        cash_levels_display = [level / exchange_rate for level in cash_levels]
        threshold_display = atm_info['min_threshold'] / exchange_rate
        currency_suffix = " USD"
    else:
        demands_display = demands
        cash_levels_display = cash_levels
        threshold_display = atm_info['min_threshold']
        currency_suffix = " COP"
    
    # Gráfico de demanda (barras)
    fig.add_trace(go.Bar(
        x=dates,
        y=demands_display,
        name='Demanda Proyectada',
        marker_color='#4e73df'
    ))
    
    # Gráfico de nivel de efectivo (línea)
    fig.add_trace(go.Scatter(
        x=dates,
        y=cash_levels_display,
        mode='lines+markers',
        name='Nivel de Efectivo',
        line=dict(color='#e74a3b', width=3)
    ))
    
    # Añadir línea de umbral mínimo
    fig.add_trace(go.Scatter(
        x=dates,
        y=[threshold_display] * len(dates),
        mode='lines',
        name='Umbral Mínimo',
        line=dict(color='#f6c23e', width=2, dash='dash')
    ))
    
    # Configurar diseño
    fig.update_layout(
        title=f'Pronóstico de Demanda y Nivel de Efectivo - {atm_info["name"]}',
        xaxis_title='Fecha',
        yaxis_title=f'Monto (${currency_suffix})',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    # Formatear eje Y para moneda
    fig.update_yaxes(tickprefix='$', tickformat=',')
    
    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar información adicional
    days_to_empty = atm_predictions['days_until_empty'].iloc[0]
    priority_level = atm_predictions['priority'].iloc[0]
    priority_text = 'Alta' if priority_level == 3 else 'Media' if priority_level == 2 else 'Baja'
    
    # Importar la función de formateo de moneda
    from api.utils.helpers import format_currency
    
    # Formato para métricas con la moneda seleccionada
    current_cash_str = format_currency(atm_info['current_cash'], currency, exchange_rate)
    min_threshold_str = format_currency(atm_info['min_threshold'], currency, exchange_rate)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Efectivo actual", current_cash_str)
    col2.metric("Umbral mínimo", min_threshold_str)
    col3.metric("Días hasta agotamiento", f"{days_to_empty:.1f}", help=f"Prioridad: {priority_text}")