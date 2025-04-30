"""
Página de optimización de rutas.

Este módulo define la página que permite generar y visualizar rutas
optimizadas para el reabastecimiento de cajeros.
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
from api.models.prediction import predict_cash_demand, get_priority_atms
from api.models.optimization import optimize_routes_for_date, simulate_scenario
from api.utils.helpers import format_currency, get_exchange_rate
from frontend.components.maps import route_visualization, atm_status_map
from api.utils.metrics import calculate_savings

# Configuración de la página
st.set_page_config(
    page_title="Optimización de Rutas - ATM Optimizer",
    page_icon="💰",
    layout="wide"
)

# Título
st.title("Optimización de Rutas")
st.markdown("Generación de rutas óptimas para reabastecimiento de cajeros")

# Inicialización de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_optimization_data():
    """Carga o simula datos necesarios para optimización"""
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
        
        return current_status, predictions, carriers_df, atms_df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        # Generar datos de ejemplo como fallback
        atms_df = pd.DataFrame()  # Los datos se generarán en simulate_current_atm_status
        current_status = simulate_current_atm_status(atms_df)
        predictions = predict_cash_demand(current_status, days_ahead=7)
        carriers_df = generate_sample_carriers(3)
        
        return current_status, predictions, carriers_df, current_status

# Cargar datos
current_status, predictions, carriers_df, atms_df = load_optimization_data()

# Sidebar para parámetros
st.sidebar.title("Parámetros de Optimización")

# Selector de fecha
available_dates = sorted(predictions['date'].unique())
selected_date = st.sidebar.selectbox(
    "Fecha para optimización",
    options=available_dates,
    index=0
)

# Selector de transportadora
carrier_options = carriers_df.copy()
carrier_options['label'] = carrier_options.apply(lambda x: f"{x['id']} - {x['name']}", axis=1)
selected_carrier_label = st.sidebar.selectbox(
    "Transportadora",
    options=carrier_options['label'].tolist(),
    index=0
)

selected_carrier_id = selected_carrier_label.split(' - ')[0]

# Selector de prioridad
priority_options = {
    "Todas": 1,
    "Media y Alta": 2,
    "Solo Alta": 3
}
selected_priority = st.sidebar.radio(
    "Prioridad mínima:",
    options=list(priority_options.keys()),
    index=1
)
min_priority = priority_options[selected_priority]

# Selector de vehículos
selected_carrier = carriers_df[carriers_df['id'] == selected_carrier_id].iloc[0]
max_vehicles = selected_carrier['vehicles']
num_vehicles = st.sidebar.slider(
    "Número de vehículos:",
    min_value=1,
    max_value=max_vehicles,
    value=min(3, max_vehicles)
)

# Control adicional para número máximo de cajeros
max_cajeros = st.sidebar.slider(
    "Máximo de cajeros a visitar:",
    min_value=5,
    max_value=50,
    value=15
)

# Botón para ejecutar optimización
run_optimization = st.sidebar.button("Optimizar Rutas", type="primary")

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

# Contenedor para resultados de optimización
if run_optimization:
    with st.spinner("Optimizando rutas..."):
        # Ejecutar optimización
        routes, atms_to_restock, selected_carrier, locations = optimize_routes_for_date(
            predictions_df=predictions,
            date=selected_date,
            carrier_df=carriers_df[carriers_df['id'] == selected_carrier_id],
            atm_df=atms_df,
            min_priority=min_priority,
            num_vehicles=num_vehicles,
            max_atms=max_cajeros
        )
        
        # Guardar resultados en sesión para comparaciones
        st.session_state.last_optimization = {
            'routes': routes,
            'atms_to_restock': atms_to_restock,
            'selected_carrier': selected_carrier,
            'locations': locations,
            'parameters': {
                'date': selected_date,
                'carrier_id': selected_carrier_id,
                'min_priority': min_priority,
                'num_vehicles': num_vehicles
            }
        }
        
        # Mostrar resultados
        if routes and len(routes) > 0:
            st.success(f"Optimización completada. Se generaron {len(routes)} rutas.")
            
            # Métricas básicas
            total_distance = sum(route['distance'] for route in routes)
            total_atms = sum(len(route['route']) - 2 for route in routes)  # -2 para excluir depósito al inicio y final
            
            # Mostrar métricas en columnas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cajeros a visitar", total_atms)
            with col2:
                st.metric("Distancia total", f"{total_distance:.2f} km")
            with col3:
                st.metric("Vehículos utilizados", len(routes))
            
            # Visualizar rutas
            st.write("### Visualización de Rutas Optimizadas")
            route_visualization(locations, routes, selected_carrier, width=800, height=500)
            
            # Tabla de resumen de rutas
            st.write("### Detalle de Rutas")
            
            route_details = []
            for i, route in enumerate(routes):
                atms_in_route = len(route['route']) - 2  # -2 para excluir depósito al inicio y final
                route_details.append({
                    'Ruta': i+1,
                    'Vehículo': f"Vehículo {route['vehicle_id']+1}",
                    'Cajeros': atms_in_route,
                    'Distancia (km)': f"{route['distance']:.2f}",
                    'Secuencia': ' → '.join([locations[idx]['name'] for idx in route['route']])
                })
            
            route_df = pd.DataFrame(route_details)
            st.dataframe(route_df, use_container_width=True)
            
            # Análisis de ahorro
            st.write("### Análisis de Ahorro")
            savings = calculate_savings(routes, atms_to_restock, selected_carrier)
            
            # Mostrar ahorros en columnas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distancia sin optimización", f"{savings['distancia_no_optimizada']:.2f} km")
                st.metric("Ahorro en distancia", f"{savings['ahorro_distancia']:.2f} km", 
                         delta=f"{savings['ahorro_porcentaje']:.1f}%")
            
            with col2:
                ahorro_formatted = format_currency(savings['ahorro_costo'], currency, exchange_rate)
                ahorro_mensual = format_currency(savings['ahorro_mensual'], currency, exchange_rate)
                
                st.metric("Ahorro por día", ahorro_formatted)
                st.metric("Ahorro mensual proyectado", ahorro_mensual)
            
            # Mostrar tabla de cajeros a visitar
            st.write("### Cajeros a Reabastecer")
            if len(atms_to_restock) > 0:
                # Preparar datos para tabla
                display_cols = ['atm_id', 'name', 'current_cash', 'predicted_demand', 'days_until_empty', 'priority']
                display_df = atms_to_restock[display_cols].copy()
                
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
                display_df.columns = ['ID', 'Nombre', 'Efectivo Actual', 'Demanda Estimada', 
                                     'Días hasta Agotamiento', 'Prioridad']
                
                # Ordenar por prioridad y días
                display_df = display_df.sort_values(
                    ['Prioridad', 'Días hasta Agotamiento'], 
                    ascending=[False, True]
                )
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.error("No se pudieron generar rutas. Intenta con diferentes parámetros.")
        
else:
    # Si hay resultados guardados, mostrarlos
    if 'last_optimization' in st.session_state:
        st.info("Mostrando resultados de la última optimización. Usa el botón 'Optimizar Rutas' para actualizar.")
        
        last_opt = st.session_state.last_optimization
        routes = last_opt['routes']
        atms_to_restock = last_opt['atms_to_restock']
        selected_carrier = last_opt['selected_carrier']
        locations = last_opt['locations']
        
        if routes and len(routes) > 0:
            # Métricas básicas
            total_distance = sum(route['distance'] for route in routes)
            total_atms = sum(len(route['route']) - 2 for route in routes)
            
            # Mostrar métricas en columnas 
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cajeros a visitar", total_atms)
            with col2:
                st.metric("Distancia total", f"{total_distance:.2f} km")
            with col3:
                st.metric("Vehículos utilizados", len(routes))
            
            # Visualizar rutas
            st.write("### Visualización de Rutas Optimizadas")
            route_visualization(locations, routes, selected_carrier, width=800, height=500)
            
            # Tabla de resumen de rutas
            st.write("### Detalle de Rutas")
            
            route_details = []
            for i, route in enumerate(routes):
                atms_in_route = len(route['route']) - 2  # -2 para excluir depósito al inicio y final
                route_details.append({
                    'Ruta': i+1,
                    'Vehículo': f"Vehículo {route['vehicle_id']+1}",
                    'Cajeros': atms_in_route,
                    'Distancia (km)': route['distance'],  # Quitar el formateo aquí
                    'Secuencia': ' → '.join([str(locations[idx]['name']) for idx in route['route']])
                })
            
            route_df = pd.DataFrame(route_details)
            route_df['Distancia (km)'] = route_df['Distancia (km)'].apply(lambda x: f"{x:.2f}")
            st.dataframe(route_df, use_container_width=True)
            
            # Análisis de ahorro
            st.write("### Análisis de Ahorro")
            savings = calculate_savings(routes, atms_to_restock, selected_carrier)
            
            # Mostrar ahorros en columnas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distancia sin optimización", f"{savings['distancia_no_optimizada']:.2f} km")
                st.metric("Ahorro en distancia", f"{savings['ahorro_distancia']:.2f} km", 
                         delta=f"{savings['ahorro_porcentaje']:.1f}%")
            
            with col2:
                ahorro_formatted = format_currency(savings['ahorro_costo'], currency, exchange_rate)
                ahorro_mensual = format_currency(savings['ahorro_mensual'], currency, exchange_rate)
                
                st.metric("Ahorro por día", ahorro_formatted)
                st.metric("Ahorro mensual proyectado", ahorro_mensual)
            
            # Mostrar tabla de cajeros a visitar
            st.write("### Cajeros a Reabastecer")
            if len(atms_to_restock) > 0:
                # Preparar datos para tabla
                display_cols = ['atm_id', 'name', 'current_cash', 'predicted_demand', 'days_until_empty', 'priority']
                display_df = atms_to_restock[display_cols].copy()
                
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
                display_df.columns = ['ID', 'Nombre', 'Efectivo Actual', 'Demanda Estimada', 
                                     'Días hasta Agotamiento', 'Prioridad']
                
                # Ordenar por prioridad y días
                display_df = display_df.sort_values(
                    ['Prioridad', 'Días hasta Agotamiento'], 
                    ascending=[False, True]
                )
                
                st.dataframe(display_df, use_container_width=True)
    else:
        # Mostrar instrucciones iniciales si no hay optimización previa
        st.info("Configura los parámetros en el panel lateral y haz clic en 'Optimizar Rutas' para generar rutas óptimas.")
        
        # Mostrar imagen de ejemplo o ilustración
        st.markdown("""
        ### ¿Cómo funciona la optimización de rutas?
        
        El sistema utiliza algoritmos avanzados para encontrar las rutas más eficientes para reabastecer los cajeros automáticos, minimizando la distancia total recorrida mientras se asegura que todos los cajeros prioritarios sean atendidos.
        
        **Beneficios:**
        - Reducción significativa de costos operativos
        - Mejor utilización de vehículos y personal
        - Optimización del tiempo de reabastecimiento
        - Planificación efectiva de recursos
        
        **Parámetros clave:**
        - **Fecha**: Selecciona el día para optimizar las rutas
        - **Prioridad**: Filtra cajeros por nivel de urgencia
        - **Transportadora**: Selecciona la empresa de transporte de valores
        - **Vehículos**: Especifica cuántos vehículos están disponibles
        
        Configura estos parámetros en el panel lateral y haz clic en "Optimizar Rutas" para comenzar.
        """)

# Simulación de escenarios alternativos
st.markdown("---")
st.write("## Simulación de Escenarios Alternativos")

# Interfaz para configurar escenario alternativo
st.write("Compara diferentes escenarios modificando parámetros clave:")

# Parámetros para simulación alternativa
col1, col2 = st.columns(2)

with col1:
    alt_priority_options = {
        "Todas": 1,
        "Media y Alta": 2,
        "Solo Alta": 3
    }
    alt_priority = st.radio(
        "Prioridad en escenario alternativo:",
        options=list(alt_priority_options.keys()),
        index=0 if min_priority != 1 else 2  # Sugerir un valor diferente al principal
    )
    alt_min_priority = alt_priority_options[alt_priority]

with col2:
    alt_num_vehicles = st.slider(
        "Vehículos en escenario alternativo:",
        min_value=1,
        max_value=max_vehicles,
        value=1 if num_vehicles > 1 else 3
    )
    
    alt_max_cajeros = st.slider(
        "Máximo de cajeros en escenario alternativo:",
        min_value=5,
        max_value=50,
        value=max(5, max_cajeros - 5)  # ejemplo: menor que el base
    )


# Botón para simular escenario alternativo
simulate_alt = st.button("Simular Escenario Alternativo", type="secondary")

# Mostrar resultados de simulación alternativa
if simulate_alt:
    with st.spinner("Simulando escenario alternativo..."):
        # Ejecutar simulación alternativa
        alt_routes, alt_atms, alt_carrier, alt_locations = simulate_scenario(
            predictions_df=predictions,
            atm_df=current_status,
            carrier_df=carriers_df[carriers_df['id'] == selected_carrier_id],
            date=selected_date,
            num_vehicles=alt_num_vehicles,
            min_priority=alt_min_priority,
            max_atms=alt_max_cajeros
        )
        
        # Guardar resultados en sesión
        st.session_state.alt_scenario = {
            'routes': alt_routes,
            'atms_to_restock': alt_atms,
            'selected_carrier': alt_carrier,
            'locations': alt_locations,
            'parameters': {
                'date': selected_date,
                'carrier_id': selected_carrier_id,
                'min_priority': alt_min_priority,
                'num_vehicles': alt_num_vehicles
            }
        }
        
        # Mostrar comparación si hay resultados en ambos escenarios
        if 'last_optimization' in st.session_state and alt_routes and len(alt_routes) > 0:
            st.success("Simulación alternativa completada. Comparando con escenario base.")
            
            # Extraer datos del escenario base
            base_routes = st.session_state.last_optimization['routes']
            base_atms = st.session_state.last_optimization['atms_to_restock']
            
            # Comparar métricas clave
            base_distance = sum(route['distance'] for route in base_routes)
            base_atms_count = sum(len(route['route']) - 2 for route in base_routes)
            
            alt_distance = sum(route['distance'] for route in alt_routes)
            alt_atms_count = sum(len(route['route']) - 2 for route in alt_routes)
            
            # Calcular diferencias
            distance_diff = alt_distance - base_distance
            distance_pct = (distance_diff / base_distance * 100) if base_distance > 0 else 0
            
            atms_diff = alt_atms_count - base_atms_count
            
            # Mostrar comparación
            st.write("### Comparación de Escenarios")
            
            # Comparar columnas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cajeros visitados", 
                          f"{alt_atms_count} vs {base_atms_count}", 
                          delta=f"{atms_diff:+d}")
            
            with col2:
                st.metric("Distancia total", 
                          f"{alt_distance:.2f} km vs {base_distance:.2f} km", 
                          delta=f"{distance_diff:+.2f} km ({distance_pct:+.1f}%)",
                          delta_color="inverse")
            
            with col3:
                st.metric("Vehículos utilizados", 
                          f"{len(alt_routes)} vs {len(base_routes)}", 
                          delta=f"{len(alt_routes) - len(base_routes):+d}")
            
            # Visualizar rutas alternativas
            st.write("### Rutas en Escenario Alternativo")
            route_visualization(alt_locations, alt_routes, alt_carrier, width=800, height=500)
            
            # Análisis de costo-beneficio
            st.write("### Análisis de Costo-Beneficio")
            
            # Calcular ahorros para ambos escenarios
            base_savings = calculate_savings(base_routes, base_atms, st.session_state.last_optimization['selected_carrier'])
            alt_savings = calculate_savings(alt_routes, alt_atms, alt_carrier)
            
            # Comparar ahorros
            savings_diff = alt_savings['ahorro_mensual'] - base_savings['ahorro_mensual']
            
            # Formatear montos
            base_savings_fmt = format_currency(base_savings['ahorro_mensual'], currency, exchange_rate)
            alt_savings_fmt = format_currency(alt_savings['ahorro_mensual'], currency, exchange_rate)
            diff_fmt = format_currency(savings_diff, currency, exchange_rate)
            
            # Mostrar comparación
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ahorro mensual escenario base", base_savings_fmt)
                st.metric("Ahorro mensual escenario alternativo", alt_savings_fmt)
            
            with col2:
                st.metric("Diferencia en ahorro", diff_fmt, 
                          delta=f"{(savings_diff / base_savings['ahorro_mensual'] * 100):+.1f}%" if base_savings['ahorro_mensual'] > 0 else "N/A")
                
                # Añadir conclusión
                if savings_diff > 0:
                    st.success("El escenario alternativo genera mayor ahorro.")
                elif savings_diff < 0:
                    st.error("El escenario base genera mayor ahorro.")
                else:
                    st.info("Ambos escenarios generan ahorros similares.")
            
            # Tabla comparativa detallada (expandible)
            with st.expander("Ver comparación detallada"):
                # Función auxiliar para formatear valores
                def format_value(value, metric):
                    if value is None:
                        return "N/A"
                    if "Distancia" in metric:
                        return f"{value:.2f}"
                    if "Ahorro" in metric:
                        return format_currency(value, currency, exchange_rate)
                    return str(value)
                
                # Crear tabla de comparación
                comparison_data = [
                    {"Métrica": "Cajeros visitados", "Escenario Base": base_atms_count, "Escenario Alternativo": alt_atms_count, "Diferencia": atms_diff},
                    {"Métrica": "Distancia total (km)", "Escenario Base": base_distance, "Escenario Alternativo": alt_distance, "Diferencia": distance_diff},
                    {"Métrica": "Vehículos utilizados", "Escenario Base": len(base_routes), "Escenario Alternativo": len(alt_routes), "Diferencia": len(alt_routes) - len(base_routes)},
                    {"Métrica": "Prioridad mínima", "Escenario Base": min_priority, "Escenario Alternativo": alt_min_priority, "Diferencia": None},
                    {"Métrica": "Ahorro diario", "Escenario Base": base_savings['ahorro_costo'], "Escenario Alternativo": alt_savings['ahorro_costo'], "Diferencia": alt_savings['ahorro_costo'] - base_savings['ahorro_costo']},
                    {"Métrica": "Ahorro mensual", "Escenario Base": base_savings['ahorro_mensual'], "Escenario Alternativo": alt_savings['ahorro_mensual'], "Diferencia": savings_diff},
                ]
                
                comparison_df = pd.DataFrame(comparison_data)

                for i, row in comparison_df.iterrows():
                    for col in ["Escenario Base", "Escenario Alternativo", "Diferencia"]:
                        comparison_df.at[i, col] = format_value(row[col], row["Métrica"])

                st.dataframe(comparison_df, use_container_width=True)

                # Añadir notas explicativas
                st.markdown("""
                **Notas:**
                - Los valores positivos en la columna "Diferencia" indican un aumento en el escenario alternativo.
                - El ahorro mensual se calcula asumiendo 20 días operativos por mes.
                - La distancia se calcula usando fórmulas de geolocalización entre los puntos.
                """)
        
        elif alt_routes and len(alt_routes) > 0:
            st.success("Simulación alternativa completada.")
            
            # Mostrar métricas básicas
            total_distance = sum(route['distance'] for route in alt_routes)
            total_atms = sum(len(route['route']) - 2 for route in alt_routes)
            
            # Mostrar métricas en columnas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cajeros a visitar", total_atms)
            with col2:
                st.metric("Distancia total", f"{total_distance:.2f} km")
            with col3:
                st.metric("Vehículos utilizados", len(alt_routes))
            
            # Visualizar rutas alternativas
            st.write("### Rutas en Escenario Alternativo")
            route_visualization(alt_locations, alt_routes, alt_carrier, width=800, height=500)
            
            # Análisis de ahorro
            st.write("### Análisis de Ahorro")
            savings = calculate_savings(alt_routes, alt_atms, alt_carrier)
            
            # Mostrar ahorros en columnas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distancia sin optimización", f"{savings['distancia_no_optimizada']:.2f} km")
                st.metric("Ahorro en distancia", f"{savings['ahorro_distancia']:.2f} km", 
                         delta=f"{savings['ahorro_porcentaje']:.1f}%")
            
            with col2:
                ahorro_formatted = format_currency(savings['ahorro_costo'], currency, exchange_rate)
                ahorro_mensual = format_currency(savings['ahorro_mensual'], currency, exchange_rate)
                
                st.metric("Ahorro por día", ahorro_formatted)
                st.metric("Ahorro mensual proyectado", ahorro_mensual)
        else:
            st.error("No se pudieron generar rutas para el escenario alternativo. Intenta con diferentes parámetros.")