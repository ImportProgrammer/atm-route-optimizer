"""
Simulador avanzado de cajeros automáticos.

Este módulo implementa un simulador de eventos discretos para modelar
y analizar escenarios complejos de gestión de efectivo en cajeros automáticos,
considerando patrones de consumo, seguridad y optimización financiera.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import random
from scipy import stats

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
    page_title="Simulador Avanzado - ATM Optimizer",
    page_icon="💰",
    layout="wide"
)

# Título
st.title("Simulador Avanzado de Cajeros")
st.markdown("Simulación de escenarios complejos para optimización de efectivo y rutas")

# Inicialización de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_simulation_data():
    """Carga o simula datos necesarios para la simulación"""
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
        
        # Generar predicciones base
        predictions = predict_cash_demand(current_status, days_ahead=7)
        
        return current_status, predictions, carriers_df, restrictions_df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        # Generar datos de ejemplo como fallback
        atms_df = pd.DataFrame()
        current_status = simulate_current_atm_status(atms_df)
        predictions = predict_cash_demand(current_status, days_ahead=7)
        carriers_df = generate_sample_carriers(3)
        
        # Generar restricciones de ejemplo
        restrictions_df = pd.DataFrame({
            'zone_id': ['Z1', 'Z2', 'Z3', 'Z4', 'Z5'],
            'zone_name': ['Centro', 'Norte', 'Sur', 'Occidente', 'Oriente'],
            'risk_level': ['Alto', 'Bajo', 'Medio', 'Bajo', 'Medio'],
            'allowed_start_time': [time(8, 0), time(6, 0), time(7, 0), time(6, 0), time(7, 0)],
            'allowed_end_time': [time(15, 0), time(20, 0), time(18, 0), time(20, 0), time(19, 0)],
            'max_cash_transport': [100000000, 300000000, 200000000, 300000000, 200000000]
        })
        
        return current_status, predictions, carriers_df, restrictions_df

# Cargar datos
current_status, predictions_base, carriers_df, restrictions_df = load_simulation_data()

# Definir pestañas para los diferentes componentes del simulador
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Configuración de Simulación", 
    "Simulación de Demanda", 
    "Simulación de Seguridad",
    "Optimización Financiera", 
    "Predicción de Disponibilidad Técnica"
])


# Variables de sesión para mantener estado entre pestañas
if 'simulation_params' not in st.session_state:
    st.session_state.simulation_params = {
        'run_simulation': False,
        'simulation_days': 7,
        'demand_pattern': 'Normal',
        'risk_scenario': 'Normal',
        'atms_selected': [],
        'date_selected': datetime.now().strftime('%Y-%m-%d'),
        'simulation_results': {},
        'technical_components': ['Dispensador', 'Lector de Tarjetas', 'Teclado', 'Monitor', 'Lector Biométrico'],
        'technical_scenario': 'Normal'
    }

#####################################################
# PESTAÑA 1: CONFIGURACIÓN DE SIMULACIÓN
#####################################################
with tab1:
    st.header("⚙️ Configuración de Simulación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Parámetros Generales")
        
        # Horizonte de simulación
        sim_days = st.slider(
            "Horizonte de simulación (días)",
            min_value=1,
            max_value=30,
            value=st.session_state.simulation_params['simulation_days'],
            help="Número de días a simular"
        )
        st.session_state.simulation_params['simulation_days'] = sim_days
        
        # Fecha inicial
        start_date = st.date_input(
            "Fecha de inicio",
            value=datetime.strptime(st.session_state.simulation_params['date_selected'], '%Y-%m-%d'),
            help="Fecha de inicio de la simulación"
        )
        st.session_state.simulation_params['date_selected'] = start_date.strftime('%Y-%m-%d')
        
        # Selección de cajeros para simulación detallada
        if len(current_status) > 0:
            atm_options = current_status[['id', 'name', 'location_type']].copy()
            atm_options['label'] = atm_options.apply(
                lambda x: f"{x['id']} - {x['name']} ({x['location_type']})", axis=1
            )
            
            default_atms = st.session_state.simulation_params['atms_selected'] or atm_options['id'].iloc[:3].tolist()
            
            selected_atms = st.multiselect(
                "Cajeros para análisis detallado",
                options=atm_options['label'].tolist(),
                default=[opt for opt in atm_options['label'].tolist() if opt.split(' - ')[0] in default_atms],
                help="Seleccione cajeros específicos para analizar en detalle"
            )
            
            # Extraer IDs de cajeros seleccionados
            st.session_state.simulation_params['atms_selected'] = [atm.split(' - ')[0] for atm in selected_atms]
        
    with col2:
        st.write("### Escenarios de Simulación")
        
        # Patrones de demanda
        demand_patterns = {
            'Normal': 'Patrón regular basado en históricos',
            'Alta Volatilidad': 'Mayor variabilidad en patrones de retiro',
            'Temporada Alta': 'Aumento general de demanda (vacaciones, fiestas)',
            'Quincena/Fin de Mes': 'Picos pronunciados en fechas de pago',
            'Evento Especial': 'Simulación de evento local que aumenta demanda'
        }
        
        selected_pattern = st.selectbox(
            "Patrón de demanda",
            options=list(demand_patterns.keys()),
            index=list(demand_patterns.keys()).index(st.session_state.simulation_params['demand_pattern']),
            help="Seleccione el patrón de comportamiento de la demanda"
        )
        st.session_state.simulation_params['demand_pattern'] = selected_pattern
        
        st.caption(demand_patterns[selected_pattern])
        
        # Escenarios de riesgo
        risk_scenarios = {
            'Normal': 'Condiciones estándar de seguridad',
            'Alto Riesgo': 'Aumento generalizado de inseguridad',
            'Temporada Festiva': 'Mayor actividad delictiva en fechas especiales',
            'Zonas Específicas': 'Focos de inseguridad en áreas concretas',
            'Crisis de Seguridad': 'Escenario extremo con restricciones severas'
        }
        
        selected_risk = st.selectbox(
            "Escenario de seguridad",
            options=list(risk_scenarios.keys()),
            index=list(risk_scenarios.keys()).index(st.session_state.simulation_params['risk_scenario']),
            help="Seleccione el escenario de seguridad para la simulación"
        )
        st.session_state.simulation_params['risk_scenario'] = selected_risk
        
        st.caption(risk_scenarios[selected_risk])
        
        # Añadir escenarios técnicos
        technical_scenarios = {
            'Normal': 'Tasas de fallas estándar basadas en datos históricos',
            'Alta Frecuencia': 'Mayor frecuencia de fallas técnicas en todos los componentes',
            'Componentes Críticos': 'Fallas concentradas en dispensador y lector de tarjetas',
            'Problemas Ambientales': 'Fallas relacionadas con factores externos (temperatura, humedad)',
            'Obsolescencia': 'Fallas por equipos antiguos o con mantenimiento deficiente'
        }
        
        selected_technical = st.selectbox(
            "Escenario técnico",
            options=list(technical_scenarios.keys()),
            index=list(technical_scenarios.keys()).index(st.session_state.simulation_params.get('technical_scenario', 'Normal')),
            help="Seleccione el escenario de fallas técnicas para la simulación"
        )
        st.session_state.simulation_params['technical_scenario'] = selected_technical
        
        st.caption(technical_scenarios[selected_technical])
        
        # Configuración de moneda
        st.write("### Configuración")
        currency = st.radio(
            "Moneda:",
            options=["COP", "USD"],
            index=0,
            horizontal=True
        )
        
        # Obtener tasa de cambio si es necesario
        exchange_rate = None
        if currency == "USD":
            exchange_rate = get_exchange_rate()
            st.info(f"Tasa de cambio: 1 USD = {exchange_rate:,.2f} COP")
    
    # Botón para ejecutar simulación
    run_simulation = st.button("Ejecutar Simulación", type="primary")
    
    # Si se solicita ejecutar simulación
    if run_simulation:
        st.session_state.simulation_params['run_simulation'] = True
        
        with st.spinner("Ejecutando simulación..."):
            # Aquí ejecutaríamos la simulación completa que afectaría a todas las pestañas
            # Por ahora usaremos funciones simuladas para cada pestaña
            
            # Función para simular escenario de demanda con patrones variables
            def simulate_demand_with_patterns(base_predictions, pattern, days, start_date_str):
                """
                Simula la demanda de efectivo con diferentes patrones.
                
                Args:
                    base_predictions: Predicciones base
                    pattern: Patrón de demanda seleccionado
                    days: Número de días a simular
                    start_date_str: Fecha inicial en formato string
                
                Returns:
                    DataFrame con simulación de demanda
                """
                # Crear fechas para la simulación
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                dates = [start_date + timedelta(days=i) for i in range(days)]
                
                # Obtener cajeros únicos
                atms = base_predictions['atm_id'].unique()
                
                # Crear dataframe de simulación
                sim_rows = []
                
                for atm_id in atms:
                    # Obtener datos base del cajero
                    atm_base = base_predictions[base_predictions['atm_id'] == atm_id].iloc[0]
                    base_demand = atm_base['predicted_demand']
                    
                    for i, date in enumerate(dates):
                        # Factores base por día de semana (mayor en fin de semana)
                        weekday = date.weekday()
                        weekday_factor = 1.0 + (0.2 if weekday >= 5 else 0.0)
                        
                        # Factor de quincena/fin de mes
                        day_of_month = date.day
                        payday_factor = 1.0
                        if day_of_month in [15, 16, 30, 31, 1]:
                            payday_factor = 1.3
                        
                        # Volatilidad base
                        base_volatility = 0.1  # 10% de variación aleatoria
                        
                        # Ajustar según patrón seleccionado
                        if pattern == 'Normal':
                            # Patrón normal con ligera variación
                            volatility = base_volatility
                            event_factor = 1.0
                            seasonal_factor = 1.0
                        
                        elif pattern == 'Alta Volatilidad':
                            # Mayor variabilidad en la demanda
                            volatility = base_volatility * 3.0
                            event_factor = 1.0
                            seasonal_factor = 1.0
                            
                        elif pattern == 'Temporada Alta':
                            # Aumento general por temporada
                            volatility = base_volatility * 1.5
                            event_factor = 1.0
                            seasonal_factor = 1.4
                            
                        elif pattern == 'Quincena/Fin de Mes':
                            # Mayor efecto en días de pago
                            volatility = base_volatility * 1.2
                            event_factor = 1.0
                            payday_factor = payday_factor * 1.5
                            seasonal_factor = 1.0
                            
                        elif pattern == 'Evento Especial':
                            # Evento especial en ciertos días
                            volatility = base_volatility * 2.0
                            # Evento en días 3, 4 y 5 de la simulación
                            event_days = [3, 4, 5]
                            event_factor = 1.5 if i in event_days else 1.0
                            seasonal_factor = 1.0
                        
                        # Calcular demanda ajustada
                        random_factor = np.random.normal(1.0, volatility)
                        adjusted_demand = base_demand * weekday_factor * payday_factor * seasonal_factor * event_factor * random_factor
                        
                        # Asegurarse de que no sea negativa
                        adjusted_demand = max(0, adjusted_demand)
                        
                        # Datos adicionales para el cajero y fecha
                        atm_data = base_predictions[base_predictions['atm_id'] == atm_id].iloc[0].to_dict()
                        
                        # Crear fila de simulación
                        sim_row = {
                            'atm_id': atm_id,
                            'date': date,
                            'predicted_demand': adjusted_demand,
                            'base_demand': base_demand,
                            'weekday_factor': weekday_factor,
                            'payday_factor': payday_factor,
                            'seasonal_factor': seasonal_factor,
                            'event_factor': event_factor,
                            'random_factor': random_factor,
                            'name': atm_data.get('name', f'ATM {atm_id}'),
                            'location_type': atm_data.get('location_type', 'Unknown'),
                            'current_cash': atm_data.get('current_cash', 0)
                        }
                        
                        sim_rows.append(sim_row)
                
                # Crear DataFrame con todas las simulaciones
                sim_df = pd.DataFrame(sim_rows)
                
                # Calcular niveles de efectivo proyectados y días hasta agotamiento
                atm_groups = sim_df.groupby('atm_id')
                
                # Función para calcular niveles de efectivo y días hasta agotamiento
                def calc_cash_levels(group):
                    group = group.sort_values('date')
                    
                    # Obtener efectivo inicial y calcular niveles
                    initial_cash = group['current_cash'].iloc[0]
                    demands = group['predicted_demand'].values
                    
                    # Calcular nivel de efectivo para cada día
                    cash_level = initial_cash
                    cash_levels = []
                    days_until_empty = None
                    
                    for i, demand in enumerate(demands):
                        cash_level = max(0, cash_level - demand)
                        cash_levels.append(cash_level)
                        
                        # Determinar días hasta agotamiento
                        if cash_level <= 0 and days_until_empty is None:
                            days_until_empty = i
                    
                    # Si nunca se agota, establecer al máximo
                    if days_until_empty is None:
                        days_until_empty = len(demands)
                    
                    # Asignar a cada fila
                    group['projected_cash'] = [initial_cash] + cash_levels[:-1]
                    group['days_until_empty'] = days_until_empty
                    
                    # Calcular prioridad basada en días hasta agotamiento
                    group['priority'] = 3  # Alta por defecto
                    group.loc[group['days_until_empty'] > 3, 'priority'] = 2  # Media
                    group.loc[group['days_until_empty'] > 7, 'priority'] = 1  # Baja
                    
                    return group
                
                # Aplicar cálculos a cada grupo de cajero
                sim_df = atm_groups.apply(calc_cash_levels).reset_index(drop=True)
                
                return sim_df
            
            # Función para simular escenarios de seguridad
            def simulate_security_scenarios(atms_df, risk_scenario):
                """
                Simula diferentes escenarios de seguridad.
                
                Args:
                    atms_df: DataFrame con información de cajeros
                    risk_scenario: Escenario de seguridad seleccionado
                
                Returns:
                    DataFrame con métricas de seguridad
                """

                ## print("Columnas disponibles:", atms_df.columns.tolist())
                # Crear DataFrame para métricas de seguridad
                security_df = atms_df[['id', 'name', 'location_type', 'latitude', 'longitude']].drop_duplicates('id')
                security_df = security_df.rename(columns={'id': 'atm_id'})
                
                # Generar zonas de seguridad si no existen
                if 'zone_id' not in security_df.columns:
                    # Asignar zonas basadas en ubicación (simplificado)
                    lats = security_df['latitude']
                    lons = security_df['longitude']
                    
                    # Dividir en cuadrantes básicos
                    lat_median = lats.median()
                    lon_median = lons.median()
                    
                    conditions = [
                        (lats >= lat_median) & (lons >= lon_median),
                        (lats >= lat_median) & (lons < lon_median),
                        (lats < lat_median) & (lons >= lon_median),
                        (lats < lat_median) & (lons < lon_median)
                    ]
                    
                    zones = ['Norte-Este', 'Norte-Oeste', 'Sur-Este', 'Sur-Oeste']
                    zone_ids = ['Z1', 'Z2', 'Z3', 'Z4']
                    
                    security_df['zone'] = np.select(conditions, zones, default='Centro')
                    security_df['zone_id'] = np.select(conditions, zone_ids, default='Z5')
                
                # Métricas base de seguridad
                security_df['base_risk_score'] = np.random.uniform(1, 10, len(security_df))
                
                # Ajustar según tipo de ubicación
                location_risk = {
                    'Centro Comercial': 0.8,
                    'Supermercado': 0.9,
                    'Calle': 1.3,
                    'Estación': 1.2,
                    'Comercio': 1.0,
                    'Oficina': 0.7,
                    'Universidad': 0.8,
                    'Hospital': 0.9
                }
                
                # Aplicar factor de ubicación
                security_df['location_factor'] = security_df['location_type'].map(
                    lambda x: location_risk.get(x, 1.0)
                )
                
                # Ajustar según escenario seleccionado
                if risk_scenario == 'Normal':
                    # Escenario normal
                    security_df['scenario_factor'] = 1.0
                    security_df['time_restriction'] = 'Normal'
                    
                elif risk_scenario == 'Alto Riesgo':
                    # Escenario de alto riesgo general
                    security_df['scenario_factor'] = 1.5
                    security_df['time_restriction'] = 'Reducido'
                    
                elif risk_scenario == 'Temporada Festiva':
                    # Mayor riesgo en temporada festiva
                    security_df['scenario_factor'] = 1.3
                    security_df['time_restriction'] = 'Festivo'
                    
                elif risk_scenario == 'Zonas Específicas':
                    # Riesgo focalizado en zonas específicas
                    high_risk_zones = ['Z1', 'Z3']  # Ejemplo de zonas de alto riesgo
                    
                    security_df['scenario_factor'] = security_df['zone_id'].apply(
                        lambda z: 2.0 if z in high_risk_zones else 1.0
                    )
                    security_df['time_restriction'] = security_df['zone_id'].apply(
                        lambda z: 'Severo' if z in high_risk_zones else 'Normal'
                    )
                    
                elif risk_scenario == 'Crisis de Seguridad':
                    # Escenario extremo
                    security_df['scenario_factor'] = 2.5
                    security_df['time_restriction'] = 'Crítico'
                
                # Calcular riesgo final
                security_df['risk_score'] = security_df['base_risk_score'] * security_df['location_factor'] * security_df['scenario_factor']
                
                # Clasificar nivel de riesgo
                security_df['risk_level'] = pd.cut(
                    security_df['risk_score'],
                    bins=[0, 5, 10, 15, 100],
                    labels=['Bajo', 'Medio', 'Alto', 'Extremo']
                )
                
                # Generar horarios permitidos basados en restricciones
                time_windows = {
                    'Normal': {'start': '07:00', 'end': '19:00'},
                    'Reducido': {'start': '08:00', 'end': '17:00'},
                    'Festivo': {'start': '09:00', 'end': '16:00'},
                    'Severo': {'start': '10:00', 'end': '15:00'},
                    'Crítico': {'start': '11:00', 'end': '14:00'}
                }
                
                security_df['allowed_start'] = security_df['time_restriction'].map(lambda x: time_windows[x]['start'])
                security_df['allowed_end'] = security_df['time_restriction'].map(lambda x: time_windows[x]['end'])
                
                # Probabilidad de incidente por nivel de riesgo
                incident_prob = {
                    'Bajo': 0.01,
                    'Medio': 0.03,
                    'Alto': 0.08,
                    'Extremo': 0.15
                }
                
                security_df['incident_probability'] = security_df['risk_level'].map(incident_prob)
                
                # Costo estimado de incidente (basado en cantidad de efectivo)
                security_df['incident_cost_factor'] = np.random.uniform(0.5, 1.5, len(security_df))
                
                return security_df
            
            # Función para simular optimización financiera
            def simulate_financial_optimization(demand_df, security_df):
                """
                Simula optimización financiera basada en demanda y seguridad.
                
                Args:
                    demand_df: DataFrame con simulación de demanda
                    security_df: DataFrame con métricas de seguridad
                
                Returns:
                    DataFrame con métricas financieras
                """
                # Crear DataFrame para análisis financiero
                atm_ids = demand_df['atm_id'].unique()
                finance_rows = []
                
                for atm_id in atm_ids:
                    # Obtener datos del cajero
                    atm_demand = demand_df[demand_df['atm_id'] == atm_id]
                    atm_security = security_df[security_df['atm_id'] == atm_id].iloc[0]
                    
                    # Datos base
                    atm_name = atm_demand['name'].iloc[0]
                    current_cash = atm_demand['current_cash'].iloc[0]
                    location_type = atm_demand['location_type'].iloc[0]
                    
                    # Calcular demanda promedio y volatilidad
                    avg_demand = atm_demand['predicted_demand'].mean()
                    demand_volatility = atm_demand['predicted_demand'].std() / avg_demand if avg_demand > 0 else 0
                    
                    # Calcular nivel óptimo de efectivo basado en demanda y seguridad
                    # Fórmula simplificada: demanda promedio * días de cobertura + margen de seguridad
                    risk_factor = 1.0 + (0.1 * atm_security['risk_score'] / 10)  # Mayor riesgo -> menor efectivo
                    volatility_factor = 1.0 + (demand_volatility * 2)  # Mayor volatilidad -> más efectivo
                    
                    # Días de cobertura deseados
                    coverage_days = 5  # Base
                    
                    # Ajustar cobertura según tipo de ubicación
                    location_coverage = {
                        'Centro Comercial': 1.2,
                        'Supermercado': 1.1,
                        'Calle': 0.8,
                        'Estación': 0.9,
                        'Comercio': 1.0,
                        'Oficina': 1.1,
                        'Universidad': 1.0,
                        'Hospital': 1.1
                    }
                    
                    location_factor = location_coverage.get(location_type, 1.0)
                    coverage_days *= location_factor
                    
                    # Nivel óptimo calculado
                    optimal_cash_level = (avg_demand * coverage_days * volatility_factor) / risk_factor
                    
                    # Calcular costos operativos
                    # Costo de oportunidad del capital inmovilizado (tasa anual)
                    interest_rate = 0.12  # 12% anual
                    daily_rate = interest_rate / 365
                    
                    # Costo diario del capital inmovilizado
                    capital_cost = current_cash * daily_rate
                    
                    # Costo del transporte de valores (fijo + variable por monto)
                    # Esto se calcularía mejor con la información de rutas optimizadas
                    transport_base_cost = 500000  # COP por visita
                    transport_variable_cost = current_cash * 0.0001  # 0.01% del monto
                    
                    transport_cost = transport_base_cost + transport_variable_cost
                    
                    # Costo de agotamiento (pérdida de transacciones, reputación)
                    # Mayor si el cajero está en una ubicación de alto tráfico
                    stockout_traffic_factor = {
                        'Centro Comercial': 2.0,
                        'Supermercado': 1.8,
                        'Calle': 1.5,
                        'Estación': 1.7,
                        'Comercio': 1.3,
                        'Oficina': 1.0,
                        'Universidad': 1.2,
                        'Hospital': 1.4
                    }
                    
                    traffic_factor = stockout_traffic_factor.get(location_type, 1.0)
                    
                    # Costo base por día de agotamiento
                    stockout_base_cost = 2000000  # COP por día
                    stockout_cost = stockout_base_cost * traffic_factor
                    
                    # Probabilidad de agotamiento
                    days_to_empty = atm_demand['days_until_empty'].iloc[0]
                    if days_to_empty < len(atm_demand):
                        stockout_prob = 1.0
                    else:
                        stockout_prob = 0.1  # Base low probability
                    
                    expected_stockout_cost = stockout_cost * stockout_prob
                    
                    # Costo total
                    total_cost = capital_cost + transport_cost + expected_stockout_cost
                    
                    # Eficiencia actual vs óptima
                    current_efficiency = avg_demand / current_cash if current_cash > 0 else 0
                    optimal_efficiency = avg_demand / optimal_cash_level if optimal_cash_level > 0 else 0
                    
                    # Crear fila para este cajero
                    finance_row = {
                        'atm_id': atm_id,
                        'name': atm_name,
                        'current_cash': current_cash,
                        'avg_demand': avg_demand,
                        'demand_volatility': demand_volatility,
                        'optimal_cash_level': optimal_cash_level,
                        'current_vs_optimal': current_cash / optimal_cash_level if optimal_cash_level > 0 else 0,
                        'capital_cost': capital_cost,
                        'transport_cost': transport_cost,
                        'stockout_cost': expected_stockout_cost,
                        'total_cost': total_cost,
                        'current_efficiency': current_efficiency,
                        'optimal_efficiency': optimal_efficiency,
                        'days_to_empty': days_to_empty,
                        'risk_score': atm_security['risk_score'],
                        'risk_level': atm_security['risk_level']
                    }
                    
                    finance_rows.append(finance_row)
                
                # Crear DataFrame financiero
                finance_df = pd.DataFrame(finance_rows)
                
                return finance_df
            
            # Función para simular datos de disponibilidad técnica
            def simulate_technical_availability(atms_df, technical_scenario, risk_scenario, demand_pattern, days):
                """
                Simula la disponibilidad técnica de los cajeros automáticos.
                
                Args:
                    atms_df: DataFrame con información de cajeros
                    technical_scenario: Escenario técnico seleccionado
                    risk_scenario: Escenario de seguridad
                    demand_pattern: Patrón de demanda
                    days: Número de días a simular
                    
                Returns:
                    DataFrame con métricas de disponibilidad técnica
                """
                # Crear DataFrame base con ATMs
                tech_df = atms_df[['id', 'name', 'location_type', 'latitude', 'longitude']].drop_duplicates('id')
                tech_df = tech_df.rename(columns={'id': 'atm_id'})
                
                # Definir componentes técnicos y sus códigos
                components = {
                    'Dispensador': 'DSP',
                    'Lector de Tarjetas': 'CRD',
                    'Teclado': 'EPP',
                    'Monitor': 'MT',
                    'Lector Biométrico': 'BRD'
                }
                
                # Definir tasas base de falla por componente (eventos por año)
                base_failure_rates = {
                    'Dispensador': 3.0,        # 3 fallas por año en promedio
                    'Lector de Tarjetas': 2.0,  # 2 fallas por año en promedio
                    'Teclado': 1.0,             # 1 falla por año en promedio
                    'Monitor': 0.5,             # 0.5 fallas por año en promedio
                    'Lector Biométrico': 1.5    # 1.5 fallas por año en promedio
                }
                
                # Definir MTTR (Mean Time To Repair) base por componente en horas
                base_mttr = {
                    'Dispensador': 6,        # 6 horas en promedio
                    'Lector de Tarjetas': 4,  # 4 horas en promedio
                    'Teclado': 3,             # 3 horas en promedio
                    'Monitor': 2,             # 2 horas en promedio
                    'Lector Biométrico': 4    # 4 horas en promedio
                }
                
                # Ajustar tasas de falla según escenario técnico
                scenario_factors = {
                    'Normal': {'factor': 1.0, 'bias': None},
                    'Alta Frecuencia': {'factor': 2.5, 'bias': None},
                    'Componentes Críticos': {
                        'factor': 1.0,
                        'bias': {
                            'Dispensador': 3.0,
                            'Lector de Tarjetas': 2.5,
                            'Teclado': 1.0,
                            'Monitor': 1.0,
                            'Lector Biométrico': 1.0
                        }
                    },
                    'Problemas Ambientales': {'factor': 1.5, 'bias': None},
                    'Obsolescencia': {'factor': 2.0, 'bias': None}
                }
                
                # Ajustar MTTR según escenario de seguridad
                risk_mttr_factors = {
                    'Normal': 1.0,
                    'Alto Riesgo': 1.5,     # Toma más tiempo reparar en zonas de alto riesgo
                    'Temporada Festiva': 1.3,
                    'Zonas Específicas': 1.2,
                    'Crisis de Seguridad': 2.0  # Tiempo de reparación se duplica en crisis
                }
                
                # Ajustar tasas de falla según patrón de demanda (mayor uso -> más fallas)
                demand_factors = {
                    'Normal': 1.0,
                    'Alta Volatilidad': 1.2,
                    'Temporada Alta': 1.4,
                    'Quincena/Fin de Mes': 1.3,
                    'Evento Especial': 1.5
                }
                
                # Simular datos para cada ATM
                tech_results = []
                
                for _, atm in tech_df.iterrows():
                    atm_id = atm['atm_id']
                    atm_name = atm['name']
                    atm_type = atm['location_type']
                    
                    # Factor de ajuste por tipo de ubicación
                    location_factor = {
                        'Centro Comercial': 0.8,   # Mejor ambiente, menos fallas
                        'Supermercado': 0.9,
                        'Calle': 1.3,              # Peor ambiente, más fallas
                        'Estación': 1.2,
                        'Comercio': 1.0,
                        'Oficina': 0.7,            # Ambiente controlado, menos fallas
                        'Universidad': 0.8,
                        'Hospital': 0.9
                    }.get(atm_type, 1.0)
                    
                    # Generar datos para cada componente
                    for component, code in components.items():
                        # Obtener tasa base de falla para este componente
                        base_rate = base_failure_rates[component]
                        
                        # Aplicar factores de ajuste
                        scenario_factor = scenario_factors[technical_scenario]['factor']
                        
                        # Aplicar bias específico si existe
                        scenario_bias = 1.0
                        if scenario_factors[technical_scenario]['bias'] and component in scenario_factors[technical_scenario]['bias']:
                            scenario_bias = scenario_factors[technical_scenario]['bias'][component]
                        
                        demand_factor = demand_factors[demand_pattern]
                        
                        # Calcular tasa ajustada (convertir de anual a diaria)
                        adjusted_rate = (base_rate * scenario_factor * scenario_bias * location_factor * demand_factor) / 365.0
                        
                        # Probabilidad de falla en el período de simulación
                        failure_prob = 1 - np.exp(-adjusted_rate * days)
                        
                        # Número esperado de fallas en el período
                        expected_failures = adjusted_rate * days
                        
                        # MTBF (Mean Time Between Failures) en días
                        mtbf = 1 / adjusted_rate if adjusted_rate > 0 else float('inf')
                        
                        # MTTR ajustado por factor de riesgo
                        mttr_hours = base_mttr[component] * risk_mttr_factors[risk_scenario]
                        
                        # Disponibilidad técnica (porcentaje de tiempo operativo)
                        availability = mtbf / (mtbf + (mttr_hours / 24)) if mtbf != float('inf') else 1.0
                        
                        # Calcular código específico de falla (simulado)
                        failure_code = f"{code}:{np.random.randint(0, 2):02d}:{np.random.randint(0, 5):02d}:{np.random.randint(0, 9):02d}"
                        
                        # Añadir registro para este ATM y componente
                        tech_results.append({
                            'atm_id': atm_id,
                            'name': atm_name,
                            'location_type': atm_type,
                            'component': component,
                            'component_code': code,
                            'failure_probability': failure_prob,
                            'expected_failures': expected_failures,
                            'mtbf_days': mtbf,
                            'mttr_hours': mttr_hours,
                            'technical_availability': availability * 100,  # Convertir a porcentaje
                            'failure_code': failure_code,
                            'latitude': atm['latitude'],
                            'longitude': atm['longitude'],
                            'risk_factor': risk_mttr_factors[risk_scenario],
                            'next_failure_days': np.random.exponential(mtbf) if mtbf != float('inf') else 999
                        })
                
                # Crear DataFrame con todos los resultados
                tech_results_df = pd.DataFrame(tech_results)
                
                return tech_results_df
            
            # Ejecutar simulaciones
            with st.spinner("Ejecutando simulaciones de demanda, seguridad, financiera y técnica..."):
                # Simulación de demanda
                demand_simulation = simulate_demand_with_patterns(
                    predictions_base, 
                    st.session_state.simulation_params['demand_pattern'],
                    st.session_state.simulation_params['simulation_days'],
                    st.session_state.simulation_params['date_selected']
                )
                
                # Simulación de seguridad
                security_simulation = simulate_security_scenarios(
                    current_status,
                    st.session_state.simulation_params['risk_scenario']
                )
                
                # Simulación financiera
                financial_simulation = simulate_financial_optimization(
                    demand_simulation,
                    security_simulation
                )
            
                # Simulación técnica
                technical_simulation = simulate_technical_availability(
                    current_status,
                    st.session_state.simulation_params['technical_scenario'],
                    st.session_state.simulation_params['risk_scenario'],
                    st.session_state.simulation_params['demand_pattern'],
                    st.session_state.simulation_params['simulation_days']
                )
                
                # Guardar resultados en sesión
                st.session_state.simulation_params['simulation_results'] = {
                    'demand': demand_simulation,
                    'security': security_simulation,
                    'financial': financial_simulation,
                    'technical': technical_simulation,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.success("¡Simulación completada! Explore los resultados en las pestañas correspondientes.")
    
    # Si ya hay una simulación previa, mostrar un resumen
    results = st.session_state.simulation_params.get('simulation_results', {})
    if results and 'timestamp' in results:
        st.success(f"Simulación anterior disponible (generada: {results['timestamp']})")    
        # Mostrar un resumen breve
        st.write("### Resumen de la simulación")
        
        # Crear métricas de resumen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Total de cajeros que se agotan en el periodo
            depleted_atms = results['demand'][results['demand']['days_until_empty'] < st.session_state.simulation_params['simulation_days']]['atm_id'].nunique()
            total_atms = results['demand']['atm_id'].nunique()
            
            st.metric(
                "Cajeros con agotamiento",
                f"{depleted_atms} / {total_atms}",
                help="Cajeros que se agotan durante el período de simulación"
            )
        
        with col2:
            # Promedio de nivel de riesgo
            risk_levels = {
                'Bajo': 1,
                'Medio': 2,
                'Alto': 3,
                'Extremo': 4
            }
            risk_values = results['security']['risk_level'].astype(str).map(risk_levels)
            avg_risk = risk_values.mean()
            
            st.metric(
                "Nivel de riesgo promedio",
                f"{avg_risk:.2f} / 4",
                help="Nivel de riesgo promedio de todos los cajeros (escala 1-4)"
            )
        
        with col3:
            # Eficiencia financiera promedio
            avg_efficiency = results['financial']['current_efficiency'].mean() * 100
            avg_optimal = results['financial']['optimal_efficiency'].mean() * 100
            efficiency_gap = avg_optimal - avg_efficiency
            
            st.metric(
                "Eficiencia financiera",
                f"{avg_efficiency:.1f}%",
                delta=f"{efficiency_gap:+.1f}% al óptimo",
                help="Eficiencia actual vs nivel óptimo calculado"
            )
    
    # Si no hay simulación, mostrar instrucciones
    else:
        st.info("Configure los parámetros de simulación y haga clic en 'Ejecutar Simulación' para comenzar.")

#####################################################
# PESTAÑA 2: SIMULACIÓN DE DEMANDA
#####################################################
with tab2:
    st.header("📈 Simulación de Demanda de Efectivo")
    
    # Verificar si hay resultados de simulación
    # if 'simulation_results' in st.session_state.simulation_params:
    #     results = st.session_state.simulation_params['simulation_results']
    #     demand_sim = results['demand']
    results = st.session_state.simulation_params.get('simulation_results', {})
    if results and 'demand' in results:
        demand_sim = results['demand']
        # Mostrar pestañas para diferentes visualizaciones
        demand_tab1, demand_tab2, demand_tab3 = st.tabs([
            "Patrones Temporales", 
            "Análisis por Cajero", 
            "Distribuciones"
        ])
        
        with demand_tab1:
            st.write("### Patrones Temporales de Demanda")
            
            # Demanda total por día
            daily_demand = demand_sim.groupby('date')['predicted_demand'].sum().reset_index()
            daily_demand['day_of_week'] = daily_demand['date'].dt.day_name()
            daily_demand['is_weekend'] = daily_demand['date'].dt.weekday >= 5
            daily_demand['day_of_month'] = daily_demand['date'].dt.day
            daily_demand['is_payday'] = daily_demand['day_of_month'].isin([15, 16, 30, 31, 1])
            
            # Gráfico de demanda diaria
            fig = px.bar(
                daily_demand,
                x='date',
                y='predicted_demand',
                color='is_weekend',
                color_discrete_map={True: '#1cc88a', False: '#4e73df'},
                labels={'predicted_demand': 'Demanda Total', 'date': 'Fecha', 'is_weekend': 'Fin de Semana'},
                title=f'Demanda Total Diaria - Patrón: {st.session_state.simulation_params["demand_pattern"]}'
            )
            
            # Resaltar días de pago
            for i, row in daily_demand[daily_demand['is_payday']].iterrows():
                fig.add_annotation(
                    x=row['date'],
                    y=row['predicted_demand'] * 1.05,
                    text="Día de Pago",
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor="#e74a3b",
                    arrowsize=1,
                    arrowwidth=2
                )
            
            # Formatear eje Y para moneda
            if currency == "USD" and exchange_rate:
                fig.update_layout(yaxis=dict(
                    tickprefix='$', 
                    ticksuffix=' USD')
                    )
                # Convertir valores
                fig.update_traces(y=daily_demand['predicted_demand'] / exchange_rate)
            else:
                fig.update_layout(yaxis=dict(
                    tickprefix='$', 
                    ticksuffix=' COP')
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Patrones horarios (simulados)
            st.write("### Patrones Horarios (Simulación)")
            
            # Generar datos horarios simulados
            hours = list(range(24))
            
            # Perfiles horarios según tipo de ubicación
            hourly_profiles = {
                'Centro Comercial': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.08, 0.10, 0.09, 0.06, 0.04, 0.03, 0.02, 0.01],
                'Supermercado': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.10, 0.09, 0.08, 0.07, 0.08, 0.09, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01],
                'Calle': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.07, 0.08, 0.07, 0.08, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01],
                'Estación': [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.09, 0.10, 0.08, 0.06, 0.06, 0.07, 0.06, 0.06, 0.07, 0.09, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.01],
                'Oficina': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.09, 0.10, 0.09, 0.08, 0.09, 0.09, 0.10, 0.09, 0.07, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01],
                'Universidad': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.07, 0.08, 0.09, 0.09, 0.10, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01]
            }
            
            # Perfiles por defecto para otros tipos
            default_profile = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
            
            # Obtener tipos de ubicación únicos
            location_types = current_status['location_type'].unique()
            
            # Crear gráfico de patrones horarios
            fig = go.Figure()
            
            for loc_type in location_types:
                # Obtener perfil para este tipo o usar el predeterminado
                profile = hourly_profiles.get(loc_type, default_profile)
                
                # Añadir línea para este tipo
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=profile,
                    mode='lines',
                    name=loc_type
                ))
            
            # Configurar diseño
            fig.update_layout(
                title='Patrones Horarios por Tipo de Ubicación',
                xaxis_title='Hora del Día',
                yaxis_title='Fracción de Demanda Diaria',
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with demand_tab2:
            st.write("### Análisis por Cajero")
            
            # Selector de cajeros para análisis
            if 'atms_selected' in st.session_state.simulation_params and st.session_state.simulation_params['atms_selected']:
                selected_atms = st.session_state.simulation_params['atms_selected']
            else:
                # Seleccionar los primeros cajeros como muestra
                selected_atms = demand_sim['atm_id'].unique()[:3].tolist()
            
            # Crear diccionario de nombres de cajeros
            atm_names = demand_sim[['atm_id', 'name']].drop_duplicates().set_index('atm_id')['name'].to_dict()
            
            # Selector interactivo
            selected_atm_id = st.selectbox(
                "Seleccione un cajero para análisis detallado:",
                options=selected_atms,
                format_func=lambda x: f"{x} - {atm_names.get(x, 'Desconocido')}"
            )
            
            # Filtrar datos para el cajero seleccionado
            atm_data = demand_sim[demand_sim['atm_id'] == selected_atm_id].sort_values('date')
            
            # Obtener datos financieros y de seguridad para este cajero
            atm_finance = results['financial'][results['financial']['atm_id'] == selected_atm_id].iloc[0]
            atm_security = results['security'][results['security']['atm_id'] == selected_atm_id].iloc[0]
            
            # Mostrar métricas clave
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_cash = atm_data['current_cash'].iloc[0]
                formatted_cash = format_currency(current_cash, currency, exchange_rate)
                
                st.metric(
                    "Efectivo Actual",
                    formatted_cash,
                    help="Cantidad de efectivo disponible al inicio de la simulación"
                )
            
            with col2:
                avg_demand = atm_finance['avg_demand']
                formatted_demand = format_currency(avg_demand, currency, exchange_rate)
                
                st.metric(
                    "Demanda Diaria Promedio",
                    formatted_demand,
                    help="Demanda diaria promedio durante el período de simulación"
                )
            
            with col3:
                days_to_empty = atm_finance['days_to_empty']
                
                st.metric(
                    "Días hasta Agotamiento",
                    f"{days_to_empty:.1f} días",
                    help="Días estimados hasta agotamiento del efectivo"
                )
            
            # Gráfico de demanda vs nivel de efectivo
            st.write("#### Proyección de Demanda y Efectivo")
            
            # Crear gráfico
            fig = go.Figure()
            
            # Convertir valores para mostrar
            if currency == "USD" and exchange_rate:
                demand_values = atm_data['predicted_demand'] / exchange_rate
                cash_values = [current_cash / exchange_rate]
                
                # Calcular niveles de efectivo proyectados
                for demand in demand_values:
                    next_level = max(0, cash_values[-1] - demand)
                    cash_values.append(next_level)
                
                cash_values = cash_values[:-1]  # Quitar el último nivel extra
                
                currency_suffix = " USD"
            else:
                demand_values = atm_data['predicted_demand']
                cash_values = [current_cash]
                
                # Calcular niveles de efectivo proyectados
                for demand in demand_values:
                    next_level = max(0, cash_values[-1] - demand)
                    cash_values.append(next_level)
                
                cash_values = cash_values[:-1]  # Quitar el último nivel extra
                
                currency_suffix = " COP"
            
            # Gráfico de barras para demanda
            fig.add_trace(go.Bar(
                x=atm_data['date'],
                y=demand_values,
                name='Demanda Diaria',
                marker_color='#4e73df',
                opacity=0.7
            ))
            
            # Gráfico de línea para nivel de efectivo
            fig.add_trace(go.Scatter(
                x=atm_data['date'],
                y=cash_values,
                mode='lines+markers',
                name='Nivel de Efectivo',
                line=dict(color='#e74a3b', width=3)
            ))
            
            # Nivel óptimo de efectivo
            optimal_level = atm_finance['optimal_cash_level']
            if currency == "USD" and exchange_rate:
                optimal_level /= exchange_rate
            
            fig.add_trace(go.Scatter(
                x=atm_data['date'],
                y=[optimal_level] * len(atm_data),
                mode='lines',
                name='Nivel Óptimo',
                line=dict(color='#1cc88a', width=2, dash='dash')
            ))
            
            # Configurar diseño
            fig.update_layout(
                title=f'Proyección para {atm_names.get(selected_atm_id, "Desconocido")}',
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Información adicional
            st.write("#### Análisis de Factores")
            
            # Mostrar factores
            col1, col2 = st.columns(2)
            
            with col1:
                # Mostrar factores de demanda
                st.write("##### Factores de Demanda")
                
                factors_df = pd.DataFrame({
                    'Factor': [
                        'Patrón de Demanda',
                        'Volatilidad',
                        'Ubicación',
                        'Días hasta Agotamiento',
                        'Eficiencia de Capital'
                    ],
                    'Valor': [
                        st.session_state.simulation_params['demand_pattern'],
                        f"{atm_finance['demand_volatility']:.2f}",
                        atm_data['location_type'].iloc[0],
                        f"{atm_finance['days_to_empty']:.1f} días",
                        f"{atm_finance['current_efficiency']*100:.1f}%"
                    ]
                })
                
                st.dataframe(factors_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Mostrar factores de riesgo
                st.write("##### Factores de Riesgo")
                
                risk_df = pd.DataFrame({
                    'Factor': [
                        'Escenario de Riesgo',
                        'Nivel de Riesgo',
                        'Zona',
                        'Ventana de Tiempo',
                        'Probabilidad de Incidente'
                    ],
                    'Valor': [
                        st.session_state.simulation_params['risk_scenario'],
                        atm_security['risk_level'],
                        atm_security.get('zone', 'No especificada'),
                        f"{atm_security.get('allowed_start', '07:00')} - {atm_security.get('allowed_end', '19:00')}",
                        f"{atm_security.get('incident_probability', 0.05)*100:.1f}%"
                    ]
                })
                
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            # Análisis de costos
            st.write("#### Análisis de Costos")
            
            # Convertir valores para mostrar
            capital_cost = atm_finance['capital_cost']
            transport_cost = atm_finance['transport_cost']
            stockout_cost = atm_finance['stockout_cost']
            total_cost = atm_finance['total_cost']
            
            if currency == "USD" and exchange_rate:
                capital_cost /= exchange_rate
                transport_cost /= exchange_rate
                stockout_cost /= exchange_rate
                total_cost /= exchange_rate
                currency_suffix = "USD"
            else:
                currency_suffix = "COP"
            
            # Crear gráfico de descomposición de costos
            cost_data = pd.DataFrame({
                'Componente': ['Capital Inmovilizado', 'Transporte de Valores', 'Agotamiento (Esperado)'],
                'Costo': [capital_cost, transport_cost, stockout_cost]
            })
            
            fig = px.pie(
                cost_data, 
                values='Costo', 
                names='Componente',
                title=f'Composición de Costos Diarios (Total: ${total_cost:,.0f} {currency_suffix})',
                color_discrete_sequence=['#4e73df', '#1cc88a', '#e74a3b']
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            
            st.plotly_chart(fig, use_container_width=True)
            
        with demand_tab3:
            st.write("### Distribuciones y Tendencias")
            
            # Histograma de demanda diaria
            st.write("#### Distribución de Demanda Diaria por Cajero")
            
            # Agrupar por atm_id y calcular estadísticas
            atm_stats = demand_sim.groupby('atm_id')['predicted_demand'].agg(['mean', 'std', 'min', 'max']).reset_index()
            atm_stats['cv'] = atm_stats['std'] / atm_stats['mean']  # Coeficiente de variación
            
            # Unir con nombre para mejor visualización
            atm_stats = pd.merge(atm_stats, demand_sim[['atm_id', 'name']].drop_duplicates(), on='atm_id')
            
            # Convertir valores si es necesario
            if currency == "USD" and exchange_rate:
                for col in ['mean', 'std', 'min', 'max']:
                    atm_stats[col] = atm_stats[col] / exchange_rate
                currency_suffix = "USD"
            else:
                currency_suffix = "COP"
            
            # Gráfico de rangos de demanda
            fig = go.Figure()
            
            # Ordenar por demanda media
            atm_stats = atm_stats.sort_values('mean', ascending=False)
            
            # Añadir barras de error
            fig.add_trace(go.Bar(
                x=atm_stats['name'],
                y=atm_stats['mean'],
                error_y=dict(
                    type='data',
                    array=atm_stats['std'],
                    visible=True
                ),
                name='Demanda Media',
                marker_color='#4e73df'
            ))
            
            # Configurar diseño
            fig.update_layout(
                title='Demanda Media y Desviación por Cajero',
                xaxis_title='Cajero',
                yaxis_title=f'Demanda Diaria (${currency_suffix})',
                xaxis={'categoryorder':'total descending'},
                height=500
            )
            
            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de volatilidad
            st.write("#### Análisis de Volatilidad")
            
            # Gráfico de dispersión: demanda media vs coeficiente de variación
            fig = px.scatter(
                atm_stats,
                x='mean',
                y='cv',
                color='cv',
                size='mean',
                hover_name='name',
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={
                    'mean': f'Demanda Media (${currency_suffix})',
                    'cv': 'Coeficiente de Variación',
                    'name': 'Cajero'
                },
                title='Relación entre Demanda Media y Volatilidad'
            )
            
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title='Volatilidad'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de correlación con factores externos
            st.write("#### Simulación Monte Carlo de Agotamiento")
            
            # Generar simulación Monte Carlo para probabilidad de agotamiento
            @st.cache_data
            def monte_carlo_depletion(n_simulations=1000):
                """Simula agotamiento de cajeros con Monte Carlo"""
                selected_atms = demand_sim['atm_id'].unique()[:5].tolist()  # Limitamos a 5 para el ejemplo
                
                results = []
                
                for atm_id in selected_atms:
                    atm_data = demand_sim[demand_sim['atm_id'] == atm_id].iloc[0]
                    current_cash = atm_data['current_cash']
                    avg_demand = results['financial'][results['financial']['atm_id'] == atm_id].iloc[0]['avg_demand']
                    volatility = results['financial'][results['financial']['atm_id'] == atm_id].iloc[0]['demand_volatility']
                    
                    # Simular días hasta agotamiento
                    depletion_days = []
                    
                    for _ in range(n_simulations):
                        cash = current_cash
                        days = 0
                        
                        while cash > 0 and days < 30:  # Límite de 30 días
                            # Generar demanda diaria con distribución normal
                            daily_demand = np.random.normal(avg_demand, avg_demand * volatility)
                            daily_demand = max(0, daily_demand)  # No puede ser negativa
                            
                            cash -= daily_demand
                            days += 1
                        
                        depletion_days.append(days)
                    
                    # Calcular probabilidades
                    probabilities = {}
                    for day in range(1, 31):
                        prob = sum(1 for d in depletion_days if d <= day) / n_simulations
                        probabilities[day] = prob
                    
                    # Añadir a resultados
                    results.append({
                        'atm_id': atm_id,
                        'name': atm_data['name'],
                        'probabilities': probabilities
                    })
                
                return results
            
            # Mostrar solo si hay una selección de cajeros
            if 'atms_selected' in st.session_state.simulation_params and st.session_state.simulation_params['atms_selected']:
                # Generar datos simulados para los gráficos
                # En una implementación real, usaríamos monte_carlo_depletion()
                
                selected_atms = st.session_state.simulation_params['atms_selected']
                
                # Para esta demo, generamos datos simulados directamente
                mc_results = []
                
                for atm_id in selected_atms:
                    atm_data = demand_sim[demand_sim['atm_id'] == atm_id].iloc[0]
                    days_to_empty = results['financial'][results['financial']['atm_id'] == atm_id].iloc[0]['days_to_empty']
                    
                    # Generar curva de probabilidad basada en días hasta agotamiento
                    probabilities = {}
                    for day in range(1, 31):
                        if days_to_empty < 30:
                            # Distribución logística centrada en days_to_empty
                            x = (day - days_to_empty) / 2
                            prob = 1 / (1 + np.exp(-x))
                        else:
                            # Cajero no se agota en el periodo
                            scale = 0.1
                            prob = day * scale
                            if prob > 1:
                                prob = 1
                        
                        probabilities[day] = prob
                    
                    mc_results.append({
                        'atm_id': atm_id,
                        'name': atm_data['name'],
                        'probabilities': probabilities
                    })
                
                # Crear gráfico de probabilidades de agotamiento
                fig = go.Figure()
                
                for result in mc_results:
                    days = list(result['probabilities'].keys())
                    probs = list(result['probabilities'].values())
                    
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=probs,
                        mode='lines',
                        name=result['name']
                    ))
                
                # Configurar diseño
                fig.update_layout(
                    title='Probabilidad de Agotamiento por Día',
                    xaxis_title='Día',
                    yaxis_title='Probabilidad de Agotamiento',
                    yaxis=dict(tickformat='.0%'),
                    xaxis=dict(tickmode='linear', tick0=0, dtick=5),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir explicación
                st.caption("""
                El gráfico muestra la probabilidad de que el cajero se quede sin efectivo en o antes de un día específico.
                Esto permite evaluar el riesgo de agotamiento y planificar el reabastecimiento de manera más precisa.
                """)
            else:
                st.info("Seleccione cajeros en la pestaña de configuración para ver la simulación Monte Carlo.")
            
    else:
        st.info("Ejecute una simulación en la pestaña de Configuración para ver resultados de demanda.")

#####################################################
# PESTAÑA 3: SIMULACIÓN DE SEGURIDAD
#####################################################
with tab3:
    st.header("🔒 Simulación de Seguridad")
    
    # Verificar si hay resultados de simulación
    # if 'simulation_results' in st.session_state.simulation_params:
    #     results = st.session_state.simulation_params['simulation_results']
    #     security_sim = results['security']
    results = st.session_state.simulation_params.get('simulation_results', {})
    if results and 'security' in results:
        security_sim = results['security']    

        # Mostrar pestañas para diferentes visualizaciones
        security_tab1, security_tab2, security_tab3 = st.tabs([
            "Mapa de Riesgo", 
            "Ventanas de Tiempo", 
            "Análisis de Incidentes"
        ])
        
        with security_tab1:
            st.write("### Mapa de Niveles de Riesgo")
            
            # Crear mapa de calor de riesgo
            risk_map = folium.Map(
                location=[security_sim['latitude'].mean(), security_sim['longitude'].mean()],
                zoom_start=12
            )
            
            # Colores según nivel de riesgo
            risk_colors = {
                'Bajo': 'green',
                'Medio': 'blue',
                'Alto': 'orange',
                'Extremo': 'red'
            }
            
            # Añadir marcadores para cada cajero
            for _, row in security_sim.iterrows():
                # Determinar color según nivel de riesgo
                color = risk_colors.get(row['risk_level'], 'gray')
                
                # Crear popup con información detallada
                popup_html = f"""
                    <div style="width: 200px">
                        <h4>{row['name']}</h4>
                        <b>ID:</b> {row['atm_id']}<br>
                        <b>Nivel de Riesgo:</b> {row['risk_level']}<br>
                        <b>Score:</b> {row['risk_score']:.1f}<br>
                        <b>Zona:</b> {row.get('zone', 'N/A')}<br>
                        <b>Ventana de Tiempo:</b> {row.get('allowed_start', '07:00')} - {row.get('allowed_end', '19:00')}<br>
                        <b>Probabilidad de Incidente:</b> {row.get('incident_probability', 0)*100:.1f}%
                    </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row['name']} - Riesgo: {row['risk_level']}",
                    icon=folium.Icon(color=color, icon='money-bill-alt', prefix='fa')
                ).add_to(risk_map)
            
            # Renderizar mapa
            st.write("Distribución geográfica de niveles de riesgo:")
            folium_static(risk_map, width=800, height=500)
            
            # Mostrar estadísticas de riesgo
            st.write("### Distribución de Niveles de Riesgo")
            
            # Contar cajeros por nivel de riesgo
            risk_counts = security_sim['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['Nivel de Riesgo', 'Cantidad']
            
            # Ordenar por nivel de riesgo
            risk_order = {'Bajo': 0, 'Medio': 1, 'Alto': 2, 'Extremo': 3}
            risk_counts['Orden'] = risk_counts['Nivel de Riesgo'].map(risk_order)
            risk_counts = risk_counts.sort_values('Orden').drop('Orden', axis=1)
            
            # Crear gráfico
            fig = px.bar(
                risk_counts,
                x='Nivel de Riesgo',
                y='Cantidad',
                color='Nivel de Riesgo',
                color_discrete_map={
                    'Bajo': '#1cc88a',
                    'Medio': '#4e73df',
                    'Alto': '#f6c23e',
                    'Extremo': '#e74a3b'
                },
                text='Cantidad',
                title=f'Distribución de Cajeros por Nivel de Riesgo - {st.session_state.simulation_params["risk_scenario"]}'
            )
            
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Cantidad de Cajeros',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with security_tab2:
            st.write("### Ventanas de Tiempo Seguras")
            
            # Mostrar ventanas de tiempo por zona
            if 'zone' in security_sim.columns and 'allowed_start' in security_sim.columns:
                # Agrupar por zona y obtener ventanas de tiempo
                zone_windows = security_sim.groupby('zone')[['allowed_start', 'allowed_end']].first().reset_index()
                
                # Crear gráfico de ventanas de tiempo
                fig = go.Figure()
                
                # Horas del día
                hours = list(range(24))
                hour_labels = [f"{h:02d}:00" for h in hours]
                
                # Añadir una línea para cada zona
                for _, row in zone_windows.iterrows():
                    # Convertir horarios a índices de hora
                    start_hour = int(row['allowed_start'].split(':')[0])
                    end_hour = int(row['allowed_end'].split(':')[0])
                    
                    # Crear vector de disponibilidad
                    availability = [0] * 24
                    for h in range(start_hour, end_hour + 1):
                        if h < 24:
                            availability[h] = 1
                    
                    # Añadir línea para esta zona
                    fig.add_trace(go.Scatter(
                        x=hour_labels,
                        y=availability,
                        mode='lines+markers',
                        name=row['zone'],
                        line=dict(width=3)
                    ))
                
                # Configurar diseño
                fig.update_layout(
                    title='Ventanas de Tiempo Seguras por Zona',
                    xaxis_title='Hora del Día',
                    yaxis=dict(
                        title='',
                        tickmode='array',
                        tickvals=[0, 1],
                        ticktext=['No Disponible', 'Disponible']
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar tabla de ventanas de tiempo
                st.write("#### Detalle de Ventanas de Tiempo por Zona")
                st.dataframe(zone_windows, use_container_width=True, hide_index=True)
                
                # Añadir factor limitante
                st.write("#### Factor Limitante de Ventanas de Tiempo")
                
                # Calcular solapamiento entre ventanas
                overlap_start = max([int(t.split(':')[0]) for t in zone_windows['allowed_start']])
                overlap_end = min([int(t.split(':')[0]) for t in zone_windows['allowed_end']])
                
                if overlap_end > overlap_start:
                    overlap_hours = f"{overlap_start:02d}:00 - {overlap_end:02d}:00"
                    st.success(f"Ventana de tiempo óptima para todas las zonas: **{overlap_hours}**")
                else:
                    st.error("No hay solapamiento entre las ventanas de tiempo de todas las zonas.")
                    
                    # Encontrar el mejor solapamiento posible
                    best_overlap = 0
                    best_zones = []
                    
                    # Revisar todas las combinaciones de zonas
                    from itertools import combinations
                    for r in range(2, len(zone_windows) + 1):
                        for combo in combinations(range(len(zone_windows)), r):
                            subset = zone_windows.iloc[list(combo)]
                            sub_start = max([int(t.split(':')[0]) for t in subset['allowed_start']])
                            sub_end = min([int(t.split(':')[0]) for t in subset['allowed_end']])
                            
                            overlap = sub_end - sub_start
                            if overlap > best_overlap and sub_end > sub_start:
                                best_overlap = overlap
                                best_zones = subset['zone'].tolist()
                    
                    if best_overlap > 0:
                        st.warning(f"Mejor solapamiento posible ({best_overlap} horas) en zonas: {', '.join(best_zones)}")
            else:
                st.info("No hay información detallada de ventanas de tiempo en esta simulación.")
            
            # Mostrar recomendaciones de seguridad
            st.write("### Recomendaciones de Seguridad")
            
            # Crear recomendaciones basadas en el escenario
            risk_scenario = st.session_state.simulation_params['risk_scenario']
            
            if risk_scenario == 'Normal':
                recommendations = [
                    "Mantener protocolos estándar de seguridad",
                    "Reabastecimiento en horarios regulares",
                    "Rotación normal de rutas"
                ]
            elif risk_scenario == 'Alto Riesgo':
                recommendations = [
                    "Restringir operaciones a horarios de menor riesgo",
                    "Aumentar escoltas en zonas de alto riesgo",
                    "Reducir montos máximos por vehículo",
                    "Implementar rutas alternativas"
                ]
            elif risk_scenario == 'Temporada Festiva':
                recommendations = [
                    "Ajustar horarios para evitar horas pico",
                    "Reducir visibilidad de operaciones",
                    "Incrementar frecuencia con montos menores",
                    "Coordinar con autoridades en fechas críticas"
                ]
            elif risk_scenario == 'Zonas Específicas':
                recommendations = [
                    "Protocolo especial para zonas de alto riesgo",
                    "Considerar escoltas adicionales solo en zonas críticas",
                    "Mantener operación normal en zonas seguras",
                    "Evaluar rutas alternativas para zonas problemáticas"
                ]
            elif risk_scenario == 'Crisis de Seguridad':
                recommendations = [
                    "Limitar operaciones a lo estrictamente necesario",
                    "Coordinar con fuerzas de seguridad",
                    "Considerar cierre temporal de cajeros en zonas extremas",
                    "Implementar protocolo de emergencia",
                    "Reducir significativamente montos transportados"
                ]
            
            # Mostrar recomendaciones como lista
            for rec in recommendations:
                st.markdown(f"* {rec}")
        
        with security_tab3:
            st.write("### Análisis de Probabilidad de Incidentes")
            
            # Calcular probabilidad total de incidentes
            if 'incident_probability' in security_sim.columns:
                # Probabilidad total (al menos un incidente)
                incident_probs = security_sim['incident_probability'].astype(float)
                total_prob = 1 - np.prod(1 - incident_probs)
                
                # Calcular costo esperado de incidentes
                incident_costs = []
                for _, row in security_sim.iterrows():
                    # Obtener datos financieros para este cajero
                    atm_finance = results['financial'][results['financial']['atm_id'] == row['atm_id']]
                    
                    if len(atm_finance) > 0:
                        current_cash = atm_finance.iloc[0]['current_cash']
                        # Costo del incidente: efectivo + costo operativo + daños
                        incident_cost = current_cash + 5000000  # 5M COP adicionales por daños
                        
                        # Costo esperado
                        expected_cost = incident_cost * row['incident_probability']
                        
                        incident_costs.append({
                            'atm_id': row['atm_id'],
                            'name': row['name'],
                            'risk_level': row['risk_level'],
                            'probability': row['incident_probability'],
                            'incident_cost': incident_cost,
                            'expected_cost': expected_cost
                        })
                
                # Crear DataFrame de costos
                costs_df = pd.DataFrame(incident_costs)
                
                # Convertir a moneda seleccionada
                if currency == "USD" and exchange_rate:
                    for col in ['incident_cost', 'expected_cost']:
                        costs_df[col] = costs_df[col] / exchange_rate
                    currency_suffix = "USD"
                else:
                    currency_suffix = "COP"
                
                # Calcular costo total esperado
                total_expected_cost = costs_df['expected_cost'].sum()
                
                # Mostrar métricas clave
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Probabilidad de al menos un incidente",
                        f"{total_prob:.2%}",
                        help="Probabilidad de que ocurra al menos un incidente en el período"
                    )
                
                with col2:
                    formatted_cost = f"${total_expected_cost:,.0f} {currency_suffix}"
                    st.metric(
                        "Costo esperado de incidentes",
                        formatted_cost,
                        help="Valor esperado del costo de incidentes en el período"
                    )
                
                # Gráfico de probabilidades por nivel de riesgo
                st.write("#### Probabilidad de Incidentes por Nivel de Riesgo")
                
                # Agrupar por nivel de riesgo
                risk_probs = costs_df.groupby('risk_level')['probability'].mean().reset_index()
                
                # Ordenar por nivel de riesgo
                risk_order = {'Bajo': 0, 'Medio': 1, 'Alto': 2, 'Extremo': 3}
                risk_probs['Orden'] = risk_probs['risk_level'].map(risk_order)
                risk_probs = risk_probs.sort_values('Orden').drop('Orden', axis=1)
                
                # Crear gráfico
                fig = px.bar(
                    risk_probs,
                    x='risk_level',
                    y='probability',
                    color='risk_level',
                    color_discrete_map={
                        'Bajo': '#1cc88a',
                        'Medio': '#4e73df',
                        'Alto': '#f6c23e',
                        'Extremo': '#e74a3b'
                    },
                    labels={
                        'risk_level': 'Nivel de Riesgo',
                        'probability': 'Probabilidad de Incidente'
                    },
                    title='Probabilidad de Incidentes por Nivel de Riesgo'
                )
                
                fig.update_layout(
                    xaxis_title='Nivel de Riesgo',
                    yaxis_title='Probabilidad',
                    yaxis=dict(tickformat='.0%'),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar tabla de cajeros con mayor riesgo
                st.write("#### Cajeros con Mayor Riesgo")
                
                # Ordenar por costo esperado
                top_risk = costs_df.sort_values('expected_cost', ascending=False).head(10)
                
                # Formatear para visualización
                display_df = top_risk[['name', 'risk_level', 'probability', 'expected_cost']].copy()
                display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
                display_df['expected_cost'] = display_df['expected_cost'].apply(lambda x: f"${x:,.0f}")
                
                # Renombrar columnas
                display_df.columns = ['Cajero', 'Nivel de Riesgo', 'Probabilidad', f'Costo Esperado ({currency_suffix})']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Análisis de sensibilidad
                st.write("#### Análisis de Sensibilidad")
                
                # Calcular costos esperados para diferentes niveles de seguridad
                security_levels = ['Bajo', 'Medio', 'Alto', 'Extremo']
                security_costs = [500000, 1000000, 2000000, 4000000]  # Costos adicionales de seguridad
                
                # Convertir costos si es necesario
                if currency == "USD" and exchange_rate:
                    security_costs = [cost / exchange_rate for cost in security_costs]
                
                # Factores de reducción de probabilidad por nivel de seguridad
                reduction_factors = [0.1, 0.3, 0.6, 0.9]  # Reducción de probabilidad
                
                # Calcular costos esperados con diferentes niveles
                sensitivity_data = []
                
                for level, cost, factor in zip(security_levels, security_costs, reduction_factors):
                    # Probabilidad ajustada
                    adjusted_prob = total_prob * (1 - factor)
                    
                    # Costo ajustado
                    adjusted_cost = total_expected_cost * (1 - factor)
                    
                    # Costo total (costo esperado + costo de seguridad)
                    total_cost = adjusted_cost + cost
                    
                    # Ahorro neto (reducción de costo esperado - costo de seguridad)
                    savings = (total_expected_cost - adjusted_cost) - cost
                    
                    sensitivity_data.append({
                        'Nivel': level,
                        'Costo de Seguridad': cost,
                        'Probabilidad Ajustada': adjusted_prob,
                        'Costo Esperado Ajustado': adjusted_cost,
                        'Costo Total': total_cost,
                        'Ahorro Neto': savings
                    })
                
                # Crear DataFrame de sensibilidad
                sensitivity_df = pd.DataFrame(sensitivity_data)
                
                # Formatear para visualización
                display_cols = ['Nivel', 'Probabilidad Ajustada', 'Costo Total', 'Ahorro Neto']
                display_df = sensitivity_df[display_cols].copy()
                display_df['Probabilidad Ajustada'] = display_df['Probabilidad Ajustada'].apply(lambda x: f"{x:.2%}")
                display_df['Costo Total'] = display_df['Costo Total'].apply(lambda x: f"${x:,.0f}")
                display_df['Ahorro Neto'] = display_df['Ahorro Neto'].apply(lambda x: f"${x:,.0f}")
                
                # Mostrar tabla
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Encontrar nivel óptimo
                optimal_level = sensitivity_df.loc[sensitivity_df['Ahorro Neto'].idxmax()]
                
                # Mostrar recomendación
                st.success(f"Nivel de seguridad óptimo recomendado: **{optimal_level['Nivel']}**")
                st.caption(f"Este nivel proporciona un ahorro neto de ${optimal_level['Ahorro Neto']:,.0f} {currency_suffix}")
    else:
        st.info("Ejecute una simulación en la pestaña de Configuración para ver resultados de seguridad.")

#####################################################
# PESTAÑA 4: OPTIMIZACIÓN FINANCIERA
#####################################################
with tab4:
    st.header("💰 Optimización Financiera")
    
    # Verificar si hay resultados de simulación
    # if 'simulation_results' in st.session_state.simulation_params:
    #     results = st.session_state.simulation_params['simulation_results']
    #     financial_sim = results['financial']
    results = st.session_state.simulation_params.get('simulation_results', {})

    if results and 'financial' in results:
        financial_sim = results['financial']    
        # Mostrar pestañas para diferentes visualizaciones
        finance_tab1, finance_tab2, finance_tab3 = st.tabs([
            "Optimización de Efectivo", 
            "Análisis de Costos", 
            "Recomendaciones"
        ])
        
        with finance_tab1:
            st.write("### Optimización de Niveles de Efectivo")
            
            # Comparación de niveles actuales vs óptimos
            # Convertir a moneda seleccionada
            if currency == "USD" and exchange_rate:
                financial_sim['current_cash_display'] = financial_sim['current_cash'] / exchange_rate
                financial_sim['optimal_cash_display'] = financial_sim['optimal_cash_level'] / exchange_rate
                currency_suffix = "USD"
            else:
                financial_sim['current_cash_display'] = financial_sim['current_cash']
                financial_sim['optimal_cash_display'] = financial_sim['optimal_cash_level']
                currency_suffix = "COP"
            
            # Calcular diferencias
            financial_sim['cash_difference'] = financial_sim['current_cash_display'] - financial_sim['optimal_cash_display']
            financial_sim['cash_difference_pct'] = (financial_sim['cash_difference'] / financial_sim['optimal_cash_display']) * 100
            
            # Gráfico de barras comparativo
            fig = go.Figure()
            
            # Ordenar por diferencia absoluta
            sorted_data = financial_sim.sort_values('cash_difference', key=abs, ascending=False).head(10)
            
            # Añadir barras para nivel actual
            fig.add_trace(go.Bar(
                x=sorted_data['name'],
                y=sorted_data['current_cash_display'],
                name='Nivel Actual',
                marker_color='#4e73df'
            ))
            
            # Añadir barras para nivel óptimo
            fig.add_trace(go.Bar(
                x=sorted_data['name'],
                y=sorted_data['optimal_cash_display'],
                name='Nivel Óptimo',
                marker_color='#1cc88a'
            ))
            
            # Configurar diseño
            fig.update_layout(
                title='Comparación de Niveles de Efectivo: Actual vs. Óptimo (Top 10 por diferencia)',
                xaxis_title='Cajero',
                yaxis_title=f'Nivel de Efectivo (${currency_suffix})',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Resumen general de optimización
            st.write("### Resumen de Optimización")
            
            # Calcular métricas globales
            total_current = financial_sim['current_cash_display'].sum()
            total_optimal = financial_sim['optimal_cash_display'].sum()
            total_difference = total_current - total_optimal
            
            # Mostrar métricas en columnas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                formatted_current = f"${total_current:,.0f} {currency_suffix}"
                st.metric(
                    "Efectivo Total Actual",
                    formatted_current,
                    help="Suma del efectivo actual en todos los cajeros"
                )
            
            with col2:
                formatted_optimal = f"${total_optimal:,.0f} {currency_suffix}"
                st.metric(
                    "Efectivo Total Óptimo",
                    formatted_optimal,
                    help="Suma del nivel óptimo calculado para todos los cajeros"
                )
            
            with col3:
                formatted_diff = f"${abs(total_difference):,.0f} {currency_suffix}"
                diff_pct = (total_difference / total_optimal) * 100 if total_optimal > 0 else 0
                
                if total_difference > 0:
                    # Exceso de efectivo
                    st.metric(
                        "Exceso de Efectivo",
                        formatted_diff,
                        delta=f"{diff_pct:.1f}% sobre el óptimo",
                        delta_color="inverse",
                        help="Cantidad de efectivo que podría reducirse"
                    )
                else:
                    # Déficit de efectivo
                    st.metric(
                        "Déficit de Efectivo",
                        formatted_diff,
                        delta=f"{abs(diff_pct):.1f}% bajo el óptimo",
                        help="Cantidad adicional de efectivo necesaria"
                    )
            
            # Mostrar tabla detallada
            st.write("### Detalle de Optimización por Cajero")
            
            # Preparar datos para tabla
            detail_df = financial_sim[['name', 'current_cash_display', 'optimal_cash_display', 'cash_difference', 'current_efficiency', 'optimal_efficiency']].copy()
            
            # Formatear columnas
            detail_df['current_cash_display'] = detail_df['current_cash_display'].apply(lambda x: f"${x:,.0f}")
            detail_df['optimal_cash_display'] = detail_df['optimal_cash_display'].apply(lambda x: f"${x:,.0f}")
            detail_df['cash_difference'] = detail_df['cash_difference'].apply(lambda x: f"${x:,.0f}")
            detail_df['current_efficiency'] = detail_df['current_efficiency'].apply(lambda x: f"{x*100:.1f}%")
            detail_df['optimal_efficiency'] = detail_df['optimal_efficiency'].apply(lambda x: f"{x*100:.1f}%")
            
            # Renombrar columnas
            detail_df.columns = ['Cajero', f'Efectivo Actual ({currency_suffix})', f'Nivel Óptimo ({currency_suffix})', 
                                 f'Diferencia ({currency_suffix})', 'Eficiencia Actual', 'Eficiencia Óptima']
            
            # Mostrar tabla
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        with finance_tab2:
            st.write("### Análisis de Costos")
            
            # Costos totales por componente
            total_capital = financial_sim['capital_cost'].sum()
            total_transport = financial_sim['transport_cost'].sum()
            total_stockout = financial_sim['stockout_cost'].sum()
            
            # Convertir a moneda seleccionada
            if currency == "USD" and exchange_rate:
                total_capital /= exchange_rate
                total_transport /= exchange_rate
                total_stockout /= exchange_rate
                currency_suffix = "USD"
            else:
                currency_suffix = "COP"
            
            # Calcular total de todos los componentes
            total_all_costs = total_capital + total_transport + total_stockout
            
            # Crear gráfico de pie
            cost_data = pd.DataFrame({
                'Componente': ['Capital Inmovilizado', 'Transporte de Valores', 'Agotamiento (Esperado)'],
                'Costo': [total_capital, total_transport, total_stockout]
            })
            
            fig = px.pie(
                cost_data, 
                values='Costo', 
                names='Componente',
                title=f'Distribución de Costos Diarios (Total: ${total_all_costs:,.0f} {currency_suffix})',
                color_discrete_sequence=['#4e73df', '#1cc88a', '#e74a3b']
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de costos por tipo de ubicación
            st.write("### Costos por Tipo de Ubicación")
            
            # Unir datos de ubicación
            location_costs = financial_sim.groupby('name').agg({
                'total_cost': 'sum',
                'capital_cost': 'sum',
                'transport_cost': 'sum',
                'stockout_cost': 'sum'
            }).reset_index()
            
            # Unir tipo de ubicación
            location_costs = pd.merge(
                location_costs,
                results['demand'][['name', 'location_type']].drop_duplicates(),
                on='name'
            )
            
            # Agrupar por tipo de ubicación
            location_summary = location_costs.groupby('location_type').agg({
                'total_cost': 'sum',
                'capital_cost': 'sum',
                'transport_cost': 'sum',
                'stockout_cost': 'sum'
            }).reset_index()
            
            # Convertir a moneda seleccionada
            if currency == "USD" and exchange_rate:
                for col in ['total_cost', 'capital_cost', 'transport_cost', 'stockout_cost']:
                    location_summary[col] = location_summary[col] / exchange_rate
            
            # Crear gráfico de barras apiladas
            fig = go.Figure()
            
            # Ordenar por costo total
            location_summary = location_summary.sort_values('total_cost', ascending=False)
            
            # Añadir barras para cada componente
            fig.add_trace(go.Bar(
                x=location_summary['location_type'],
                y=location_summary['capital_cost'],
                name='Capital Inmovilizado',
                marker_color='#4e73df'
            ))
            
            fig.add_trace(go.Bar(
                x=location_summary['location_type'],
                y=location_summary['transport_cost'],
                name='Transporte de Valores',
                marker_color='#1cc88a'
            ))
            
            fig.add_trace(go.Bar(
                x=location_summary['location_type'],
                y=location_summary['stockout_cost'],
                name='Agotamiento (Esperado)',
                marker_color='#e74a3b'
            ))
            
            # Configurar diseño
            fig.update_layout(
                title='Costos por Tipo de Ubicación',
                xaxis_title='Tipo de Ubicación',
                yaxis_title=f'Costo Diario (${currency_suffix})',
                barmode='stack',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de costo-beneficio de la frecuencia de reabastecimiento
            st.write("### Análisis de Frecuencia de Reabastecimiento")
            
            # Simulación de diferentes frecuencias
            frequencies = [1, 2, 3, 5, 7, 10, 14]  # en días
            frequency_costs = []
            
            for freq in frequencies:
                # Estimar costos de reabastecimiento más frecuente/menos frecuente
                # Menos frecuente = más capital inmovilizado, menos costos de transporte
                # Más frecuente = menos capital inmovilizado, más costos de transporte
                
                # Factor de ajuste para capital
                capital_factor = freq / 7  # Normalizado a 7 días
                
                # Factor de ajuste para transporte
                transport_factor = 7 / freq  # Inverso
                
                # Factor de ajuste para agotamiento (más frecuente = menos agotamiento)
                stockout_factor = (freq / 7) ** 2  # Exponencial
                
                # Calcular costos ajustados
                adj_capital = total_capital * capital_factor
                adj_transport = total_transport * transport_factor
                adj_stockout = total_stockout * stockout_factor
                
                # Costo total
                total_cost = adj_capital + adj_transport + adj_stockout
                
                frequency_costs.append({
                    'Frecuencia': f"Cada {freq} días",
                    'Capital': adj_capital,
                    'Transporte': adj_transport,
                    'Agotamiento': adj_stockout,
                    'Total': total_cost
                })
            
            # Crear DataFrame
            freq_df = pd.DataFrame(frequency_costs)
            
            # Encontrar frecuencia óptima
            optimal_freq = freq_df.loc[freq_df['Total'].idxmin()]
            
            # Crear gráfico de línea
            fig = go.Figure()
            
            # Añadir líneas para cada componente
            fig.add_trace(go.Scatter(
                x=freq_df['Frecuencia'],
                y=freq_df['Capital'],
                mode='lines+markers',
                name='Capital Inmovilizado',
                line=dict(color='#4e73df', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=freq_df['Frecuencia'],
                y=freq_df['Transporte'],
                mode='lines+markers',
                name='Transporte de Valores',
                line=dict(color='#1cc88a', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=freq_df['Frecuencia'],
                y=freq_df['Agotamiento'],
                mode='lines+markers',
                name='Agotamiento (Esperado)',
                line=dict(color='#e74a3b', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=freq_df['Frecuencia'],
                y=freq_df['Total'],
                mode='lines+markers',
                name='Costo Total',
                line=dict(color='#36b9cc', width=3)
            ))
            
            # Configurar diseño
            fig.update_layout(
                title='Análisis de Costo-Beneficio por Frecuencia de Reabastecimiento',
                xaxis_title='Frecuencia',
                yaxis_title=f'Costo Diario (${currency_suffix})',
                height=500
            )
            
            # Añadir anotación para el óptimo
            fig.add_annotation(
                x=optimal_freq['Frecuencia'],
                y=optimal_freq['Total'],
                text=f"Óptimo: {optimal_freq['Frecuencia']}",
                showarrow=True,
                arrowhead=1,
                arrowcolor="#36b9cc",
                arrowsize=1,
                arrowwidth=2
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar mensaje con frecuencia óptima
            st.success(f"Frecuencia óptima de reabastecimiento: **{optimal_freq['Frecuencia']}**")
            st.caption(f"Esta frecuencia minimiza el costo total diario (${optimal_freq['Total']:,.0f} {currency_suffix})")
        
        with finance_tab3:
            st.write("### Recomendaciones de Optimización Financiera")
            
            # Identificar cajeros con mayor potencial de optimización
            financial_sim['optimization_potential'] = abs(financial_sim['cash_difference_pct'])
            top_optimization = financial_sim.sort_values('optimization_potential', ascending=False).head(5)
            
            st.write("#### Cajeros con Mayor Potencial de Optimización")
            
            # Preparar datos para visualización
            top_opt_display = top_optimization[['name', 'current_cash_display', 'optimal_cash_display', 'cash_difference', 'cash_difference_pct']].copy()
            
            # Formatear columnas
            top_opt_display['current_cash_display'] = top_opt_display['current_cash_display'].apply(lambda x: f"${x:,.0f}")
            top_opt_display['optimal_cash_display'] = top_opt_display['optimal_cash_display'].apply(lambda x: f"${x:,.0f}")
            top_opt_display['cash_difference'] = top_opt_display['cash_difference'].apply(lambda x: f"${x:,.0f}")
            top_opt_display['cash_difference_pct'] = top_opt_display['cash_difference_pct'].apply(lambda x: f"{x:+.1f}%")
            
            # Renombrar columnas
            top_opt_display.columns = ['Cajero', f'Nivel Actual ({currency_suffix})', f'Nivel Óptimo ({currency_suffix})', 
                                       f'Diferencia ({currency_suffix})', 'Diferencia (%)']
            
            # Mostrar tabla
            st.dataframe(top_opt_display, use_container_width=True, hide_index=True)
            
            # Generar recomendaciones específicas
            excess_count = sum(financial_sim['cash_difference'] > 0)
            deficit_count = sum(financial_sim['cash_difference'] < 0)
            
            st.write("#### Recomendaciones Estratégicas")
            
            recommendations = []
            
            # Recomendación basada en exceso/déficit general
            if total_difference > 0:
                recommendations.append(
                    f"Reducir el efectivo en circulación en aproximadamente ${abs(total_difference):,.0f} {currency_suffix} "
                    f"({excess_count} cajeros tienen exceso de efectivo)"
                )
            else:
                recommendations.append(
                    f"Aumentar el efectivo en circulación en aproximadamente ${abs(total_difference):,.0f} {currency_suffix} "
                    f"({deficit_count} cajeros tienen déficit de efectivo)"
                )
            
            # Recomendación basada en frecuencia óptima
            optimal_freq_days = int(optimal_freq['Frecuencia'].split()[1])
            recommendations.append(f"Ajustar la frecuencia de reabastecimiento a {optimal_freq['Frecuencia']}")
            
            # Recomendación basada en tipos de ubicación
            # Encontrar tipos con mayor costo
            top_location_cost = location_summary.iloc[0]
            recommendations.append(
                f"Optimizar prioritariamente los cajeros en ubicaciones tipo '{top_location_cost['location_type']}', "
                f"que representan los mayores costos"
            )
            
            # Recomendaciones basadas en costos
            cost_components = ['capital_cost', 'transport_cost', 'stockout_cost']
            cost_names = ['Capital inmovilizado', 'Transporte de valores', 'Agotamiento']
            max_cost_index = np.argmax([total_capital, total_transport, total_stockout])
            
            recommendations.append(
                f"Enfocar la optimización en reducir costos de {cost_names[max_cost_index].lower()}, "
                f"que representa el mayor componente de costo"
            )
            
            # Mostrar recomendaciones como lista
            for rec in recommendations:
                st.markdown(f"* {rec}")
            
            # Plan de acción
            st.write("#### Plan de Acción Propuesto")
            
            action_plan = [
                "**Corto plazo (1-2 semanas):**",
                f"- Ajustar los niveles de efectivo en los {min(5, len(top_optimization))} cajeros con mayor potencial identificados",
                f"- Implementar la frecuencia de reabastecimiento óptima ({optimal_freq['Frecuencia']})",
                "",
                "**Mediano plazo (1-3 meses):**",
                "- Revisar y ajustar los umbrales de alerta temprana para evitar agotamientos",
                f"- Establecer política de niveles óptimos diferenciada por tipo de ubicación",
                "- Implementar monitoreo continuo de eficiencia de capital",
                "",
                "**Largo plazo (3-6 meses):**",
                "- Desarrollar un modelo dinámico de optimización que se ajuste automáticamente",
                "- Evaluar la posibilidad de redistribución de efectivo entre cajeros cercanos",
                "- Integrar la optimización de efectivo con la optimización de rutas"
            ]
            
            # Mostrar plan de acción
            for item in action_plan:
                st.markdown(item)
            
            # Estimación de ahorro
            total_current_costs = financial_sim['total_cost'].sum()
            
            if currency == "USD" and exchange_rate:
                total_current_costs /= exchange_rate
            
            # Ahorro estimado (aproximado como 15-20% del costo actual)
            estimated_savings_min = total_current_costs * 0.15
            estimated_savings_max = total_current_costs * 0.20
            
            st.write("#### Ahorro Estimado")
            st.success(
                f"Implementando estas recomendaciones, se estima un ahorro de:\n\n"
                f"**${estimated_savings_min:,.0f} - ${estimated_savings_max:,.0f} {currency_suffix} por día**\n\n"
                f"Equivalente a **${estimated_savings_min*365:,.0f} - ${estimated_savings_max*365:,.0f} {currency_suffix} anuales**"
            )
    else:
        st.info("Ejecute una simulación en la pestaña de Configuración para ver resultados de optimización financiera.")


#####################################################
# PESTAÑA 5: PREDICCIÓN DE DISPONIBILIDAD TÉCNICA
#####################################################
with tab5:
    st.header("🔧 Predicción de Disponibilidad Técnica")
    
    # Verificar si hay resultados de simulación
    results = st.session_state.simulation_params.get('simulation_results', {})
    if results and 'technical' in results:
        technical_sim = results['technical']
        
        # Mostrar pestañas para diferentes visualizaciones técnicas
        tech_tab1, tech_tab2, tech_tab3 = st.tabs([
            "Dashboard Técnico", 
            "Análisis por Componente", 
            "Optimización de Mantenimiento"
        ])
        
        with tech_tab1:
            st.write("### Dashboard de Disponibilidad Técnica")
            
            # Calcular métricas clave
            avg_availability = technical_sim['technical_availability'].mean()
            total_expected_failures = technical_sim['expected_failures'].sum()
            avg_mttr = technical_sim['mttr_hours'].mean()
            critical_components = technical_sim[technical_sim['failure_probability'] > 0.3]
            num_critical = len(critical_components['atm_id'].unique())
            
            # Mostrar KPIs técnicos en tarjetas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Disponibilidad Técnica",
                    f"{avg_availability:.2f}%",
                    help="Porcentaje promedio de tiempo en que los cajeros están operativos"
                )
            
            with col2:
                st.metric(
                    "Fallas Proyectadas",
                    f"{total_expected_failures:.1f}",
                    help=f"Número estimado de fallas técnicas en {st.session_state.simulation_params['simulation_days']} días"
                )
            
            with col3:
                st.metric(
                    "MTTR Promedio",
                    f"{avg_mttr:.1f} hrs",
                    help="Tiempo medio de reparación proyectado"
                )
            
            with col4:
                st.metric(
                    "ATMs en Riesgo",
                    f"{num_critical}",
                    help="Cajeros con alta probabilidad de falla (>30%)"
                )
            
            # Mapa de disponibilidad técnica
            st.write("### Mapa de Riesgo Técnico")
            
            # Calcular una métrica de riesgo general por ATM
            atm_risk = technical_sim.groupby('atm_id').agg({
                'failure_probability': 'max',  # Usar el componente con mayor probabilidad
                'name': 'first',
                'latitude': 'first',
                'longitude': 'first',
                'component': lambda x: ', '.join(x[technical_sim.groupby('atm_id')['failure_probability'].transform('max') == technical_sim['failure_probability']]),
                'technical_availability': 'mean'
            }).reset_index()
            
            # Crear mapa de riesgo técnico
            risk_map = folium.Map(
                location=[atm_risk['latitude'].mean(), atm_risk['longitude'].mean()],
                zoom_start=12
            )
            
            # Definir colores según probabilidad de falla
            def get_risk_color(prob):
                if prob < 0.1:
                    return 'green'
                elif prob < 0.25:
                    return 'blue'
                elif prob < 0.5:
                    return 'orange'
                else:
                    return 'red'
            
            # Añadir marcadores para cada ATM
            for _, row in atm_risk.iterrows():
                color = get_risk_color(row['failure_probability'])
                
                # Crear popup con información detallada
                popup_html = f"""
                    <div style="width: 200px">
                        <h4>{row['name']}</h4>
                        <b>Prob. de Falla:</b> {row['failure_probability']:.1%}<br>
                        <b>Componente Crítico:</b> {row['component']}<br>
                        <b>Disponibilidad:</b> {row['technical_availability']:.1f}%<br>
                    </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row['name']} - Riesgo: {row['failure_probability']:.1%}",
                    icon=folium.Icon(color=color, icon='wrench', prefix='fa')
                ).add_to(risk_map)
            
            # Renderizar mapa
            folium_static(risk_map, width=800, height=500)
            
            # Gráfico de disponibilidad por ubicación
            st.write("### Disponibilidad Técnica por Tipo de Ubicación")
            
            # Calcular disponibilidad por tipo de ubicación
            location_avail = technical_sim.groupby('location_type')['technical_availability'].mean().reset_index()
            location_avail = location_avail.sort_values('technical_availability')
            
            # Crear gráfico
            fig = px.bar(
                location_avail,
                x='location_type',
                y='technical_availability',
                color='technical_availability',
                color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
                labels={
                    'location_type': 'Tipo de Ubicación',
                    'technical_availability': 'Disponibilidad Técnica (%)'
                },
                title='Disponibilidad Técnica por Tipo de Ubicación'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tech_tab2:
            st.write("### Análisis por Componente")
            
            # Selector de componente
            components = technical_sim['component'].unique()
            selected_component = st.selectbox(
                "Seleccione un componente para analizar:",
                options=components
            )
            
            # Filtrar datos para el componente seleccionado
            component_data = technical_sim[technical_sim['component'] == selected_component]
            
            # Análisis de estadísticas del componente
            avg_comp_avail = component_data['technical_availability'].mean()
            avg_comp_mtbf = component_data['mtbf_days'].mean()
            avg_comp_mttr = component_data['mttr_hours'].mean()
            total_comp_failures = component_data['expected_failures'].sum()
            
            # Mostrar estadísticas del componente
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Disponibilidad",
                    f"{avg_comp_avail:.2f}%",
                    help=f"Disponibilidad promedio del {selected_component}"
                )
            
            with col2:
                st.metric(
                    "MTBF",
                    f"{avg_comp_mtbf:.1f} días",
                    help=f"Tiempo medio entre fallos del {selected_component}"
                )
            
            with col3:
                st.metric(
                    "MTTR",
                    f"{avg_comp_mttr:.1f} hrs",
                    help=f"Tiempo medio de reparación del {selected_component}"
                )
            
            with col4:
                st.metric(
                    "Fallas Proyectadas",
                    f"{total_comp_failures:.1f}",
                    help=f"Fallas proyectadas para {selected_component} en el período"
                )
            
            # Gráfico de probabilidad de falla por ATM para este componente
            st.write(f"### Probabilidad de Falla del {selected_component} por ATM")
            
            # Ordenar por probabilidad de falla
            sorted_comp_data = component_data.sort_values('failure_probability', ascending=False).head(10)
            
            # Crear gráfico
            fig = px.bar(
                sorted_comp_data,
                x='name',
                y='failure_probability',
                color='failure_probability',
                color_continuous_scale=[(0, 'green'), (0.5, 'yellow'), (1, 'red')],
                labels={
                    'name': 'ATM',
                    'failure_probability': 'Probabilidad de Falla'
                },
                title=f'Top 10 ATMs con Mayor Riesgo de Falla en {selected_component}'
            )
            
            fig.update_layout(
                xaxis_title='ATM',
                yaxis_title='Probabilidad de Falla',
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de factores que afectan a este componente
            st.write(f"### Factores que Afectan al {selected_component}")
            
            # Factores simulados (en una implementación real, estos se calcularían de los datos)
            factors = {
                'Dispensador': {
                    'Uso Intensivo': 0.45,
                    'Calidad de Billetes': 0.25,
                    'Temperatura': 0.15,
                    'Edad del Equipo': 0.10,
                    'Otros': 0.05
                },
                'Lector de Tarjetas': {
                    'Uso Indebido': 0.40,
                    'Polvo/Suciedad': 0.30,
                    'Intentos de Fraude': 0.15,
                    'Temperatura': 0.10,
                    'Otros': 0.05
                },
                'Teclado': {
                    'Uso Intensivo': 0.35,
                    'Líquidos Derramados': 0.30,
                    'Vandalismo': 0.20,
                    'Edad del Equipo': 0.10,
                    'Otros': 0.05
                },
                'Monitor': {
                    'Horas de Operación': 0.30,
                    'Luz Solar Directa': 0.25,
                    'Temperatura': 0.20,
                    'Edad del Equipo': 0.15,
                    'Otros': 0.10
                },
                'Lector Biométrico': {
                    'Limpieza Inadecuada': 0.35,
                    'Uso Indebido': 0.25,
                    'Temperatura': 0.20,
                    'Humedad': 0.15,
                    'Otros': 0.05
                }
            }
            
            # Obtener factores para el componente seleccionado
            component_factors = factors.get(selected_component, {'Datos no disponibles': 1.0})
            
            # Crear gráfico de torta
            fig = px.pie(
                names=list(component_factors.keys()),
                values=list(component_factors.values()),
                title=f'Factores que Contribuyen a Fallas del {selected_component}'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Códigos de falla comunes
            st.write(f"### Códigos de Falla Comunes para {selected_component}")
            
            # En una implementación real, estos se calcularían de datos históricos
            # Aquí simulamos algunos códigos comunes
            codes = {
                'Dispensador': {
                    'DSP:01:00:00': 'Error de dispensación',
                    'DSP:00:02:00': 'Atasco de billetes',
                    'DSP:01:01:00': 'Error de conteo',
                    'DSP:00:00:03': 'Sensores bloqueados',
                    'DSP:02:00:00': 'Cassette no reconocido'
                },
                'Lector de Tarjetas': {
                    'CRD:00:01:00': 'Error de lectura de chip',
                    'CRD:01:00:00': 'Error de lectura de banda',
                    'CRD:00:00:02': 'Tarjeta retenida',
                    'CRD:01:01:00': 'Error de comunicación',
                    'CRD:00:02:00': 'Lector bloqueado'
                },
                'Teclado': {
                    'EPP:01:01:00': 'Tecla bloqueada',
                    'EPP:00:01:00': 'Error de encriptación',
                    'EPP:01:00:00': 'Error de comunicación',
                    'EPP:00:00:03': 'Daño físico',
                    'EPP:02:00:00': 'Error de inicialización'
                },
                'Monitor': {
                    'MT:00:03:00': 'Error de pantalla',
                    'MT:01:00:00': 'Error de calibración',
                    'MT:00:01:00': 'Sin respuesta táctil',
                    'MT:01:01:00': 'Error de imagen',
                    'MT:00:00:02': 'Error de luz de fondo'
                },
                'Lector Biométrico': {
                    'BRD:02:00:00': 'Error de lectura de huella',
                    'BRD:00:01:00': 'Error de calibración',
                    'BRD:01:00:00': 'Superficie sucia',
                    'BRD:00:00:03': 'Error de comunicación',
                    'BRD:01:01:00': 'Sensor defectuoso'
                }
            }
            
            # Obtener códigos para el componente seleccionado
            component_codes = codes.get(selected_component, {'000': 'Datos no disponibles'})
            
            # Crear tabla de códigos
            codes_df = pd.DataFrame({
                'Código': list(component_codes.keys()),
                'Descripción': list(component_codes.values())
            })
            
            st.dataframe(codes_df, use_container_width=True, hide_index=True)
        
        with tech_tab3:
            st.write("### Optimización de Mantenimiento")
            
            # Estrategias de mantenimiento
            st.write("#### Estrategias de Mantenimiento")
            
            # Definir estrategias
            strategies = {
                'Reactivo': 'Reparar solo cuando ocurre una falla',
                'Preventivo Básico': 'Mantenimiento periódico según calendario',
                'Preventivo Optimizado': 'Mantenimiento basado en predicciones de IA',
                'Preventivo + Rutas Optimizadas': 'Combinar mantenimiento con optimización de rutas'
            }
            
            # Simular efectos de diferentes estrategias
            strategy_effects = {
                'Reactivo': {
                    'availability': avg_availability,
                    'mttr': avg_mttr,
                    'cost': 100  # Costo base
                },
                'Preventivo Básico': {
                    'availability': min(avg_availability + 5, 100),
                    'mttr': avg_mttr * 0.9,
                    'cost': 130  # 30% más caro
                },
                'Preventivo Optimizado': {
                    'availability': min(avg_availability + 8, 100),
                    'mttr': avg_mttr * 0.8,
                    'cost': 120  # 20% más caro
                },
                'Preventivo + Rutas Optimizadas': {
                    'availability': min(avg_availability + 9, 100),
                    'mttr': avg_mttr * 0.7,
                    'cost': 110  # 10% más caro (ahorros en logística)
                }
            }
            
            # Crear tabla comparativa
            strategy_df = pd.DataFrame({
                'Estrategia': list(strategies.keys()),
                'Descripción': list(strategies.values()),
                'Disponibilidad (%)': [effects['availability'] for effects in strategy_effects.values()],
                'MTTR (horas)': [effects['mttr'] for effects in strategy_effects.values()],
                'Costo Relativo (%)': [effects['cost'] for effects in strategy_effects.values()]
            })
            
            # Mostrar tabla
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)
            
            # Gráfico comparativo
            st.write("#### Comparación de Estrategias")
            
            # Preparar datos para gráfico
            strategy_names = list(strategies.keys())
            availabilities = [effects['availability'] for effects in strategy_effects.values()]
            costs = [effects['cost'] for effects in strategy_effects.values()]
            
            # Crear figura con dos ejes Y
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Añadir barras para disponibilidad
            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=availabilities,
                    name='Disponibilidad Técnica (%)',
                    marker_color='#4e73df'
                ),
                secondary_y=False
            )
            
            # Añadir línea para costo
            fig.add_trace(
                go.Scatter(
                    x=strategy_names,
                    y=costs,
                    name='Costo Relativo (%)',
                    marker_color='#e74a3b',
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Configurar ejes
            fig.update_layout(
                title='Comparación de Estrategias de Mantenimiento',
                xaxis_title='Estrategia',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_yaxes(title_text="Disponibilidad Técnica (%)", secondary_y=False)
            fig.update_yaxes(title_text="Costo Relativo (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Optimización de rutas de mantenimiento
            st.write("### Rutas Optimizadas de Mantenimiento")
            
            # Filtrar ATMs con mayor riesgo
            high_risk_atms = atm_risk[atm_risk['failure_probability'] > 0.2].sort_values('failure_probability', ascending=False)
            
            if len(high_risk_atms) > 0:
                st.write(f"Se identificaron {len(high_risk_atms)} ATMs con alta probabilidad de falla (>20%)")
                
                # Opciones de optimización
                col1, col2 = st.columns(2)
                
                with col1:
                    # Seleccionar número de vehículos
                    num_vehicles = st.slider(
                        "Número de vehículos técnicos:",
                        min_value=1,
                        max_value=5,
                        value=2,
                        help="Número de vehículos disponibles para mantenimiento"
                    )
                
                with col2:
                    # Seleccionar tipo de optimización
                    route_type = st.radio(
                        "Tipo de optimización:",
                        options=["Solo mantenimiento", "Integrado con reabastecimiento"],
                        index=0,
                        help="Seleccionar si las rutas son exclusivas para mantenimiento o combinadas"
                    )
                
                # Botón para generar rutas
                generate_routes = st.button("Generar Rutas de Mantenimiento", type="primary")
                
                if generate_routes:
                    with st.spinner("Optimizando rutas de mantenimiento..."):
                        # En una implementación real, aquí llamaríamos a una función
                        # que realmente optimice las rutas. Aquí simulamos los resultados.
                        
                        # Función simulada para generar rutas (muy simplificada)
                        def simulate_maintenance_routes(atms, num_vehicles, route_type):
                            """Simula rutas de mantenimiento optimizadas"""
                            # En una implementación real, usaríamos un algoritmo de optimización
                            # similar al de optimize_routes_for_date en api/models/optimization.py
                            
                            # Por ahora, solo dividimos los ATMs entre los vehículos
                            routes = []
                            
                            # Ubicación de base (usamos la primera transportadora)
                            base_lat = 4.6481  # Coordenadas de ejemplo para Bogotá
                            base_lon = -74.1070
                            
                            # Distribuir ATMs entre vehículos
                            atms_per_vehicle = len(atms) // num_vehicles
                            if atms_per_vehicle == 0:
                                atms_per_vehicle = 1
                            
                            for v in range(num_vehicles):
                                start_idx = v * atms_per_vehicle
                                end_idx = start_idx + atms_per_vehicle
                                
                                if v == num_vehicles - 1:
                                    # El último vehículo toma todos los ATMs restantes
                                    end_idx = len(atms)
                                
                                if start_idx >= len(atms):
                                    break
                                
                                vehicle_atms = atms.iloc[start_idx:end_idx]
                                
                                if len(vehicle_atms) > 0:
                                    # Creamos una ruta simple: base -> ATMs (ordenados por riesgo) -> base
                                    route_points = []
                                    
                                    # Punto inicial (base)
                                    route_points.append({
                                        'lat': base_lat,
                                        'lon': base_lon,
                                        'name': 'Base de Operaciones',
                                        'type': 'base'
                                    })
                                    
                                    # Puntos intermedios (ATMs)
                                    for _, atm in vehicle_atms.iterrows():
                                        route_points.append({
                                            'lat': atm['latitude'],
                                            'lon': atm['longitude'],
                                            'name': atm['name'],
                                            'type': 'atm',
                                            'risk': atm['failure_probability'],
                                            'component': atm['component']
                                        })
                                    
                                    # Punto final (volver a base)
                                    route_points.append({
                                        'lat': base_lat,
                                        'lon': base_lon,
                                        'name': 'Base de Operaciones',
                                        'type': 'base'
                                    })
                                    
                                    # Calcular distancia total (muy simplificado)
                                    total_distance = 0
                                    for i in range(len(route_points) - 1):
                                        # Distancia euclidiana (muy simplificada)
                                        p1 = route_points[i]
                                        p2 = route_points[i + 1]
                                        dist = ((p1['lat'] - p2['lat'])**2 + (p1['lon'] - p2['lon'])**2)**0.5
                                        # Convertir a km (aproximado)
                                        dist_km = dist * 111  # 1 grado ≈ 111 km
                                        total_distance += dist_km
                                    
                                    # Crear ruta
                                    route = {
                                        'vehicle_id': v,
                                        'route': list(range(len(route_points))),
                                        'distance': total_distance,
                                        'atms': len(vehicle_atms),
                                        'type': route_type
                                    }
                                    
                                    routes.append(route)
                            
                            return routes, route_points
                        
                        # Generar rutas simuladas
                        maintenance_routes, maintenance_locations = simulate_maintenance_routes(
                            high_risk_atms,
                            num_vehicles,
                            route_type
                        )
                        
                        # Mostrar resultados
                        if maintenance_routes and len(maintenance_routes) > 0:
                            st.success(f"Se generaron {len(maintenance_routes)} rutas de mantenimiento")
                            
                            # Mostrar mapa con rutas
                            from frontend.components.maps import route_visualization
                            
                            # Crear datos de transportadora simulada
                            dummy_carrier = {
                                'id': 'MAINT',
                                'name': 'Servicio Técnico',
                                'vehicles': num_vehicles,
                                'base_latitude': maintenance_locations[0]['lat'],
                                'base_longitude': maintenance_locations[0]['lon'],
                                'service_area': 'Bogotá'
                            }
                            
                            # Visualizar rutas
                            route_visualization(maintenance_locations, maintenance_routes, dummy_carrier)
                            
                            # Mostrar tabla de resumen de rutas
                            st.write("#### Detalle de Rutas de Mantenimiento")
                            
                            route_details = []
                            for i, route in enumerate(maintenance_routes):
                                route_details.append({
                                    'Ruta': i+1,
                                    'Vehículo': f"Técnico {route['vehicle_id']+1}",
                                    'ATMs a Visitar': route['atms'],
                                    'Distancia (km)': f"{route['distance']:.2f}",
                                    'Tipo': route['type']
                                })
                            
                            route_df = pd.DataFrame(route_details)
                            st.dataframe(route_df, use_container_width=True, hide_index=True)
                            
                            # Calcular ahorros
                            maintenance_time = sum(route['atms'] * 1.5 for route in maintenance_routes)  # 1.5 horas por ATM
                            total_distance = sum(route['distance'] for route in maintenance_routes)
                            
                            # Mostrar métricas de optimización
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Tiempo Total Estimado",
                                    f"{maintenance_time:.1f} horas",
                                    help="Tiempo total estimado para completar el mantenimiento"
                                )
                            
                            with col2:
                                st.metric(
                                    "Distancia Total",
                                    f"{total_distance:.2f} km",
                                    help="Distancia total recorrida por todos los vehículos"
                                )
                            
                            with col3:
                                # Mejora estimada en disponibilidad
                                availability_improvement = 2.5  # Ejemplo: 2.5 puntos porcentuales
                                st.metric(
                                    "Mejora en Disponibilidad",
                                    f"+{availability_improvement:.1f}%",
                                    help="Mejora estimada en disponibilidad después del mantenimiento"
                                )
                            
                            # Añadir notas sobre la optimización
                            st.info("""
                            **Notas sobre la optimización:**
                            - Las rutas están priorizadas para atender primero los ATMs con mayor riesgo de falla.
                            - El mantenimiento preventivo reducirá significativamente la probabilidad de fallas.
                            - El tiempo estimado incluye diagnóstico y reparación preventiva de los componentes en riesgo.
                            """)
                        
                        else:
                            st.warning("No se pudieron generar rutas de mantenimiento. Intente con diferentes parámetros.")
            else:
                st.info("No hay ATMs con riesgo técnico significativo que requieran mantenimiento preventivo inmediato.")
            
            # Análisis de costo-beneficio
            st.write("### Análisis de Costo-Beneficio")
            
            # Crear tabla de análisis financiero
            st.write("#### Impacto Financiero del Mantenimiento Preventivo")
            
            # Datos simulados para análisis financiero
            financial_analysis = pd.DataFrame({
                'Métrica': [
                    'Costo de Mantenimiento Preventivo',
                    'Costo Evitado por Fallas',
                    'Ahorro Neto',
                    'ROI del Mantenimiento',
                    'Mejora en Disponibilidad',
                    'Reducción en MTTR',
                    'Extensión de Vida Útil'
                ],
                'Valor': [
                    '$5,000,000 COP',
                    '$8,750,000 COP',
                    '$3,750,000 COP',
                    '75%',
                    '+2.5%',
                    '-30%',
                    '+10%'
                ],
                'Descripción': [
                    'Costo total del programa de mantenimiento preventivo',
                    'Costos evitados por prevención de fallas (tiempo de inactividad, reparaciones, reputación)',
                    'Beneficio neto del programa preventivo',
                    'Retorno sobre la inversión del programa',
                    'Incremento en disponibilidad técnica',
                    'Reducción en tiempo medio de reparación',
                    'Incremento estimado en vida útil de los equipos'
                ]
            })
            
            st.dataframe(financial_analysis, use_container_width=True, hide_index=True)
            
            # Recomendaciones finales
            st.write("### Recomendaciones Técnicas")
            
            # Lista de recomendaciones basadas en el análisis
            recommendations = [
                "**Implementar mantenimiento preventivo optimizado** para los ATMs de mayor riesgo identificados.",
                "**Incrementar la frecuencia de mantenimiento** en ubicaciones con condiciones ambientales adversas (calle, estación).",
                "**Integrar rutas de mantenimiento con reabastecimiento** para maximizar la eficiencia logística.",
                "**Monitorear continuamente** los componentes críticos, especialmente Dispensadores y Lectores de Tarjetas.",
                "**Establecer un inventario estratégico** de repuestos basado en las predicciones de falla."
            ]
            
            # Mostrar recomendaciones
            for rec in recommendations:
                st.markdown(f"* {rec}")
            
            # Añadir explicación sobre el valor de la predicción
            st.info("""
            El enfoque predictivo permite identificar problemas potenciales antes de que ocurran, reduciendo significativamente el tiempo de inactividad y mejorando la experiencia del cliente. La integración con el sistema de optimización de rutas garantiza que los recursos técnicos se utilicen de la manera más eficiente posible.
            """)
    else:
        st.info("Ejecute una simulación en la pestaña de Configuración para ver resultados de disponibilidad técnica.")

            