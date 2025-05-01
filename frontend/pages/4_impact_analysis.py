"""
Página de análisis de impacto.

Este módulo define la página que muestra el análisis de impacto en
KPIs de negocio y cálculos de ahorro estimado.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Librerias para generar el reporte
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

# Agregar ruta para importación de módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar módulos necesarios
from api.data.db_connector import create_db_connection, load_atm_data
from api.data.simulation import simulate_current_atm_status, generate_sample_carriers
from api.models.prediction import predict_cash_demand, get_priority_atms
from api.models.optimization import optimize_routes_for_date, simulate_scenario
from api.utils.helpers import format_currency, get_exchange_rate
from api.utils.metrics import calculate_current_kpis, calculate_improved_kpis, calculate_savings

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Impacto - ATM Optimizer",
    page_icon="💰",
    layout="wide"
)

# Título
st.title("Análisis de Impacto")
st.markdown("Evaluación del impacto económico y operativo de la optimización de rutas")

# Inicialización de datos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_impact_data():
    """Carga o simula datos necesarios para el análisis de impacto"""
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
        atms_df = pd.DataFrame()
        current_status = simulate_current_atm_status(atms_df)
        predictions = predict_cash_demand(current_status, days_ahead=7)
        carriers_df = generate_sample_carriers(3)
        
        return current_status, predictions, carriers_df, current_status

# Cargar datos
current_status, predictions, carriers_df, atms_df = load_impact_data()

# Configuración en sidebar
st.sidebar.title("Parámetros de Análisis")

# Selector de fecha
available_dates = sorted(predictions['date'].unique())
selected_date = st.sidebar.selectbox(
    "Fecha para análisis",
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

# Control para número máximo de cajeros
max_cajeros = st.sidebar.slider(
    "Máximo de cajeros a visitar:",
    min_value=5,
    max_value=50,
    value=15
)

# Botón para calcular impacto
calculate_impact = st.sidebar.button("Calcular Impacto", type="primary")

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

# Sección principal para mostrar el análisis de impacto
if calculate_impact or 'impact_calculated' in st.session_state:
    
    # Si es la primera vez o se solicita recalcular
    if calculate_impact or 'optimization_results' not in st.session_state:
        with st.spinner("Calculando impacto de optimización..."):
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
            
            # Calcular KPIs actuales
            current_kpis = calculate_current_kpis(current_status)
            
            # Calcular KPIs mejorados después de optimización
            improved_kpis = calculate_improved_kpis(current_status, atms_to_restock)
            
            # Calcular ahorros económicos
            savings = calculate_savings(routes, atms_to_restock, selected_carrier)
            
            # Guardar resultados en estado de sesión
            st.session_state.optimization_results = {
                'routes': routes,
                'atms_to_restock': atms_to_restock,
                'selected_carrier': selected_carrier,
                'locations': locations,
                'current_kpis': current_kpis,
                'improved_kpis': improved_kpis,
                'savings': savings
            }
            
            st.session_state.impact_calculated = True
    
    # Obtener resultados de la sesión
    results = st.session_state.optimization_results
    routes = results['routes']
    atms_to_restock = results['atms_to_restock']
    current_kpis = results['current_kpis']
    improved_kpis = results['improved_kpis']
    savings = results['savings']
    
    # Mostrar resultados solo si hay rutas
    if routes and len(routes) > 0:
        st.success("Análisis de impacto completado.")
        
        # Sección 1: Impacto en KPIs de Negocio
        st.header("💼 Impacto en KPIs de Negocio")
        
        # Crear tabla comparativa de KPIs
        kpi_comparison = []
        
        # Definir los KPIs y sus descripciones
        kpi_info = {
            'disponibilidad': {
                'name': 'Disponibilidad de Efectivo',
                'unit': '%',
                'description': 'Porcentaje de cajeros con efectivo por encima del umbral mínimo',
                'target': '≥ 85%',
                'is_good': lambda v: v >= 85
            },
            'downtime': {
                'name': 'Downtime por Agotamiento',
                'unit': '%',
                'description': 'Porcentaje de cajeros en estado crítico (sin efectivo disponible)',
                'target': '≤ 15%',
                'is_good': lambda v: v <= 15
            },
            'eficiencia_capital': {
                'name': 'Eficiencia de Capital',
                'unit': '%',
                'description': 'Utilización promedio de la capacidad de los cajeros',
                'target': '40-70%',
                'is_good': lambda v: 40 <= v <= 70
            },
            'dias_hasta_agotamiento': {
                'name': 'Días hasta Agotamiento',
                'unit': ' días',
                'description': 'Promedio de días hasta que los cajeros lleguen al umbral mínimo',
                'target': '≥ 5 días',
                'is_good': lambda v: v >= 5
            },
            'requieren_atencion_pronto': {
                'name': 'Requieren Atención Pronto',
                'unit': ' cajeros',
                'description': 'Número de cajeros que necesitan reabastecimiento en los próximos 3 días',
                'target': 'Mínimo posible',
                'is_good': lambda v: v <= len(current_status) * 0.1  # menos del 10% del total
            }
        }
        
        # Crear datos para tabla comparativa
        for key, info in kpi_info.items():
            current = current_kpis.get(key, 0)
            improved = improved_kpis.get(key, 0)
            change = improved - current
            pct_change = (change / current * 100) if current != 0 else 0
            
            # Determinar si el cambio es positivo o negativo
            # Para downtime y cajeros que requieren atención, menor es mejor
            if key in ['downtime', 'requieren_atencion_pronto']:
                is_improvement = change < 0
                arrow = "↓" if change < 0 else "↑"
            else:
                is_improvement = change > 0
                arrow = "↑" if change > 0 else "↓"
            
            # Determinar si cumple objetivo después de mejora
            meets_target = info['is_good'](improved)
            
            kpi_comparison.append({
                'KPI': info['name'],
                'Descripción': info['description'],
                'Valor Actual': f"{current}{info['unit']}",
                'Valor Optimizado': f"{improved}{info['unit']}",
                'Cambio': f"{arrow} {abs(change):.2f}{info['unit']} ({abs(pct_change):.1f}%)",
                'Objetivo': info['target'],
                'Cumple': "✅" if meets_target else "❌",
                'is_improvement': is_improvement
            })
        
        # Crear DataFrame para mostrar
        kpi_df = pd.DataFrame(kpi_comparison)
        
        # Visualizar tabla con colores según mejora o empeoramiento
        st.markdown("### Comparación de Métricas Clave")
        st.markdown("Análisis del impacto de la optimización en los indicadores clave de desempeño:")
        
        # Mostrar la tabla
        st.dataframe(
            kpi_df[['KPI', 'Valor Actual', 'Valor Optimizado', 'Cambio', 'Objetivo', 'Cumple']],
            use_container_width=True,
            column_config={
                "Cambio": st.column_config.TextColumn(
                    "Cambio",
                    help="Variación entre el valor actual y optimizado"
                ),
                "Cumple": st.column_config.TextColumn(
                    "Cumple Objetivo",
                    help="Indica si el valor optimizado cumple con el objetivo establecido"
                )
            }
        )
        
        # Sección 2: Impacto Económico
        st.header("💰 Impacto Económico")
        
        # Calcular métricas adicionales de ahorro
        monthly_savings = savings['ahorro_mensual']
        annual_savings = monthly_savings * 12
        
        # Variables para análisis económico
        costo_operativo_anual = 1200000000  # Ejemplo: 1.2 mil millones COP
        if currency == "USD" and exchange_rate:
            costo_operativo_anual /= exchange_rate
        
        # Calcular ROI y otros indicadores
        reduccion_costos_pct = (annual_savings / costo_operativo_anual * 100) if costo_operativo_anual > 0 else 0
        
        # Mostrar indicadores económicos clave
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Ahorro Mensual Proyectado", 
                format_currency(monthly_savings, currency, exchange_rate)
            )
        
        with col2:
            st.metric(
                "Ahorro Anual Proyectado", 
                format_currency(annual_savings, currency, exchange_rate)
            )
        
        with col3:
            st.metric(
                "Reducción de Costos Operativos", 
                f"{reduccion_costos_pct:.2f}%",
                help="Porcentaje de reducción en los costos operativos anuales"
            )
        
        # Gráfico de ahorro proyectado a 12 meses
        st.markdown("### Proyección de Ahorro a 12 Meses")
        
        # Generar datos para proyección
        months = list(range(1, 13))
        cumulative_savings = [monthly_savings * month for month in months]
        
        # Crear gráfico de área
        fig = go.Figure()
        
        # Añadir área para ahorro acumulado
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_savings,
            fill='tozeroy',
            name='Ahorro Acumulado',
            line=dict(color='#1cc88a', width=2),
            fillcolor='rgba(28, 200, 138, 0.2)'
        ))
        
        # Añadir línea para ahorro mensual
        fig.add_trace(go.Scatter(
            x=months,
            y=[monthly_savings] * 12,
            mode='lines',
            name='Ahorro Mensual',
            line=dict(color='#4e73df', width=2, dash='dash')
        ))
        
        # Configurar diseño
        fig.update_layout(
            title='Proyección de Ahorro Acumulado a 12 Meses',
            xaxis_title='Mes',
            yaxis_title=f'Ahorro ({currency})',
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
        if currency == "USD":
            fig.update_yaxes(tickprefix='$', ticksuffix=' USD')
        else:
            fig.update_yaxes(tickprefix='$', ticksuffix=' COP')
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Sección 3: Análisis Detallado del Ahorro
        st.header("🔍 Análisis Detallado del Ahorro")
        
        # Crear pestañas para diferentes análisis
        tab1, tab2 = st.tabs(["Composición del Ahorro", "Análisis por Componente"])
        
        with tab1:
            # Crear gráfico de composición del ahorro
            st.markdown("### Composición del Ahorro")
            
            # Datos para el gráfico
            ahorro_combustible = savings['ahorro_distancia'] * 2000  # 2000 COP por km
            ahorro_tiempo = savings['ahorro_distancia'] * 100000 / 60  # 100K COP por hora, a 60km/h en promedio
            ahorro_mantenimiento = savings['ahorro_distancia'] * 1500  # 1500 COP por km
            
            # Convertir a la moneda seleccionada
            if currency == "USD" and exchange_rate:
                ahorro_combustible /= exchange_rate
                ahorro_tiempo /= exchange_rate
                ahorro_mantenimiento /= exchange_rate
            
            # Datos para gráfico de torta
            labels = ['Combustible', 'Tiempo', 'Mantenimiento']
            values = [ahorro_combustible, ahorro_tiempo, ahorro_mantenimiento]
            color_palette = ['#4e73df', '#1cc88a', '#36b9cc']
            
            # Crear gráfico
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker_colors=color_palette
            )])
            
            fig.update_layout(
                title="Composición del Ahorro Diario",
                height=400
            )
            
            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla detallada
            saving_details = pd.DataFrame({
                'Componente': labels,
                'Ahorro Diario': [
                    format_currency(ahorro_combustible, currency, exchange_rate),
                    format_currency(ahorro_tiempo, currency, exchange_rate),
                    format_currency(ahorro_mantenimiento, currency, exchange_rate)
                ],
                'Ahorro Mensual': [
                    format_currency(ahorro_combustible * 20, currency, exchange_rate),
                    format_currency(ahorro_tiempo * 20, currency, exchange_rate),
                    format_currency(ahorro_mantenimiento * 20, currency, exchange_rate)
                ],
                'Porcentaje': [
                    f"{ahorro_combustible/sum(values)*100:.1f}%",
                    f"{ahorro_tiempo/sum(values)*100:.1f}%",
                    f"{ahorro_mantenimiento/sum(values)*100:.1f}%"
                ]
            })
            
            st.dataframe(saving_details, use_container_width=True)
        
        with tab2:
            # Análisis por componente
            st.markdown("### Análisis por Componente")
            
            # Crear gráfico de barras comparativo por componente
            components = ['Distancia (km)', 'Tiempo (horas)', 'Combustible', 'Mantenimiento']
            
            # Valores sin optimización
            no_opt = [
                savings['distancia_no_optimizada'],
                savings['distancia_no_optimizada'] / 60,  # horas a 60km/h
                savings['distancia_no_optimizada'] * 2000,  # costo combustible
                savings['distancia_no_optimizada'] * 1500   # costo mantenimiento
            ]
            
            # Valores con optimización
            opt = [
                savings['distancia_no_optimizada'] - savings['ahorro_distancia'],
                (savings['distancia_no_optimizada'] - savings['ahorro_distancia']) / 60,
                (savings['distancia_no_optimizada'] - savings['ahorro_distancia']) * 2000,
                (savings['distancia_no_optimizada'] - savings['ahorro_distancia']) * 1500
            ]
            
            # Convertir costos a la moneda seleccionada
            if currency == "USD" and exchange_rate:
                for i in [2, 3]:  # índices de los componentes monetarios
                    no_opt[i] /= exchange_rate
                    opt[i] /= exchange_rate
            
            # Crear figura con subplots
            fig = make_subplots(rows=2, cols=2, subplot_titles=components)
            
            # Añadir barras para cada componente
            for i, component in enumerate(components):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Bar(
                        x=['Sin Optimización', 'Con Optimización'],
                        y=[no_opt[i], opt[i]],
                        marker_color=['#e74a3b', '#1cc88a']
                    ),
                    row=row, col=col
                )
                
                # Añadir etiqueta de ahorro
                ahorro = no_opt[i] - opt[i]
                ahorro_pct = (ahorro / no_opt[i] * 100) if no_opt[i] > 0 else 0
                
                fig.add_annotation(
                    x=0.5,
                    y=max(no_opt[i], opt[i]) * 1.1,
                    text=f"Ahorro: {ahorro:.2f} ({ahorro_pct:.1f}%)",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    row=row, col=col
                )
            
            # Actualizar diseño
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Comparación de Componentes Antes y Después de Optimización"
            )
            
            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)
        
        # Sección 4: Recomendaciones
        st.header("🔄 Recomendaciones y Próximos Pasos")
        
        # Crear columnas para diferentes tipos de recomendaciones
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Recomendaciones Operativas")
            
            # Determinar recomendaciones basadas en los resultados
            recommendations = [
                "Implementar las rutas optimizadas para reducir costos operativos.",
                f"Priorizar reabastecimiento de {len(atms_to_restock)} cajeros identificados.",
                f"Utilizar {num_vehicles} vehículos para maximizar eficiencia."
            ]
            
            # Añadir recomendaciones específicas según KPIs
            if improved_kpis['eficiencia_capital'] < 40:
                recommendations.append("Reducir el efectivo en circulación para mejorar la eficiencia de capital.")
            elif improved_kpis['eficiencia_capital'] > 70:
                recommendations.append("Aumentar el efectivo en circulación para reducir frecuencia de reabastecimiento.")
            
            if improved_kpis['downtime'] > 15:
                recommendations.append("Incrementar frecuencia de reabastecimiento para reducir downtime.")
            
            # Mostrar recomendaciones como lista
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        with col2:
            st.markdown("### Próximos Pasos")
            
            next_steps = [
                "Implementar un sistema de monitoreo continuo de KPIs.",
                "Evaluar impacto en satisfacción de clientes y tiempo de servicio.",
                "Analizar patrones estacionales para ajustar optimización.",
                "Integrar datos de tráfico en tiempo real para mejorar precisión."
            ]
            
            # Mostrar próximos pasos como lista
            for step in next_steps:
                st.markdown(f"• {step}")
            
        # Botón para descargar informe detallado (simulado)
        # Función para generar el reporte PDF
        def generate_report_pdf(current_kpis, improved_kpis, savings, routes, atms_to_restock, currency, exchange_rate):
            """Genera un informe PDF detallado con los resultados del análisis de impacto"""
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Estilos personalizados
            title_style = styles["Heading1"]
            subtitle_style = styles["Heading2"]
            normal_style = styles["Normal"]
            
            # Estilos adicionales
            styles.add(ParagraphStyle(
                name='Centered',
                parent=styles['Normal'],
                alignment=1,  
            ))

            # Definir colores 
            grey_color = colors.HexColor('#BFBFBF')
            whitesmoke_color = colors.HexColor('#F5F5F5')
            beige_color = colors.HexColor('#F5F5DC') 
            
            # Título del informe
            elements.append(Paragraph("Informe de Análisis de Impacto", title_style))
            elements.append(Paragraph(f"Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Centered"]))
            elements.append(Spacer(1, 20))
            
            # Resumen ejecutivo
            elements.append(Paragraph("Resumen Ejecutivo", subtitle_style))
            
            # Calcular algunos valores clave para el resumen
            total_distance = sum(route['distance'] for route in routes)
            total_atms = sum(len(route['route']) - 2 for route in routes)
            monthly_savings = savings['ahorro_mensual']
            annual_savings = monthly_savings * 12
            
            formatted_monthly = format_currency(monthly_savings, currency, exchange_rate)
            formatted_annual = format_currency(annual_savings, currency, exchange_rate)
            
            # Crear tabla de resumen
            summary_data = [
                ["Métrica", "Valor"],
                ["Cajeros a visitar", str(total_atms)],
                ["Distancia total", f"{total_distance:.2f} km"],
                ["Vehículos utilizados", str(len(routes))],
                ["Ahorro mensual proyectado", formatted_monthly],
                ["Ahorro anual proyectado", formatted_annual],
            ]
            
            summary_table = Table(summary_data, colWidths=[250, 200])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), grey_color),
                ('TEXTCOLOR', (0, 0), (1, 0), whitesmoke_color),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), beige_color),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Comparación de KPIs
            elements.append(Paragraph("Impacto en KPIs de Negocio", subtitle_style))
            elements.append(Spacer(1, 10))
            
            # Crear tabla de KPIs
            kpi_data = [
                ["KPI", "Valor Actual", "Valor Optimizado", "Cambio", "Cumple"],
            ]
            
            # Definir los KPIs y sus descripciones
            kpi_info = {
                'disponibilidad': {
                    'name': 'Disponibilidad de Efectivo',
                    'unit': '%',
                    'is_good': lambda v: v >= 85
                },
                'downtime': {
                    'name': 'Downtime por Agotamiento',
                    'unit': '%',
                    'is_good': lambda v: v <= 15
                },
                'eficiencia_capital': {
                    'name': 'Eficiencia de Capital',
                    'unit': '%',
                    'is_good': lambda v: 40 <= v <= 70
                },
                'dias_hasta_agotamiento': {
                    'name': 'Días hasta Agotamiento',
                    'unit': ' días',
                    'is_good': lambda v: v >= 5
                },
                'requieren_atencion_pronto': {
                    'name': 'Requieren Atención Pronto',
                    'unit': ' cajeros',
                    'is_good': lambda v: True  
                }
            }
            
            # Llenar la tabla con datos
            for key, info in kpi_info.items():
                current = current_kpis.get(key, 0)
                improved = improved_kpis.get(key, 0)
                change = improved - current
                pct_change = (change / current * 100) if current != 0 else 0
                
                # Determinar si el cambio es positivo o negativo
                if key in ['downtime', 'requieren_atencion_pronto']:
                    arrow = "↓" if change < 0 else "↑"
                else:
                    arrow = "↑" if change > 0 else "↓"
                
                # Determinar si cumple objetivo
                meets_target = info['is_good'](improved)
                cumple = "✓" if meets_target else "✗"
                
                kpi_data.append([
                    info['name'],
                    f"{current}{info['unit']}",
                    f"{improved}{info['unit']}",
                    f"{arrow} {abs(change):.2f}{info['unit']} ({abs(pct_change):.1f}%)",
                    cumple
                ])
            
            kpi_table = Table(kpi_data, colWidths=[120, 90, 90, 130, 50])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), grey_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke_color),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), beige_color),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(kpi_table)
            elements.append(Spacer(1, 20))
            
            # Análisis de Ahorro
            elements.append(Paragraph("Análisis Detallado del Ahorro", subtitle_style))
            elements.append(Spacer(1, 10))
            
            # Crear tabla de componentes de ahorro
            ahorro_combustible = savings['ahorro_distancia'] * 2000  # 2000 COP por km
            ahorro_tiempo = savings['ahorro_distancia'] * 100000 / 60  # 100K COP por hora, a 60km/h
            ahorro_mantenimiento = savings['ahorro_distancia'] * 1500  # 1500 COP por km
            
            # Convertir a la moneda seleccionada
            if currency == "USD" and exchange_rate:
                ahorro_combustible /= exchange_rate
                ahorro_tiempo /= exchange_rate
                ahorro_mantenimiento /= exchange_rate
            
            savings_data = [
                ["Componente", "Ahorro Diario", "Ahorro Mensual", "Porcentaje"],
                ["Combustible", 
                format_currency(ahorro_combustible, currency, exchange_rate),
                format_currency(ahorro_combustible * 20, currency, exchange_rate),
                f"{ahorro_combustible/(ahorro_combustible+ahorro_tiempo+ahorro_mantenimiento)*100:.1f}%"],
                ["Tiempo", 
                format_currency(ahorro_tiempo, currency, exchange_rate),
                format_currency(ahorro_tiempo * 20, currency, exchange_rate),
                f"{ahorro_tiempo/(ahorro_combustible+ahorro_tiempo+ahorro_mantenimiento)*100:.1f}%"],
                ["Mantenimiento", 
                format_currency(ahorro_mantenimiento, currency, exchange_rate),
                format_currency(ahorro_mantenimiento * 20, currency, exchange_rate),
                f"{ahorro_mantenimiento/(ahorro_combustible+ahorro_tiempo+ahorro_mantenimiento)*100:.1f}%"],
            ]
            
            savings_table = Table(savings_data, colWidths=[120, 110, 130, 100])
            savings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), grey_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke_color),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), beige_color),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(savings_table)
            elements.append(Spacer(1, 20))
            
            # Recomendaciones
            elements.append(Paragraph("Recomendaciones y Próximos Pasos", subtitle_style))
            elements.append(Spacer(1, 10))
            
            # Recomendaciones operativas
            elements.append(Paragraph("Recomendaciones Operativas:", styles["Heading3"]))
            recommendations = [
                f"Implementar las rutas optimizadas para reducir costos operativos.",
                f"Priorizar reabastecimiento de {len(atms_to_restock)} cajeros identificados.",
                f"Utilizar {len(routes)} vehículos para maximizar eficiencia."
            ]
            
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", normal_style))
            
            elements.append(Spacer(1, 10))
            
            # Próximos pasos
            elements.append(Paragraph("Próximos Pasos:", styles["Heading3"]))
            next_steps = [
                "Implementar un sistema de monitoreo continuo de KPIs.",
                "Evaluar impacto en satisfacción de clientes y tiempo de servicio.",
                "Analizar patrones estacionales para ajustar optimización.",
                "Integrar datos de tráfico en tiempo real para mejorar precisión."
            ]
            
            for step in next_steps:
                elements.append(Paragraph(f"• {step}", normal_style))
            
            # Pie de página
            elements.append(Spacer(1, 40))
            elements.append(Paragraph("Este informe fue generado automáticamente por ATM Optimizer.", 
                                    styles["Centered"]))
            
            # Construir PDF
            doc.build(elements)
            
            # Obtener el contenido del buffer
            pdf = buffer.getvalue()
            buffer.close()
            
            return pdf

        # Generar el PDF
        pdf_report = generate_report_pdf(
            current_kpis=current_kpis, 
            improved_kpis=improved_kpis, 
            savings=savings, 
            routes=routes, 
            atms_to_restock=atms_to_restock,
            currency=currency,
            exchange_rate=exchange_rate
        )

        # Botón para descargar informe detallado
        st.download_button(
            label="Descargar Informe Detallado",
            data=pdf_report,
            file_name=f"Informe_Impacto_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            help="Descargar un informe detallado en formato PDF"
        )
        
    else:
        st.error("No se pudieron generar rutas para el análisis. Intenta con diferentes parámetros.")

else:
    # Mostrar instrucciones iniciales
    st.info("Configura los parámetros en el panel lateral y haz clic en 'Calcular Impacto' para generar un análisis detallado.")
    
    # Información introductoria
    st.markdown("""
    ### Análisis de Impacto de la Optimización
    
    Esta herramienta permite evaluar el impacto de la optimización de rutas en diferentes dimensiones:
    
    **Impacto en KPIs de Negocio:**
    - Disponibilidad de efectivo
    - Downtime por agotamiento
    - Eficiencia de capital
    - Días hasta agotamiento
    
    **Impacto Económico:**
    - Ahorro en costos de transporte
    - Reducción de costos operativos
    - Proyección de ahorros a largo plazo
    
    **Análisis Detallado:**
    - Composición del ahorro
    - Comparación por componente
    - Recomendaciones operativas
    
    Para comenzar, configura los parámetros de análisis en el panel lateral y haz clic en "Calcular Impacto".
    """)
    
    # Mostrar imagen ilustrativa
    st.image("https://via.placeholder.com/800x400?text=An%C3%A1lisis+de+Impacto", 
             caption="Ejemplo ilustrativo de análisis de impacto")