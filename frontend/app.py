"""
Aplicación Streamlit para optimización de rutas y provisiones de cajeros automáticos.

Esta aplicación sirve como la página principal (Home) para el sistema de gestión
de cajeros automáticos. Desde aquí se puede acceder a las diferentes secciones
a través del menú de navegación lateral de Streamlit.
"""

import streamlit as st
import os
import sys
import folium
from streamlit_folium import folium_static

# Agregar la ruta del proyecto al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Importar los modulos nocesarios y que hacen falta
from api.data.db_connector import create_db_connection, load_atm_data
from api.data.simulation import simulate_current_atm_status, generate_sample_carriers

# Configuración de la página
st.set_page_config(
    page_title="ATM Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal y descripción del proyecto
st.title("Sistema de Optimización de Rutas y Provisiones para Cajeros Automáticos")
st.subheader("Gestión inteligente de efectivo con GIS e IA")

# Contenido de la página principal
st.markdown("""
## Bienvenido a la demostración del sistema de optimización de cajeros

Esta aplicación permite gestionar eficientemente la distribución de efectivo en cajeros automáticos 
mediante técnicas avanzadas de georreferenciación e inteligencia artificial.

### Objetivos de la DEMO
 
- **Optimizar la operación del centro de efectivo**, asegurando disponibilidad de efectivo en cajeros
- **Reducir costos logísticos** mediante rutas eficientes y planificación inteligente
- **Aumentar la precisión de pronósticos** a más del 87%
- **Reducir el downtime por agotamiento** a menos del 15%
- **Mejorar la eficiencia operativa** mediante decisiones basadas en datos

### Cómo usar la aplicación:

Utilice el menú de navegación en la barra lateral izquierda para acceder a las diferentes secciones del sistema.
""")

# Añadir una imagen o gráfico ilustrativo en la página principal
# Cargar datos para el mapa
@st.cache_data(ttl=600)  # Cache por 10 minutos
def load_map_data():
    """Carga datos para mostrar en el mapa de la página principal"""
    try:
        # Intentar cargar datos reales
        engine = create_db_connection()
        atms_df, carriers_df, _ = load_atm_data(engine)
        
        # Si no hay datos, generar datos de ejemplo
        if len(atms_df) == 0:
            st.warning("No se encontraron datos en la base de datos. Usando datos simulados.")
            atms_df = pd.DataFrame()  # Los datos se generarán en simulate_current_atm_status
            carriers_df = generate_sample_carriers(3)
        
        # Simular estado actual
        current_status = simulate_current_atm_status(atms_df)
        
        return current_status, carriers_df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        # Generar datos de ejemplo como fallback
        import pandas as pd
        atms_df = pd.DataFrame()
        current_status = simulate_current_atm_status(atms_df)
        carriers_df = generate_sample_carriers(3)
        
        return current_status, carriers_df

# Crear mapa para la página principal
def create_overview_map(atms_df, carriers_df):
    """
    Crea un mapa general mostrando todos los cajeros y transportadoras
    """
    # Verificar que haya datos
    if atms_df is None or len(atms_df) == 0:
        st.warning("No hay datos de cajeros para mostrar en el mapa.")
        return None
    
    # Crear mapa centrado en el promedio de coordenadas
    center_lat = atms_df['latitude'].mean()
    center_lon = atms_df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Añadir cajeros al mapa (todos con el mismo estilo)
    for _, row in atms_df.iterrows():
        # Crear popup simple con información básica
        popup_html = f"""
            <div style="width: 150px">
                <h4>{row['name']}</h4>
                <b>ID:</b> {row['id']}<br>
                <b>Tipo:</b> {row['location_type']}<br>
            </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['name']}",
            icon=folium.Icon(color='blue', icon='money-bill-alt', prefix='fa')
        ).add_to(m)
    
    # Añadir transportadoras si se proporcionan
    if carriers_df is not None and len(carriers_df) > 0:
        for _, row in carriers_df.iterrows():
            popup_html = f"""
                <div style="width: 150px">
                    <h4>{row['name']}</h4>
                    <b>ID:</b> {row['id']}<br>
                    <b>Vehículos:</b> {row['vehicles']}<br>
                </div>
            """
            
            folium.Marker(
                location=[row['base_latitude'], row['base_longitude']],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"Transportadora: {row['name']}",
                icon=folium.Icon(color='black', icon='truck', prefix='fa')
            ).add_to(m)
    
    return m

# Cargar datos y mostrar mapa en la página principal
try:
    atms_data, carriers_data = load_map_data()
    
    st.write("### Mapa de distribución de cajeros y transportadoras")
    map_obj = create_overview_map(atms_data, carriers_data)
    
    if map_obj:
        folium_static(map_obj, width=800, height=500)
        st.caption("Distribución geográfica de cajeros (azul) y transportadoras (negro) en Bogotá.")
    else:
        # Si no se pudo crear el mapa, mostrar imagen de respaldo
        st.image("https://via.placeholder.com/800x400?text=ATM+Optimization+System", 
                caption="Visualización conceptual del sistema de optimización")
except Exception as e:
    st.error(f"Error al generar el mapa: {e}")
    # Mostrar imagen de respaldo en caso de error
    st.image("https://via.placeholder.com/800x400?text=ATM+Optimization+System", 
            caption="Visualización conceptual del sistema de optimización")

# Información del proyecto en el sidebar
#st.sidebar.markdown("---")


# Información sobre modo de desarrollo
if st.sidebar.checkbox("Modo desarrollo", False):
    st.sidebar.info("Desarrollado con Streamlit, Folium y OR-Tools")
    st.sidebar.write("Versión: 0.1.0")
    st.sidebar.write("Estructura actual:")
    st.sidebar.code("""
    atm-optimizer/
    ├── api/
    │   ├── data/
    │   ├── models/
    │   ├── routes/
    │   ├── utils/
    │   └── api.py
    ├── frontend/
    │   ├── components/
    │   ├── pages/
    │   │   ├── 1_dashboard.py
    │   │   ├── 2_predictions.py
    │   │   ├── 3_route_optimization.py
    │   │   ├── 4_impact_analysis.py
    │   │   └── 5_simulator.py
    │   ├── utils/
    │   └── app.py
    """)

# Main execution
if __name__ == "__main__":
    pass