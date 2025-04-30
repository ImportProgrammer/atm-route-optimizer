"""
Aplicación Streamlit para optimización de rutas y provisiones de cajeros automáticos.

Esta aplicación sirve como la página principal (Home) para el sistema de gestión
de cajeros automáticos. Desde aquí se puede acceder a las diferentes secciones
a través del menú de navegación lateral de Streamlit.
"""

import streamlit as st
import os
import sys

# Agregar la ruta del proyecto al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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