# ATM Optimizer - Sistema de Optimización de Rutas y Gestión Integral de Cajeros Automáticos


## Descripción

ATM Optimizer es una aplicación de demostración que combina sistemas de georreferenciación (GIS) con inteligencia artificial para optimizar la gestión de rutas y provisiones de efectivo para cajeros automáticos. La solución incorpora también la predicción y prevención de fallas técnicas, permitiendo una gestión integral de la red de ATMs.

La aplicación está construida con herramientas de código abierto, lo que la hace económicamente accesible y fácilmente escalable, manteniendo un enfoque en la visualización geoespacial interactiva y el análisis predictivo.

## Características Principales

### Gestión de Efectivo
- **Monitoreo en tiempo real** del estado de los cajeros
- **Predicción de demanda** mediante modelos de IA
- **Optimización de rutas** de reabastecimiento
- **Análisis de impacto económico** de las estrategias implementadas
- **Simulación de escenarios** para evaluación de estrategias

### Gestión Técnica
- **Predicción de fallas técnicas** por componente
- **Análisis de disponibilidad técnica** y factores de riesgo
- **Optimización de rutas de mantenimiento preventivo**
- **Análisis costo-beneficio** de estrategias de mantenimiento
- **Integración con rutas de reabastecimiento**

## Beneficios

- Reducción significativa de costos operativos
- Disminución del tiempo de inactividad de cajeros
- Mejor utilización del efectivo en circulación
- Optimización de recursos técnicos
- Toma de decisiones basada en datos

## Demostración

La aplicación de demostración incluye:

1. **Dashboard de Estado Actual**: Visión general del estado de los cajeros, métricas clave y visualización geoespacial.
2. **Predicciones de Demanda**: Análisis predictivo de la demanda futura de efectivo por cajero.
3. **Optimización de Rutas**: Generación automática de rutas óptimas para reabastecimiento.
4. **Análisis de Impacto**: Evaluación del impacto económico y operativo de la optimización.
5. **Simulador Avanzado**: Herramienta para modelar escenarios complejos incluyendo análisis de disponibilidad técnica.

## Tecnologías Utilizadas

- **Backend**: Python, OR-Tools, scikit-learn
- **Frontend**: Streamlit, Plotly, Folium
- **Análisis de Datos**: Pandas, NumPy
- **Visualización Geoespacial**: Folium, streamlit-folium
- **Generación de Informes**: ReportLab

## Instalación y Ejecución

### Requisitos previos
- Python 3.8+
- pip

### Instalación

1. Clonar el repositorio:
```
git clone https://github.com/ImportProgrammer/atm-route-optimizer.git
cd atm-route-optimizer
```

2. Instalar dependencias:
```
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```
streamlit run frontend/app.py
```

## Estructura del Proyecto

```
atm-optimizer/
├── api/
│   ├── data/           # Conectores de datos y simulación
│   ├── models/         # Modelos de predicción y optimización
│   ├── utils/          # Utilidades y helpers
│   └── routes/         # Definición de rutas de API
├── frontend/
│   ├── components/     # Componentes reutilizables
│   ├── pages/          # Páginas de la aplicación
│   │   ├── 1_dashboard.py
│   │   ├── 2_predictions.py
│   │   ├── 3_route_optimization.py
│   │   ├── 4_impact_analysis.py
│   │   └── 5_simulator.py
│   ├── utils/          # Utilidades frontend
│   └── app.py          # Punto de entrada principal
└── requirements.txt    # Dependencias
```

## Personalización

La aplicación es totalmente personalizable y puede adaptarse a diferentes necesidades:

- Integración con fuentes de datos reales
- Ajuste de modelos predictivos para escenarios específicos
- Incorporación de restricciones operativas adicionales
- Ampliación a otros componentes técnicos o necesidades específicas

## Contribución

Este proyecto es una demostración de capacidades. Para más información sobre implementaciones personalizadas, contáctenos.

## Licencia

Copyright © 2025. Todos los derechos reservados.

## Contacto

Para más información, consultas o demostraciones, contacte a:
- Email: import.games.dev@gmail.com