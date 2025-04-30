"""
Componentes de mapas para la interfaz de usuario.

Este módulo proporciona componentes Streamlit reutilizables para la
visualización de mapas interactivos de cajeros y rutas.

Componentes principales:
    - atm_status_map: Mapa con estado de cajeros automáticos
    - route_visualization: Visualización de rutas optimizadas 
    - heatmap: Mapa de calor para análisis de actividad
"""

import streamlit as st
import folium
from folium.plugins import MarkerCluster, AntPath
from streamlit_folium import folium_static
import pandas as pd
import numpy as np

def atm_status_map(status_df, carriers_df=None, width=800, height=600, currency="COP", exchange_rate=None):
    """
    Crea un mapa interactivo mostrando el estado actual de los cajeros.
    
    Args:
        status_df: DataFrame con estado actual de cajeros incluyendo coordenadas
        carriers_df: DataFrame opcional con transportadoras
        width: Ancho del mapa en píxeles
        height: Altura del mapa en píxeles
        currency: Moneda a mostrar ("COP" o "USD")
        exchange_rate: Tasa de cambio COP/USD
        
    Returns:
        Componente de mapa folium renderizado en Streamlit
    """
    # Importamos la función de formateo de moneda
    from api.utils.helpers import format_currency
    
    # Verificar que haya datos
    if status_df is None or len(status_df) == 0:
        st.warning("No hay datos de cajeros para mostrar en el mapa.")
        return None
    
    # Crear mapa centrado en el promedio de coordenadas
    center_lat = status_df['latitude'].mean()
    center_lon = status_df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Añadir cajeros al mapa con colores según estado
    for _, row in status_df.iterrows():
        # Determinar color según estado
        if row['status'] == 'Crítico':
            color = 'red'
        elif row['status'] == 'Advertencia':
            color = 'orange'
        else:
            color = 'green'
        
        # Formato para valores monetarios
        current_cash_str = format_currency(row['current_cash'], currency, exchange_rate)
        
        # Crear popup con información detallada
        popup_html = f"""
            <div style="width: 200px">
                <h4>{row['name']}</h4>
                <b>ID:</b> {row['id']}<br>
                <b>Tipo:</b> {row['location_type']}<br>
                <b>Efectivo:</b> {current_cash_str}<br>
                <b>Capacidad:</b> {row['usage_percent']:.1f}%<br>
                <b>Último reabast.:</b> {row['last_restock']}<br>
                <b>Días hasta agotamiento:</b> {row['days_until_empty']:.1f}
            </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['name']} - {row['status']}",
            icon=folium.Icon(color=color, icon='money-bill-alt', prefix='fa')
        ).add_to(m)
    
    # Añadir transportadoras si se proporcionan
    if carriers_df is not None and len(carriers_df) > 0:
        for _, row in carriers_df.iterrows():
            popup_html = f"""
                <div style="width: 200px">
                    <h4>{row['name']}</h4>
                    <b>ID:</b> {row['id']}<br>
                    <b>Vehículos:</b> {row['vehicles']}<br>
                    <b>Zona:</b> {row['service_area']}<br>
                </div>
            """
            
            folium.Marker(
                location=[row['base_latitude'], row['base_longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Transportadora: {row['name']}",
                icon=folium.Icon(color='black', icon='truck', prefix='fa')
            ).add_to(m)
    
    # Renderizar el mapa en Streamlit
    st.write("### Mapa de Estado de Cajeros")
    folium_static(m, width=width, height=height)
    
    return m

def route_visualization(locations, routes, carrier, width=800, height=600):
    """
    Visualiza rutas optimizadas en un mapa.
    
    Args:
        locations: Lista de ubicaciones (cajeros y transportadora)
        routes: Lista de rutas optimizadas
        carrier: Información de la transportadora
        width: Ancho del mapa en píxeles
        height: Altura del mapa en píxeles
        
    Returns:
        Componente de mapa folium renderizado en Streamlit
    """
    if not routes or not locations or carrier is None:
        st.warning("No hay rutas para visualizar.")
        return None
    
    # Crear mapa
    m = folium.Map(
        location=[carrier['base_latitude'], carrier['base_longitude']], 
        zoom_start=12
    )
    
    # Colores para las rutas
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Añadir transportadora
    folium.Marker(
        location=[carrier['base_latitude'], carrier['base_longitude']],
        popup=f"<b>{carrier['name']}</b><br>Base de operaciones",
        icon=folium.Icon(color='black', icon='truck', prefix='fa')
    ).add_to(m)
    
    # Añadir cada ruta
    for i, route_data in enumerate(routes):
        route = route_data['route']
        color = colors[i % len(colors)]
        
        # Coordenadas de la ruta
        route_coords = []
        for node_idx in route:
            location = locations[node_idx]
            route_coords.append([location['lat'], location['lon']])
        
        # Usar AntPath para mostrar dirección
        AntPath(
            locations=route_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"Ruta {i+1}: {route_data['distance']:.1f} km",
            delay=1000,
            dash_array=[10, 20],
            pulse_color='#FFF'
        ).add_to(m)
        
        # Añadir marcadores para los cajeros
        for j, node_idx in enumerate(route):
            location = locations[node_idx]
            if location['type'] == 'atm':
                # Añadir marcador numerado
                folium.Marker(
                    location=[location['lat'], location['lon']],
                    popup=f"<b>{location['name']}</b><br>Parada #{j}",
                    icon=folium.Icon(color=color, icon='money-bill-alt', prefix='fa')
                ).add_to(m)
    
    # Renderizar mapa en Streamlit
    # st.write("### Visualización de Rutas Optimizadas")
    folium_static(m, width=width, height=height)
    
    return m