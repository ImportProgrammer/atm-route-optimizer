
import os
import osmnx as ox
import geopandas as gpd

# Crear carpeta destino si no existe
output_dir = "data/bogota_map"
os.makedirs(output_dir, exist_ok=True)

# 1. Descargar límites administrativos de Bogotá
print("Descargando límites de Bogotá...")
bogota_boundary = ox.geocode_to_gdf("Bogotá, Colombia")
bogota_boundary.to_file(os.path.join(output_dir, "bogota_boundary.geojson"), driver="GeoJSON")
bogota_boundary.to_file(os.path.join(output_dir, "bogota_boundary.shp"))
print("Límites de Bogotá guardados.")

# 2. Descargar red vial principal (solo vías para automóviles)
print("Descargando red vial principal de Bogotá...")
G = ox.graph_from_place("Bogotá, Colombia", network_type="drive")

# Convertir a GeoDataFrame de aristas
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Filtrar por vías principales (ej. highway: primary, trunk, motorway)
main_highways = edges[edges['highway'].isin(['motorway', 'trunk', 'primary'])]

# Guardar como GeoJSON y Shapefile
main_highways.to_file(os.path.join(output_dir, "bogota_roads.geojson"), driver="GeoJSON")
main_highways.to_file(os.path.join(output_dir, "bogota_roads.shp"))
print("Red vial principal guardada.")
