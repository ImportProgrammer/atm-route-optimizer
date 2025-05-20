"""
Módulo para la conexión a la base de datos PostgreSQL.

Este módulo proporciona funciones para conectarse a la base de datos PostgreSQL
y cargar datos de cajeros automáticos, transportadoras, restricciones y transacciones.

Funciones principales:
    - create_db_connection: Establece conexión con la base de datos
    - load_atm_data: Carga datos de cajeros automáticos
    - load_carrier_data: Carga datos de transportadoras
    - load_restrictions: Carga restricciones para reabastecimiento
    - load_transactions: Carga historial de transacciones
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import warnings

# Ignorar las advertencias para una visualización mas limpia
warnings.filterwarnings('ignore')

def create_db_connection():
    """
    Se crean las conexiones a la base de datos
    """
    load_dotenv()

    # Obtener los valores del archivo .env
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'yourpassword')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'atm_optimizer')

    # Cadena de conexión para SQLAlchemy
    conn_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(conn_string)

    print(f"Conectado a: {db_name} en {db_host}")
    return engine

# Función para cargar datos consolidados
def load_atm_data(engine=None):
    """
    Carga todos los datos necesarios para la optimización de cajeros.
    
    Args:
        engine: SQLAlchemy engine. Si es None, se crea uno nuevo.
        
    Returns:
        Tuple con (atms_df, carriers_df, restrictions_df)
    """
    if engine is None:
        engine = create_db_connection()
        
    try:
        # Cargar datos de cajeros
        query_atms = """
        SELECT id, name, 
               ST_X(location::geometry) as longitude, 
               ST_Y(location::geometry) as latitude,
               capacity, cash_type, location_type, 
               min_threshold, max_capacity
        FROM atms;
        """
        atms_df = pd.read_sql(query_atms, engine)
        print(f"Cajeros cargados: {len(atms_df)}")
        
        # Cargar datos de transportadoras
        query_carriers = """
        SELECT id, name, 
               ST_X(base_location::geometry) as base_longitude, 
               ST_Y(base_location::geometry) as base_latitude,
               capacity, vehicles, service_area
        FROM carriers;
        """
        carriers_df = pd.read_sql(query_carriers, engine)
        print(f"Transportadoras cargadas: {len(carriers_df)}")
        
        # Cargar restricciones
        query_restrictions = """
        SELECT atm_id, day_of_week, open_time, close_time, restock_allowed
        FROM restrictions;
        """
        restrictions_df = pd.read_sql(query_restrictions, engine)
        print(f"Restricciones cargadas: {len(restrictions_df)}")
        
        return atms_df, carriers_df, restrictions_df
    
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        print("Devolviendo DataFrames vacíos. Se puede usar simulación para generar datos.")
        
        # Crear DataFrames vacíos con las columnas correctas
        atms_df = pd.DataFrame(columns=[
            'id', 'name', 'longitude', 'latitude', 'capacity', 
            'cash_type', 'location_type', 'min_threshold', 'max_capacity'
        ])
        
        carriers_df = pd.DataFrame(columns=[
            'id', 'name', 'base_longitude', 'base_latitude', 
            'capacity', 'vehicles', 'service_area'
        ])
        
        restrictions_df = pd.DataFrame(columns=[
            'atm_id', 'day_of_week', 'open_time', 'close_time', 'restock_allowed'
        ])
        
        return atms_df, carriers_df, restrictions_df

def load_transactions(engine=None, limit=1000, days=30):
    """
    Carga transacciones de cajeros para análisis.
    
    Args:
        engine: SQLAlchemy engine. Si es None, se crea uno nuevo.
        limit: Número máximo de transacciones a cargar
        days: Cargar transacciones de los últimos X días
        
    Returns:
        DataFrame con transacciones o None si hay un error
    """
    if engine is None:
        engine = create_db_connection()
        
    try:
        query = f"""
        SELECT transaction_id, atm_id, transaction_date, 
               transaction_type, amount
        FROM transactions
        WHERE transaction_date >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY transaction_date DESC
        LIMIT {limit};
        """
        
        transactions_df = pd.read_sql(query, engine)
        print(f"Transacciones cargadas: {len(transactions_df)}")
        
        return transactions_df
    
    except Exception as e:
        print(f"Error al cargar transacciones: {e}")
        return None