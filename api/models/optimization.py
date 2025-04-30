"""
Módulo para la predicción de demanda de efectivo.

Este módulo implementa algoritmos para predecir la demanda futura de efectivo
en cajeros automáticos basados en datos históricos, patrones y factores externos.
"""


import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import random
from api.models.prediction import predict_cash_demand

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Advertencia: OR-Tools no está instalado. Se usará un algoritmo de optimización básico.")

def select_atms_for_restock(predictions_df, date, min_priority=2, restrictions_df=None):
    day_predictions = predictions_df[predictions_df['date'] == date].copy()
    selected_atms = day_predictions[day_predictions['priority'] >= min_priority].copy()
    if restrictions_df is not None and len(restrictions_df) > 0:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_of_week = date_obj.weekday()
        day_restrictions = restrictions_df[
            (restrictions_df['day_of_week'] == day_of_week) &
            (restrictions_df['restock_allowed'] == False)
        ]
        if len(day_restrictions) > 0:
            restricted_atms = day_restrictions['atm_id'].unique()
            selected_atms = selected_atms[~selected_atms['atm_id'].isin(restricted_atms)].copy()
            print(f"Se excluyeron {len(restricted_atms)} cajeros debido a restricciones.")
    return selected_atms

def limit_atms_for_optimization(atms_to_restock, max_atms=15):
    if len(atms_to_restock) > max_atms:
        sorted_atms = atms_to_restock.sort_values(['priority', 'days_until_empty'], ascending=[False, True])
        return sorted_atms.head(max_atms)
    else:
        return atms_to_restock

def calculate_distance_matrix(carrier, atms_to_restock):
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    locations = [{
        'id': carrier['id'],
        'name': carrier['name'],
        'lat': carrier['base_latitude'],
        'lon': carrier['base_longitude'],
        'type': 'carrier'
    }]
    for _, atm in atms_to_restock.iterrows():
        locations.append({
            'id': atm['atm_id'],
            'name': atm['name'],
            'lat': atm['latitude'],
            'lon': atm['longitude'],
            'type': 'atm'
        })
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = 0 if i == j else haversine_distance(
                locations[i]['lat'], locations[i]['lon'],
                locations[j]['lat'], locations[j]['lon']
            )
    place_names = [loc['name'] for loc in locations]
    return distance_matrix, locations, place_names

def create_data_model(distance_matrix, num_vehicles=1, demands=None, vehicle_capacity=None):
    data = {}
    data['distance_matrix'] = np.round(distance_matrix * 1000).astype(int).tolist()
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    if demands is None:
        demands = [1] * (len(distance_matrix) - 1)
    data['demands'] = [0] + demands
    if vehicle_capacity is None:
        if num_vehicles > 1:
            total_demand = sum(demands)
            vehicle_capacity = max((total_demand // num_vehicles) + 1, 2)
        else:
            vehicle_capacity = sum(demands) + 1
    data['vehicle_capacities'] = [vehicle_capacity] * num_vehicles
    return data

def solve_vrp(data):
    if not ORTOOLS_AVAILABLE:
        return None, None, None
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, "Capacity"
    )
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30
    solution = routing.SolveWithParameters(search_parameters)
    return manager, routing, solution

def get_solution_routes(manager, routing, solution, place_names):
    routes = []
    if not solution:
        return routes
    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        if not routing.IsEnd(solution.Value(routing.NextVar(index))):
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            routes.append({
                'vehicle_id': vehicle_id,
                'route': route,
                'distance': route_distance / 1000.0
            })
    return routes

def optimize_routes_for_date(predictions_df, date, carrier_df, atm_df, 
                            min_priority=2, max_atms=15, num_vehicles=None):
    atms_to_restock = select_atms_for_restock(predictions_df, date, min_priority)
    if len(atms_to_restock) == 0:
        return [], pd.DataFrame(), None, None
    atms_to_restock = pd.merge(
        atms_to_restock,
        atm_df[['id', 'name', 'latitude', 'longitude', 'max_capacity']],
        left_on='atm_id',
        right_on='id'
    )
    atms_to_restock = limit_atms_for_optimization(atms_to_restock, max_atms)
    selected_carrier = carrier_df.iloc[0]
    distance_matrix, locations, place_names = calculate_distance_matrix(selected_carrier, atms_to_restock)
    if num_vehicles is None:
        num_vehicles = min(3, selected_carrier['vehicles'])
    else:
        num_vehicles = min(num_vehicles, selected_carrier['vehicles'])
    demands = [1] * (len(locations) - 1)
    if num_vehicles > 1:
        capacity_per_vehicle = max(2, (len(demands) // num_vehicles) + 1)
    else:
        capacity_per_vehicle = len(demands) + 1
    data = create_data_model(distance_matrix, num_vehicles, demands, vehicle_capacity=capacity_per_vehicle)
    manager, routing, solution = solve_vrp(data)
    if solution:
        routes = get_solution_routes(manager, routing, solution, place_names)
    else:
        routes = []
    return routes, atms_to_restock, selected_carrier, locations

def simulate_scenario(predictions_df, atm_df, carrier_df, 
                      date, num_vehicles=None, min_priority=None, max_atms=15):
    min_priority = min_priority if min_priority is not None else 2
    atm_copy = atm_df.copy()
    atm_copy['daily_consumption'] *= np.random.uniform(1.1, 1.3, size=len(atm_copy))
    atm_copy['current_cash'] *= np.random.uniform(0.85, 1.0, size=len(atm_copy))
    predictions_copy = predict_cash_demand(atm_copy, days_ahead=7)
    if isinstance(carrier_df, pd.DataFrame) and len(carrier_df) > 0:
        carrier_copy = carrier_df.copy()
    else:
        return [], pd.DataFrame(), None, None
    return optimize_routes_for_date(
        predictions_df=predictions_copy,
        date=date,
        carrier_df=carrier_copy,
        atm_df=atm_copy,
        min_priority=min_priority,
        num_vehicles=num_vehicles,
        max_atms=max_atms
    )
