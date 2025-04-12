import numpy as np
import pandas as pd
from datetime import datetime, time
from sklearn.ensemble import RandomForestRegressor
import cv2
import os
from typing import Dict, List, Tuple

class TrafficControlSystem:
    def __init__(self):
        self.vehicle_density = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        self.traffic_patterns = {}
        self.model = RandomForestRegressor()
        self.traffic_light_timings = {
            'north_south': 30,  # default timing in seconds
            'east_west': 30
        }
        self.load_historical_data()
        
    def load_historical_data(self):
        """Load historical traffic data if available"""
        try:
            if os.path.exists('traffic_data.csv'):
                self.historical_data = pd.read_csv('traffic_data.csv')
                self.train_model()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            self.historical_data = pd.DataFrame()

    def update_vehicle_density(self, direction: str, count: int):
        """Update vehicle density for a specific direction"""
        self.vehicle_density[direction] = count
        self.adjust_traffic_lights()

    def adjust_traffic_lights(self):
        """Adjust traffic light timings based on current density"""
        # Calculate total vehicles in each direction
        north_south_total = self.vehicle_density['north'] + self.vehicle_density['south']
        east_west_total = self.vehicle_density['east'] + self.vehicle_density['west']
        
        # Adjust timings based on density ratio
        total_vehicles = north_south_total + east_west_total
        if total_vehicles > 0:
            north_south_ratio = north_south_total / total_vehicles
            self.traffic_light_timings['north_south'] = int(60 * north_south_ratio)
            self.traffic_light_timings['east_west'] = 60 - self.traffic_light_timings['north_south']

    def record_traffic_pattern(self, timestamp: datetime, density: Dict[str, int]):
        """Record current traffic pattern for learning"""
        pattern = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'north_density': density['north'],
            'south_density': density['south'],
            'east_density': density['east'],
            'west_density': density['west']
        }
        
        if not hasattr(self, 'historical_data'):
            self.historical_data = pd.DataFrame()
        
        self.historical_data = pd.concat([
            self.historical_data,
            pd.DataFrame([pattern])
        ], ignore_index=True)
        
        # Save to CSV
        self.historical_data.to_csv('traffic_data.csv', index=False)
        
        # Retrain model with new data
        self.train_model()

    def train_model(self):
        """Train the model on historical data"""
        if len(self.historical_data) > 0:
            X = self.historical_data[['hour', 'day_of_week']]
            y = self.historical_data[['north_density', 'south_density', 'east_density', 'west_density']]
            self.model.fit(X, y)

    def predict_traffic_pattern(self, timestamp: datetime) -> Dict[str, int]:
        """Predict traffic density based on time"""
        if len(self.historical_data) > 0:
            X = pd.DataFrame({
                'hour': [timestamp.hour],
                'day_of_week': [timestamp.weekday()]
            })
            prediction = self.model.predict(X)[0]
            return {
                'north': int(prediction[0]),
                'south': int(prediction[1]),
                'east': int(prediction[2]),
                'west': int(prediction[3])
            }
        return self.vehicle_density

    def get_traffic_light_timings(self) -> Dict[str, int]:
        """Get current traffic light timings"""
        return self.traffic_light_timings 