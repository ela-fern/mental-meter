from datetime import datetime, timedelta
import pandas as pd

class PHQ9Scorer:
    def __init__(self):
        self.num_symptoms = 9
        self.max_score = 27
        
    def calculate_rolling_score(self, database_manager, end_date):
        """Calculate PHQ-9 score based on symptom frequency in past 14 days"""
        rolling_data = database_manager.get_rolling_data(end_date, days=14)
        
        if rolling_data.empty:
            return 0
        
        total_score = 0
        
        for i in range(self.num_symptoms):
            symptom_col = f'phq9_{i}'
            if symptom_col in rolling_data.columns:
                # Count days with this symptom
                symptom_days = rolling_data[symptom_col].sum()
                
                # Apply clinical scoring
                if symptom_days <= 1:
                    score = 0
                elif symptom_days <= 6:
                    score = 1
                elif symptom_days <= 10:
                    score = 2
                else:  # 11-14 days
                    score = 3
                
                total_score += score
        
        return min(total_score, self.max_score)
    
    def get_severity_level(self, score):
        """Get severity level based on PHQ-9 score"""
        if score <= 4:
            return "Minimal"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        elif score <= 19:
            return "Moderately Severe"
        else:
            return "Severe"
    
    def get_severity_color(self, score):
        """Get color for severity level"""
        severity = self.get_severity_level(score)
        colors = {
            "Minimal": "green",
            "Mild": "yellow",
            "Moderate": "orange",
            "Moderately Severe": "red",
            "Severe": "darkred"
        }
        return colors.get(severity, "gray")

class GAD7Scorer:
    def __init__(self):
        self.num_symptoms = 7
        self.max_score = 21
        
    def calculate_rolling_score(self, database_manager, end_date):
        """Calculate GAD-7 score based on symptom frequency in past 14 days"""
        rolling_data = database_manager.get_rolling_data(end_date, days=14)
        
        if rolling_data.empty:
            return 0
        
        total_score = 0
        
        for i in range(self.num_symptoms):
            symptom_col = f'gad7_{i}'
            if symptom_col in rolling_data.columns:
                # Count days with this symptom
                symptom_days = rolling_data[symptom_col].sum()
                
                # Apply clinical scoring (same as PHQ-9)
                if symptom_days <= 1:
                    score = 0
                elif symptom_days <= 6:
                    score = 1
                elif symptom_days <= 10:
                    score = 2
                else:  # 11-14 days
                    score = 3
                
                total_score += score
        
        return min(total_score, self.max_score)
    
    def get_severity_level(self, score):
        """Get severity level based on GAD-7 score"""
        if score <= 4:
            return "Minimal"
        elif score <= 9:
            return "Mild"
        elif score <= 14:
            return "Moderate"
        else:
            return "Severe"
    
    def get_severity_color(self, score):
        """Get color for severity level"""
        severity = self.get_severity_level(score)
        colors = {
            "Minimal": "green",
            "Mild": "yellow",
            "Moderate": "orange",
            "Severe": "red"
        }
        return colors.get(severity, "gray")
