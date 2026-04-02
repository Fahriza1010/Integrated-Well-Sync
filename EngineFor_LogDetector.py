"""
EngineFor_LogDetector.py
Engine for the Log Detector feature.
Provides logic to find which wells contain specific sets of logs.
"""
import os
import lasio

class LogDetectorEngine:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_all_unique_logs(self):
        """
        Scans all available wells to find every unique log curve name.
        """
        unique_logs = set()
        well_names = self.data_loader.get_well_names()
        
        for well in well_names:
            curves = self.data_loader.get_available_curves(well)
            for curve in curves:
                unique_logs.add(curve.upper())
        
        return sorted(list(unique_logs))

    def find_wells_with_logs(self, selected_logs):
        """
        Returns a list of well names that contain ALL of the logs in selected_logs.
        """
        if not selected_logs:
            return []
            
        matching_wells = []
        well_names = self.data_loader.get_well_names()
        selected_logs_upper = [log.upper() for log in selected_logs]
        
        for well in well_names:
            available_curves = [c.upper() for c in self.data_loader.get_available_curves(well)]
            # Check if every selected log is available in this well
            if all(log in available_curves for log in selected_logs_upper):
                matching_wells.append(well)
                
        return matching_wells
    def export_availability_matrix(self, selected_logs):
        """
        Creates a DataFrame showing availability of selected logs across all wells.
        
        Parameters:
        -----------
        selected_logs : list of str
            The log names to include in the overview.
            
        Returns:
        --------
        pd.DataFrame
            A matrix with wells as rows and logs as columns.
        """
        import pandas as pd
        well_names = self.data_loader.get_well_names()
        selected_logs_upper = [log.upper() for log in selected_logs]
        
        data = []
        for well in well_names:
            row = {"Well Name": well}
            available_curves = [c.upper() for c in self.data_loader.get_available_curves(well)]
            for log in selected_logs_upper:
                row[log] = "Available" if log in available_curves else "Not Found"
            data.append(row)
            
        return pd.DataFrame(data)
