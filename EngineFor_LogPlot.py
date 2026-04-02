"""
EngineFor_LogPlot.py
Specialized engine for Log Plot analysis (Lithology and Vshale)
"""
import numpy as np

class LogPlotEngine:
    @staticmethod
    def classify_lithology(gr_values, res_values=None):
        """
        Classifies lithology based on GR and optional Resistivity values.
        Vectorized version for performance.
        
        Returns:
        --------
        np.ndarray
            Array of indices corresponding to lithology types:
            0: Coal, 1: Sand, 2: Shaly sand, 3: Shale
        """
        if gr_values is None or len(gr_values) == 0:
            return np.array([], dtype=int)
            
        if res_values is None:
            res_values = np.full(len(gr_values), np.nan)
            
        # Initialize with Shale (default)
        idx = np.full(len(gr_values), 3, dtype=int)
        
        # Coal condition: GR <= 32.5 or Resistivity > 5000
        coal_mask = (gr_values <= 32.5) | (res_values > 5000)
        # Sand condition: 32.5 < GR <= 60
        sand_mask = (~coal_mask) & (gr_values > 32.5) & (gr_values <= 60)
        # Shaly sand condition: 60 < GR <= 70
        shaly_sand_mask = (~coal_mask) & (~sand_mask) & (gr_values > 60) & (gr_values <= 70)
        
        idx[coal_mask] = 0
        idx[sand_mask] = 1
        idx[shaly_sand_mask] = 2
        
        return idx

    @staticmethod
    def calculate_vshale(gr_values):
        """
        Calculates Vshale using various methods.
        
        Returns:
        --------
        dict
            Dictionary containing 'ish' (linear), 'clavier', and 'steiber' results.
        """
        if gr_values is None or len(gr_values) == 0:
            return {'ish': np.array([]), 'clavier': np.array([]), 'steiber': np.array([])}

        gr_min, gr_max = np.nanmin(gr_values), np.nanmax(gr_values)
        
        if gr_max <= gr_min:
            z = np.zeros(len(gr_values))
            return {'ish': z, 'clavier': z, 'steiber': z}
            
        ish = np.clip((gr_values - gr_min) / (gr_max - gr_min), 0, 1)
        clavier = 1.7 - np.sqrt(np.maximum(0, 3.38 - (ish + 0.7)**2))
        steiber = 0.5 * ish / (1.5 - ish)
        
        return {
            'ish': ish,
            'clavier': clavier,
            'steiber': steiber
        }
