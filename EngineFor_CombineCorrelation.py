import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CombineCorrelationEngine:
    def __init__(self, main_app):
        self.app = main_app

    def get_global_depth_range(self, project_chain):
        """Calculates global depth range across all wells in the chain."""
        global_min = float('inf')
        global_max = float('-inf')
        
        for p in project_chain:
            for well_key in ['ref_well', 'off_well']:
                well = p['ui_state'][well_key]
                df = self.app.las_cache.get(well, pd.DataFrame())
                if not df.empty:
                    global_min = min(global_min, df['Depth'].min())
                    global_max = max(global_max, df['Depth'].max())
        
        if global_min == float('inf'):
            return None, None
            
        m = self.app.depth_multiplier
        global_min *= m
        global_max *= m
        pad = (global_max - global_min) * 0.05
        return global_min - pad, global_max + pad

    def get_well_data(self, project_chain, well_name):
        """Extracts all relevant metadata for a well from the project chain."""
        data = {
            'las': self.app.las_cache.get(well_name, pd.DataFrame()),
            'inpefa': None, # (grid, values)
            'mrgc': None,
            'rf': None,
            'markers_seq': None,
            'markers_sand': None
        }
        
        # Find which project in the chain contains this well's specific results
        for p in project_chain:
            # INPEFA
            if p['last_plot_args']:
                # grid, log_A, log_B, pi, pj, ref_name, off_name
                l_args = p['last_plot_args']
                if p['ui_state']['ref_well'] == well_name:
                    data['inpefa'] = (l_args[0], l_args[1])
                elif p['ui_state']['off_well'] == well_name:
                    data['inpefa'] = (l_args[0], l_args[2])
            
            # MRGC
            if well_name in p.get('mrgc_results', {}):
                data['mrgc'] = p['mrgc_results'][well_name]
            
            # RF
            if well_name in p.get('rf_results', {}):
                data['rf'] = p['rf_results'][well_name]

        # Markers are usually global in app state, but we filter them for this well
        data['markers_seq'] = self.app.geology_engine.get_markers(well_name, self.app.data_loader.all_markers)
        data['markers_sand'] = self.app.sand_engine.get_markers(well_name, self.app.data_loader.sand_markers)
        
        return data

    def get_connection_data(self, project_chain, index):
        """Extracts alignment data between two projects at the given index."""
        if index >= len(project_chain): return None
        p = project_chain[index]
        if not p['last_plot_args']: return None
        
        # grid, log_A, log_B, pi, pj, ref_name, off_name
        return {
            'grid': p['last_plot_args'][0],
            'pi': p['last_plot_args'][3],
            'pj': p['last_plot_args'][4],
            'ref_name': p['ui_state']['ref_well'],
            'off_name': p['ui_state']['off_well']
        }
