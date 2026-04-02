"""
EngineFor_SandPlot.py
Engine for handling sand plot marker retrieval and processing
"""
import pandas as pd
import re

class SandPlotEngine:
    @staticmethod
    def get_markers(well_name, sand_markers_df, id_col="Well identifier (Well name)"):
        """
        Retrieves sand plot markers (Surface, MD) for a specific well.
        Always returns MD in METERS.
        """
        if sand_markers_df is None or sand_markers_df.empty:
            return pd.DataFrame()
            
        # Standardize search name
        def standardize_search(val):
            try:
                num = int(val)
                return f"S-{num:03d}"
            except:
                s = str(val).upper().strip()
                if s.startswith("S-") or s.startswith("SBK-"):
                    match = re.search(r'\d+', s)
                    if match: return f"S-{int(match.group()):03d}"
                    return s
                match = re.search(r'\d+', s)
                if match: return f"S-{int(match.group()):03d}"
                return s

        search_name = standardize_search(well_name)
        
        # Filter markers for this well
        if id_col not in sand_markers_df.columns:
            return pd.DataFrame()

        mask = sand_markers_df[id_col].apply(standardize_search) == search_name
        well_markers = sand_markers_df[mask].copy()
        
        if well_markers.empty:
            return pd.DataFrame()
            
        # Conversion Logic
        md_col_name = 'MD'
        if md_col_name not in well_markers.columns:
            # Try to find MD column
            for c in well_markers.columns:
                if 'MD' in c.upper():
                    md_col_name = c
                    break
        
        if md_col_name not in well_markers.columns:
             return pd.DataFrame()

        # Check if column name suggests units
        is_already_meters = any(u in md_col_name.upper() for u in ['(M)', '[M]', 'METERS'])
        
        if not is_already_meters:
            # Convert FT to M
            well_markers[md_col_name] = well_markers[md_col_name] * 0.3048
            
        return well_markers[['Surface', md_col_name]].rename(columns={md_col_name: 'MD'})
