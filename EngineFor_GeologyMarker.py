"""
EngineFor_GeologyMarker.py
Engine for handling geological marker retrieval and processing
"""
import pandas as pd
import re

class GeologyMarkerEngine:
    @staticmethod
    def get_markers(well_name, all_markers_df, id_col="Well identifier (Well name)"):
        """
        Retrieves geology markers (Surface, MD) for a specific well.
        Always returns MD in METERS. (Assumes input MD from Excel is in Feet).
        """
        if all_markers_df is None or all_markers_df.empty:
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
        mask = all_markers_df[id_col].apply(standardize_search) == search_name
        well_markers = all_markers_df[mask].copy()
        
        if well_markers.empty:
            return pd.DataFrame()
            
        # Conversion Logic (Excel MD is usually in FT)
        md_col_name = 'MD'
        # Check if column name suggests units (already in meters)
        is_already_meters = any(u in md_col_name.upper() for u in ['(M)', '[M]', 'METERS'])
        
        if not is_already_meters:
            # Convert FT to M
            well_markers['MD'] = well_markers['MD'] * 0.3048
            
        return well_markers[['Surface', 'MD']]
