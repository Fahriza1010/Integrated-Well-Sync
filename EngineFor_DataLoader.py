"""
EngineFor_DataLoader.py
Data loading engine for Well Correlation application
Handles Excel markers and LAS file loading
"""
import os
import pandas as pd
import numpy as np
import lasio
from scipy.interpolate import interp1d


class DataLoader:
    """
    Handles loading of Excel and LAS data for 27 wells.
    """
    def __init__(self, marker_file_path, las_dir):
        self.marker_file_path = marker_file_path
        self.las_dir = las_dir
        self.wells_info = None
        self.all_markers = None
        self.sand_markers = None
        self.id_col = "Well identifier (Well name)"
        self.load_markers()

    def load_markers(self):
        if not os.path.exists(self.marker_file_path):
            raise FileNotFoundError(f"Marker file not found: {self.marker_file_path}")
        
        # Load main markers (default sheet)
        df_geo = pd.read_excel(self.marker_file_path, sheet_name=0) 
        
        # Load Sand markers (Reservoir (Sand))
        try:
            df_sand = pd.read_excel(self.marker_file_path, sheet_name="Reservoir (Sand)")
        except Exception as e:
            print(f"Warning: 'Reservoir (Sand)' sheet not found or error loading: {e}")
            df_sand = pd.DataFrame()

        id_col = "Well identifier (Well name)"
        
        # Standardize well names in the DataFrame
        def standardize_name(val):
            try:
                # Try to parse as integer (e.g. 1 -> SBK-001)
                num = int(val)
                return f"S-{num:03d}"
            except:
                # If it's already a string, try to extract number or keep as is
                s = str(val).upper().strip()
                if s.startswith("S-") or s.startswith("SBK-"):
                    # Normalize to S-
                    match = re.search(r'\d+', s)
                    if match:
                        return f"S-{int(match.group()):03d}"
                    return s
                # Handle cases like "sbk1" or just "1" as string
                import re
                match = re.search(r'\d+', s)
                if match:
                    num = int(match.group())
                    return f"S-{num:03d}"
                return s

        if id_col in df_geo.columns:
            df_geo[id_col] = df_geo[id_col].apply(standardize_name)
            unique_wells = df_geo.drop_duplicates(subset=id_col, keep="first")
        else:
            unique_wells = pd.DataFrame()

        if not df_sand.empty and id_col in df_sand.columns:
            df_sand[id_col] = df_sand[id_col].apply(standardize_name)
        
        def has_las(well_name):
            # map S-XXX to sbk-xxx for file lookup
            las_base = well_name.lower().replace("s-", "sbk-")
            las_name = f"{las_base}_wire_raw.las"
            return os.path.exists(os.path.join(self.las_dir, las_name))
        
        self.all_markers = df_geo.copy()
        self.sand_markers = df_sand.copy()
        
        if not unique_wells.empty:
            self.wells_info = unique_wells[unique_wells[id_col].apply(has_las)].copy()
        else:
            self.wells_info = pd.DataFrame()
        
        if self.wells_info.empty:
            print(f"Warning: No wells found with associated LAS files in {self.las_dir}")
            self.wells_info = pd.DataFrame(columns=df_geo.columns if not df_geo.empty else [])
            
        return self.wells_info

    def get_well_names(self):
        id_col = "Well identifier (Well name)"
        if id_col in self.wells_info.columns:
            return self.wells_info[id_col].tolist()
        return []

    def load_las_data(self, well_name):
        # well_name comes from the combo box, so it is "S-XXX"
        # map to "sbk-xxx"
        las_base = well_name.lower().replace("s-", "sbk-")
            
        las_name = f"{las_base}_wire_raw.las"
        path = os.path.join(self.las_dir, las_name)
        
        if not os.path.exists(path):
             # Try fallback if naming varies
             path = os.path.join(self.las_dir, f"{well_name}_wire_raw.las")

        las = lasio.read(path)
        df = las.df().reset_index()
        depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
        gr_col = next((c for c in df.columns if 'GR' in c.upper()), None)
        
        if gr_col is None:
            raise ValueError(f"GR column not found in {las_name}")
            
        # Select and Clean Data
        df_clean = df[[depth_col, gr_col]].dropna().copy()
        
        # Unit Conversion Logic: Always standardize to METERS internally
        depth_unit = las.curves[depth_col].unit.upper() if depth_col in las.curves else ""
        
        if 'FT' in depth_unit or 'F' in depth_unit:
            # Confirmed Feet -> Convert to Meters
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
        elif 'M' not in depth_unit:
            # Ambiguous: if depths are large (e.g. > 5000), likely Feet
            if df_clean[depth_col].max() > 5000:
                df_clean[depth_col] = df_clean[depth_col] * 0.3048
            
        return pd.DataFrame({"Depth": df_clean[depth_col].values, "GR": df_clean[gr_col].values})

    def load_vshale_data(self, well_name):
        """
        Loads pre-calculated Vshale data from 'grn vsh' directory.
        Always returns depth in METERS.
        """
        import re
        match = re.search(r'\d+', well_name)
        if not match:
            return pd.DataFrame()
        
        id_str = f"{int(match.group()):02d}"
        vsh_dir = os.path.join(os.path.dirname(self.las_dir), "grn vsh")
        las_name = f"{id_str}_1.las"
        path = os.path.join(vsh_dir, las_name)
        
        if not os.path.exists(path):
            return pd.DataFrame()
            
        las = lasio.read(path)
        df = las.df().reset_index()
        depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
        vsh_col = next((c for c in df.columns if any(x in c.upper() for x in ['VSH', 'VSHALE'])), df.columns[1] if len(df.columns) > 1 else None)
        
        if vsh_col is None:
            return pd.DataFrame()
            
        df_clean = df[[depth_col, vsh_col]].dropna().copy()
        
        # Unit conversion to Meters
        depth_unit = las.curves[depth_col].unit.upper() if depth_col in las.curves else ""
        if 'FT' in depth_unit or 'F' in depth_unit:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
        elif 'M' not in depth_unit and df_clean[depth_col].max() > 5000:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
                
        return pd.DataFrame({"Depth": df_clean[depth_col].values, "Vsh": df_clean[vsh_col].values})

    def load_grn_data(self, well_name):
        """
        Loads GR Normalize data from 'grn vsh' directory.
        Always returns depth in METERS.
        """
        import re
        match = re.search(r'\d+', well_name)
        if not match:
            return pd.DataFrame()
        
        id_str = f"{int(match.group()):02d}"
        vsh_dir = os.path.join(os.path.dirname(self.las_dir), "grn vsh")
        las_name = f"{id_str}.las"
        path = os.path.join(vsh_dir, las_name)
        
        if not os.path.exists(path):
            return pd.DataFrame()
            
        las = lasio.read(path)
        df = las.df().reset_index()
        depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
        gr_col = next((c for c in df.columns if 'GR' in c.upper()), df.columns[1] if len(df.columns) > 1 else None)
        
        if gr_col is None:
            return pd.DataFrame()
            
        df_clean = df[[depth_col, gr_col]].dropna().copy()
        
        # Unit conversion to Meters
        depth_unit = las.curves[depth_col].unit.upper() if depth_col in las.curves else ""
        if 'FT' in depth_unit or 'F' in depth_unit:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
        elif 'M' not in depth_unit and df_clean[depth_col].max() > 5000:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
                
        return pd.DataFrame({"Depth": df_clean[depth_col].values, "Value": df_clean[gr_col].values})

    def get_available_curves(self, well_name):
        las_base = well_name.lower().replace("s-", "sbk-")
        las_name = f"{las_base}_wire_raw.las"
        path = os.path.join(self.las_dir, las_name)
        if not os.path.exists(path):
             path = os.path.join(self.las_dir, f"{well_name}_wire_raw.las")
             
        if not os.path.exists(path):
            return []
            
        try:
            las = lasio.read(path)
            # Return keys except depth
            keys = list(las.keys())
            # rudimentary filter for depth keys
            return [k for k in keys if k.upper() not in ['DEPT', 'DEPTH', 'DEPTH_M']]
        except:
            return []

    def load_curve_data(self, well_name, curve_name):
        """Standardized loading function: Always returns data in METERS."""
        las_base = well_name.lower().replace("s-", "sbk-")
        las_name = f"{las_base}_wire_raw.las"
        path = os.path.join(self.las_dir, las_name)
        if not os.path.exists(path):
             path = os.path.join(self.las_dir, f"{well_name}_wire_raw.las")
             
        las = lasio.read(path)
        df = las.df().reset_index()
        depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
        
        if curve_name not in df.columns:
            # Fallback for case sensitivity
            match = next((c for c in df.columns if c.upper() == curve_name.upper()), None)
            if match:
                curve_name = match
            else:
                raise ValueError(f"Curve {curve_name} not found in {well_name}")

        df_clean = df[[depth_col, curve_name]].dropna().copy()
        
        # Unit Conversion Logic: Always standardize to METERS
        depth_unit = las.curves[depth_col].unit.upper() if depth_col in las.curves else ""
        
        if 'FT' in depth_unit or 'F' in depth_unit:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
        elif 'M' not in depth_unit and df_clean[depth_col].max() > 5000:
            df_clean[depth_col] = df_clean[depth_col] * 0.3048
            
        return pd.DataFrame({"Depth": df_clean[depth_col].values, "Value": df_clean[curve_name].values})

    @staticmethod
    def interpolate_logs(dfA, dfB, val_col_A="GR", val_col_B="GR", depth_col_A="Depth", depth_col_B="Depth", step=None):
        start = max(dfA[depth_col_A].min(), dfB[depth_col_B].min())
        stop = min(dfA[depth_col_A].max(), dfB[depth_col_B].max())
        if stop <= start:
            return None, None, None
            
        # Accuracy Improvement: Auto-detect step if not provided
        if step is None:
            diffs = np.concatenate([np.diff(dfA[depth_col_A]), np.diff(dfB[depth_col_B])])
            step = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0
            # Clip step to reasonable geological resolutions (0.1 to 1.0)
            step = np.clip(step, 0.1, 1.0)
            
        grid = np.arange(start, stop + step, step)
        
        def interp(df, d_col, v_col):
            # Drop NaNs from input before creating interpolator
            valid_df = df.dropna(subset=[d_col, v_col])
            if valid_df.empty:
                return np.full(len(grid), np.nan)
                
            f = interp1d(valid_df[d_col], valid_df[v_col], bounds_error=False, fill_value=np.nan)
            return f(grid)
            
        log_A = interp(dfA, depth_col_A, val_col_A)
        log_B = interp(dfB, depth_col_B, val_col_B)
        
        # Ensure we only keep points where BOTH have valid data
        mask = (~np.isnan(log_A)) & (~np.isnan(log_B))
        return grid[mask], log_A[mask], log_B[mask]

    # Marker retrieval methods moved to EngineFor_GeologyMarker and EngineFor_SandPlot
