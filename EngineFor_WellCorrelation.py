"""
EngineFor_WellCorrelation.py
Combined engine for Well Correlation using INPEFA and DTW
This is a convenience wrapper that combines INPEFAEngine and DTWEngine
"""
from EngineFor_INPEFA import INPEFAEngine
from EngineFor_DTW import DTWEngine


class WellCorrelationEngine:
    """
    Combined engine for well correlation analysis.
    Provides unified interface for INPEFA and DTW operations.
    """
    
    @staticmethod
    def run_inpefa(y_series, x_series, term="long"):
        """
        Runs the INPEFA algorithm on a data series for a specific term.
        
        Parameters:
        -----------
        y_series : pd.Series or np.ndarray
            The log values (e.g., GR values)
        x_series : pd.Series or np.ndarray
            The depth values
        term : str
            INPEFA term to compute: "long", "mid", "short", or "shorter"
            
        Returns:
        --------
        np.ndarray
            INPEFA computed values
        """
        return INPEFAEngine.run_inpefa(y_series, x_series, term)
    
    @staticmethod
    def dtw_with_path(x, y, w):
        """
        DTW implementation with Sakoe-Chiba constraint.
        
        Parameters:
        -----------
        x : np.ndarray
            First time series (e.g., reference well INPEFA)
        y : np.ndarray
            Second time series (e.g., offset well INPEFA)  
        w : int
            Window size for Sakoe-Chiba constraint
            
        Returns:
        --------
        tuple : (cost, pi, pj, D)
            cost : float - Total DTW cost
            pi : np.ndarray - Path indices for x
            pj : np.ndarray - Path indices for y
            D : np.ndarray - Cost matrix
        """
        return DTWEngine.dtw_with_path(x, y, w)
    
    @staticmethod
    def get_recommended_window(x, y, downsample=4):
        """
        Implements the window sweep logic from the notebook.
        Returns the window size in ORIGINAL resolution units.
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        downsample : int
            Downsampling factor for faster computation
            
        Returns:
        --------
        int
            Recommended window size
        """
        return DTWEngine.get_recommended_window(x, y, downsample)

    @staticmethod
    def dtw_sectional(grid, log_A, log_B, ref_boundaries, off_boundaries, downsample=4):
        """
        Asymmetric section-based DTW wrapper. See DTWEngine.dtw_sectional for full docs.
        Pairs ref-well sections (by ref_boundaries) with offset-well sections (by off_boundaries).
        Each section runs DTW+auto-window independently.
        """
        return DTWEngine.dtw_sectional(grid, log_A, log_B, ref_boundaries, off_boundaries, downsample)


# Backward compatibility alias
AnalysisEngine = WellCorrelationEngine
