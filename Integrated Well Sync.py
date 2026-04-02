import sys
import os
import pandas as pd
import numpy as np
import heapq
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QPushButton, 
                             QTabWidget, QGroupBox, QFormLayout, QSlider, 
                             QStatusBar, QFrame, QSplitter, QSpinBox,
                             QListWidget, QListWidgetItem, QAbstractItemView,
                             QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QDoubleSpinBox, QProgressBar, QScrollArea, QCheckBox, 
                             QMessageBox, QTextEdit, QButtonGroup, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QPalette, QFont, QPixmap, QPolygon, QRegion
import pickle
try:
    import psutil
except ImportError:
    psutil = None

# Visualization imports
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from scipy.interpolate import interp1d

# Import backend engines
from EngineFor_DataLoader import DataLoader
from EngineFor_WellCorrelation import AnalysisEngine
from EngineFor_LithologyClustering import LithologyClusteringAnalyzer
from EngineFor_LogPlot import LogPlotEngine
from EngineFor_GeologyMarker import GeologyMarkerEngine
from EngineFor_SandPlot import SandPlotEngine
from EngineFor_RandomForestLithology import RandomForestLithologyEngine
from EngineFor_LogDetector import LogDetectorEngine
from EngineFor_CombineCorrelation import CombineCorrelationEngine

# Define Colors
DARK_BG = "#2d2d2d"      # Dark Grey code
SIDEBAR_BG = "#3c3c3c"   # Lighter grey code
TURQUOISE = "#00ced1"    # Turquoise code
SILVER = "#c0c0c0"       # Silver code
TEXT_COLOR = "#e0e0e0"   # Text Color code
MILK_WHITE = "#fdfbf7"   # Milk White for plots code
PLOT_TEXT = "#000000"    # Black text for white plots code

# --- Mode Selection Dialog ---
class ProjectSelectionDialog(QDialog):
    """
    Initialc dialog to choose between starting a new correlation or opening a recent project.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Integrated Well Sync - Mode Selection")
        self.setFixedSize(600, 450)
        self.selected_mode = None
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; border: 1px solid {TURQUOISE}; }}
            QLabel {{ color: {TURQUOISE}; font-family: 'Segoe UI'; }}
            QPushButton {{
                background-color: {SIDEBAR_BG};
                color: #FFFFFF;
                border: 2px solid #555;
                padding: 15px;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
                border: 2px solid white;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        header = QLabel("Integrated Well Sync")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        self.new_btn = QPushButton("MAKE NEW CORRELATION")
        self.new_btn.setCursor(Qt.PointingHandCursor)
        self.new_btn.clicked.connect(self.choose_new)
        self.new_btn.setStyleSheet("font-size: 20px; padding: 25px;")
        layout.addWidget(self.new_btn)

        self.load_btn = QPushButton("OPEN RECENT PROJECT")
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.clicked.connect(self.choose_load)
        self.load_btn.setStyleSheet("font-size: 20px; padding: 25px;")
        layout.addWidget(self.load_btn)

        self.combine_btn = QPushButton("MULTIPLE WELL CORRELATION (under developing)")
        self.combine_btn.setCursor(Qt.PointingHandCursor)
        self.combine_btn.clicked.connect(self.choose_combine)
        self.combine_btn.setStyleSheet("font-size: 18px; padding: 25px;")
        layout.addWidget(self.combine_btn)

        footer = QLabel("Best Solution for Well Correlation")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-style: italic; font-size: 11px; color: silver;")
        layout.addWidget(footer)

    def choose_new(self):
        self.selected_mode = "new"
        self.accept()

    def choose_load(self):
        self.selected_mode = "load"
        self.accept()

    def choose_combine(self):
        self.selected_mode = "combine"
        self.accept()

# --- MRGC Helper Classes ---
class ClusterDataViewer(QDialog):
    """
    A dedicated dialog to show cluster depth intervals in a high-contrast table.
    """
    def __init__(self, cluster_results, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Clustering Data Summary")
        self.resize(800, 600)
        self.results = cluster_results
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #2E2E2E; color: #FFFAF0; }
            QTableWidget { 
                background-color: #252525; 
                color: #FFFAF0; 
                gridline-color: #444;
                border: 1px solid #484848;
                selection-background-color: #00CED1;
                selection-color: #1A1A1A;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #00CED1;
                padding: 5px;
                border: 1px solid #484848;
                font-weight: bold;
            }
            QPushButton { 
                background-color: #00CED1; 
                color: #1A1A1A; 
                font-weight: bold; 
                border-radius: 4px; 
                padding: 8px;
            }
            QLabel { color: #00CED1; font-weight: bold; font-size: 14px; }
        """)
        layout = QVBoxLayout(self)
        title = QLabel("CLUSTERING DEPTH INTERVALS")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.table = QTableWidget()
        
        # Safer access to unit
        unit_str = "m"
        if hasattr(self.main_window, 'unit_combo'):
            unit_str = "m" if self.main_window.unit_combo.currentText() == "Meters" else "ft"
            
        self.table.setColumnCount(4)  # CRITICAL FIX: Explicitly set columns
        self.table.setHorizontalHeaderLabels(["Well Name", f"Top ({unit_str})", f"Bottom ({unit_str})", "Cluster ID"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.populate_data()
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
    def populate_data(self):
        all_intervals = []
        for well_name, data in self.results.items():
            # Ensure data is sorted by depth for accurate interval calculation
            df_sorted = data.sort_values('Depth')
            depths = df_sorted['Depth'].values
            clusters = df_sorted['Cluster'].values
            
            if len(depths) == 0: continue
            
            current_cluster = clusters[0]
            start_depth = depths[0]
            
            for i in range(1, len(clusters)):
                if clusters[i] != current_cluster:
                    # End of previous interval
                    all_intervals.append([well_name, start_depth, depths[i-1], current_cluster])
                    current_cluster = clusters[i]
                    start_depth = depths[i]
            
            # Add final interval
            all_intervals.append([well_name, start_depth, depths[-1], current_cluster])
            
        self.table.setRowCount(len(all_intervals))
        for row, (well, top, bot, cid) in enumerate(all_intervals):
            # Create items, set alignment, then add to table (Safer)
            item_well = QTableWidgetItem(str(well))
            item_well.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item_well)
            
            item_top = QTableWidgetItem(f"{top:.2f}")
            item_top.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, item_top)
            
            item_bot = QTableWidgetItem(f"{bot:.2f}")
            item_bot.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, item_bot)
            
            item_cid = QTableWidgetItem(f"Cluster {cid}")
            item_cid.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, item_cid)

class MRGCClusteringThread(QThread):
    finished = pyqtSignal(object) # Now returns a dict of results
    error = pyqtSignal(str)
    def __init__(self, analyzer, well_names, params):
        super().__init__()
        self.analyzer = analyzer
        self.well_names = well_names # list of wells
        self.params = params
    def run(self):
        try:
            from EngineFor_LithologyClustering import LithologyClusteringAnalyzer
            local_analyzer = LithologyClusteringAnalyzer(self.analyzer.las_directory)
            local_analyzer.use_gpu = self.analyzer.use_gpu # Sync GPU setting
            results = local_analyzer.cluster_mrgc_multi_well(
                self.well_names,
                alpha=self.params['alpha'],
                n_clusters_manual=self.params['n_clusters'],
                feature_curves=self.params['feature_curves']
            )
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"Combined Clustering: {str(e)}\n{traceback.format_exc()}")

class CorrelationThread(QThread):
    """Background thread for INPEFA and DTW calculations to keep UI responsive."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ref_name, off_name, term_text, boundary_mode, data_loader, engine, depth_multiplier):
        super().__init__()
        self.ref_name = ref_name
        self.off_name = off_name
        self.term_text = term_text
        self.boundary_mode = boundary_mode
        self.data_loader = data_loader
        self.engine = engine
        self.depth_multiplier = depth_multiplier

    def run(self):
        try:
            # 1. Load LAS Data
            df_ref_las = self.data_loader.load_las_data(self.ref_name)
            df_off_las = self.data_loader.load_las_data(self.off_name)
            
            # 2. Run INPEFA
            inpefa_ref = self.engine.run_inpefa(df_ref_las["GR"], df_ref_las["Depth"], term=self.term_text)
            inpefa_off = self.engine.run_inpefa(df_off_las["GR"], df_off_las["Depth"], term=self.term_text)
            
            df_ref = pd.DataFrame({"Depth": df_ref_las["Depth"], "VAL": inpefa_ref})
            df_off = pd.DataFrame({"Depth": df_off_las["Depth"], "VAL": inpefa_off})
            
            # 3. Interpolate and Overlap
            from EngineFor_DataLoader import DataLoader
            grid, log_A, log_B = DataLoader.interpolate_logs(df_ref, df_off, "VAL", "VAL")
            
            if grid is None or len(grid) == 0:
                self.error.emit("Error: No overlapping interval or valid data found.")
                return

            # Step and Windowing
            step_val = grid[1] - grid[0] if len(grid) > 1 else 1.0
            rec_indices = self.engine.get_recommended_window(log_A, log_B, downsample=4)
            rec_dist = rec_indices * step_val

            # 4. Build section boundaries from CHOSEN marker types
            try:
                from EngineFor_GeologyMarker import GeologyMarkerEngine
                
                markers_ref = pd.DataFrame()
                markers_off = pd.DataFrame()

                if self.boundary_mode == "sand":
                    markers_ref = GeologyMarkerEngine.get_markers(self.ref_name, self.data_loader.sand_markers)
                    markers_off = GeologyMarkerEngine.get_markers(self.off_name, self.data_loader.sand_markers)
                elif self.boundary_mode == "geo":
                    markers_ref = GeologyMarkerEngine.get_markers(self.ref_name, self.data_loader.all_markers)
                    markers_off = GeologyMarkerEngine.get_markers(self.off_name, self.data_loader.all_markers)
                # "full" mode leaves markers_ref empty, skipping sectioning

                g_min, g_max = float(grid[0]), float(grid[-1])

                ref_boundaries = []
                off_boundaries = []

                if (not markers_ref.empty and 'Surface' in markers_ref.columns and 'MD' in markers_ref.columns and
                        not markers_off.empty and 'Surface' in markers_off.columns and 'MD' in markers_off.columns):

                    # Build lookup: marker_name → depth (in meters) for each well
                    ref_lookup = dict(zip(markers_ref['Surface'].astype(str),
                                         markers_ref['MD'].astype(float)))
                    off_lookup = dict(zip(markers_off['Surface'].astype(str),
                                         markers_off['MD'].astype(float)))

                    # Find common marker names
                    common_names = set(ref_lookup.keys()) & set(off_lookup.keys())

                    # Build paired boundary lists, sorted by ref depth
                    paired = sorted(
                        [(ref_lookup[name], off_lookup[name])
                         for name in common_names],
                        key=lambda t: t[0]   # sort by ref depth
                    )

                    # Keep only pairs where BOTH depths fall inside the overlap range
                    for r_depth, o_depth in paired:
                        if g_min < r_depth < g_max and g_min < o_depth < g_max:
                            ref_boundaries.append(r_depth)
                            off_boundaries.append(o_depth)

            except Exception:
                # Marker loading failed → standard full-range DTW
                ref_boundaries = []
                off_boundaries = []

            # 5. Asymmetric section-based DTW
            # ref section [A→B] is paired with off section [A'→B'] where A/A' and B/B'
            # are the SAME geological marker at each well's own depth.
            cost, pi, pj = self.engine.dtw_sectional(
                grid, log_A, log_B, ref_boundaries, off_boundaries, downsample=4
            )
            
            if np.isinf(cost):
                self.error.emit("Error: No correlation path found. Try increasing the window size.")
                return
                
            # Pack results — identical format as before, downstream is unchanged
            result = {
                'cost': cost,
                'pi': pi,
                'pj': pj,
                'grid': grid,
                'log_A': log_A,
                'log_B': log_B,
                'rec_dist': rec_dist,
                'ref_name': self.ref_name,
                'off_name': self.off_name
            }
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Correlation Error: {str(e)}\n{traceback.format_exc()}")


class RFTrainingThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, engine, params):
        super().__init__()
        self.engine = engine
        self.params = params
        
    def run(self):
        try:
            # Sync GPU setting
            self.engine.use_gpu = self.params.get('use_gpu', False)
            
            # 1. Preprocess
            X_train, X_test, y_train, y_test, features, df = self.engine.preprocess_data(
                self.params['wells'], 
                feature_curves=self.params['features'],
                vsh_cutoff=self.params['vsh_cutoff']
            )
            
            # 2. Train RF
            rf_metrics = self.engine.train_random_forest(
                X_train, y_train, X_test, y_test,
                n_estimators=self.params['n_estimators'],
                optimize=self.params['optimize']
            )
            
            # 3. Train SVM for comparison
            svm_metrics = self.engine.train_svm_baseline(X_train, y_train, X_test, y_test)
            
            # 4. Predict
            results = {}
            for well in self.params['wells']:
                results[well] = self.engine.predict_lithology(well, self.params['features'])
                
            # Pack all metrics
            output = {
                'rf_metrics': rf_metrics,
                'svm_metrics': svm_metrics,
                'features': features,
                'results': results,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            self.finished.emit(output)
            
        except Exception as e:
            import traceback
            self.error.emit(f"RF Training Error: {str(e)}\n{traceback.format_exc()}")

class ResourceMonitorThread(QThread):
    """Background thread to monitor CPU and GPU RAM usage in real-time."""
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._cpu_name = self._get_cpu_name()
        self._sys_model = self._get_system_model()
        # Prime psutil to get first non-zero reading later
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        
    def _get_system_model(self):
        try:
            import subprocess
            # Use modern CimInstance instead of wmic (which is often missing)
            cmd = 'powershell -Command "Get-CimInstance -ClassName Win32_ComputerSystemProduct | Select-Object -ExpandProperty Name"'
            model = subprocess.check_output(cmd, shell=True).decode().strip()
            # Clean up long names (e.g. ROG Strix G614JU_G614JU -> ROG Strix G614JU)
            if "_" in model:
                model = model.split("_")[0]
            return model if model else "Generic PC"
        except:
            return "PC Device"

    def _get_cpu_name(self):
        try:
            import subprocess
            import platform
            if platform.system() == "Windows":
                # Get clean CPU name via PowerShell (wmic is deprecated)
                cmd = 'powershell -Command "(Get-CimInstance Win32_Processor).Name"'
                name = subprocess.check_output(cmd, shell=True).decode().strip()
                return name if name else platform.processor()
            return platform.processor()
        except:
            return "Generic CPU"

    def run(self):
        import time
        from EngineFor_LithologyClustering import HAS_GPU, cp
        
        gpu_name = "N/A"
        try:
            if HAS_GPU and cp is not None:
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = props['name'].decode()
        except:
            gpu_name = "GPU Device"

        while self._running:
            try:
                self.stats_updated.emit({
                    'sys_model': self._sys_model,
                    'cpu_name': self._cpu_name,
                    'gpu_active': HAS_GPU,
                    'gpu_name': gpu_name
                })
            except:
                pass
            time.sleep(5.0) # identification only: poll every 5s
            
    def stop(self):
        self._running = False

# --- Annotation Manager ---
class DrawingManager:
    """
    Manages free-hand drawing on Matplotlib canvases.
    Stores lines per axis label to persist across clears.
    """
    def __init__(self, canvas, on_change=None):
        self.canvas = canvas
        self.on_change = on_change
        self.drawings = {}  # key -> list of line-segments
        self.history = []   # list of actions for undo: {"type": "draw"/"erase", "key": key, "line": data, "index": idx}
        self.redo_stack = [] 
        self.current_key = None
        self.active = False
        self.mode = "draw" # "draw" or "erase"
        self.color = TURQUOISE  # Match application theme
        
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_press(self, event):
        if not self.active or event.inaxes is None or event.button != 1:
            return
        key = event.inaxes.get_label()
        if not key or key == "": return
        self.current_key = key
        
        if self.mode == "draw":
            if key not in self.drawings:
                self.drawings[key] = []
            new_line = [(event.xdata, event.ydata)]
            self.drawings[key].append(new_line)
            # Track history: what we added
            self.history.append({"type": "draw", "key": key, "line": new_line, "index": len(self.drawings[key])-1})
            self.redo_stack.clear()
        else: # Erase mode
            self.erase_at(event)

    def on_move(self, event):
        if not self.active or event.inaxes is None:
            return
        
        if self.mode == "draw":
            if self.current_key is None or event.inaxes.get_label() != self.current_key:
                return
            self.drawings[self.current_key][-1].append((event.xdata, event.ydata))
            # Immediate feedback (fast)
            line_data = self.drawings[self.current_key][-1]
            if len(line_data) > 1:
                xs, ys = zip(*line_data[-2:])
                event.inaxes.plot(xs, ys, color=self.color, lw=2.0, alpha=0.9, zorder=100)
                self.canvas.draw_idle()
        elif self.mode == "erase" and event.button == 1:
            self.erase_at(event)

    def erase_at(self, event):
        ax = event.inaxes
        if ax is None: return
        key = ax.get_label()
        if key not in self.drawings: return
        
        ex, ey = event.xdata, event.ydata
        if ex is None or ey is None: return
        
        # Robust threshold: 3% of the visible axis range
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        thresh_x = 0.03 * abs(xlim[1] - xlim[0])
        thresh_y = 0.03 * abs(ylim[1] - ylim[0])
        
        to_remove = []
        for i, line in enumerate(self.drawings[key]):
            for p in line:
                # Simple distance check for points (fast)
                if abs(p[0]-ex) < thresh_x and abs(p[1]-ey) < thresh_y:
                    to_remove.append(i)
                    break
        
        if to_remove:
            # Sort descending to pop without shifting indices
            for i in sorted(list(set(to_remove)), reverse=True):
                removed_line = self.drawings[key].pop(i)
                self.history.append({"type": "erase", "key": key, "line": removed_line, "index": i})
            
            self.redo_stack.clear()
            if self.on_change: self.on_change() # Trigger full redraw

    def on_release(self, event):
        was_drawing = (self.mode == "draw" and self.current_key is not None)
        self.current_key = None
        if was_drawing and self.on_change:
            self.on_change() # Commit draw and clean up feedback lines

    def undo(self):
        if not self.history:
            return False
        action = self.history.pop()
        self.redo_stack.append(action)
        
        key, line, idx = action["key"], action["line"], action["index"]
        if action["type"] == "draw":
            if key in self.drawings and idx < len(self.drawings[key]):
                self.drawings[key].pop(idx)
        else: # erase
            if key not in self.drawings: self.drawings[key] = []
            self.drawings[key].insert(idx, line)
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        action = self.redo_stack.pop()
        self.history.append(action)
        
        key, line, idx = action["key"], action["line"], action["index"]
        if action["type"] == "draw":
            if key not in self.drawings: self.drawings[key] = []
            self.drawings[key].insert(idx, line)
        else: # erase
            if key in self.drawings and idx < len(self.drawings[key]):
                self.drawings[key].pop(idx)
        return True

    def redraw(self, axes_list):
        for ax in axes_list:
            key = ax.get_label()
            if key in self.drawings:
                for line_data in self.drawings[key]:
                    if len(line_data) > 1:
                        xs, ys = zip(*line_data)
                        ax.plot(xs, ys, color=self.color, lw=2.0, alpha=0.9, zorder=100)

    def clear(self):
        self.drawings = {}
        self.history = []
        self.redo_stack = []
        self.canvas.draw_idle()

class VerticalToolbar(NavigationToolbar):
    """
    Custom toolbar that restricts pan and zoom to the vertical (Depth) axis.
    """
    def drag_pan(self, event):
        # Override to prevent horizontal panning
        for ax in self.canvas.figure.get_axes():
            ax.set_autoscalex_on(False)
            xlim = ax.get_xlim()
            super().drag_pan(event)
            ax.set_xlim(xlim)

    def drag_zoom(self, event):
        # Override to prevent horizontal zooming
        for ax in self.canvas.figure.get_axes():
            ax.set_autoscalex_on(False)
            xlim = ax.get_xlim()
            super().drag_zoom(event)
            ax.set_xlim(xlim)

class WellCorrelationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Well Sync")
        self.resize(1200, 900)
        
        # Color Map for well logs - Professional colors for tracks
        self.well_colors = {"Reference": "#000080", "Offset": "#8b0000"} # Navy Blue & Dark Red
        
        # Paths
        self.marker_path = r"path/to/your/marker_file.xlsx"
        self.las_dir = r"path/to/your/las_directory"
        
        # Initialize backend
        self.data_loader = DataLoader(self.marker_path, self.las_dir)
        self.mrgc_analyzer = LithologyClusteringAnalyzer(self.las_dir)
        self.log_plot_engine = LogPlotEngine()
        self.geology_engine = GeologyMarkerEngine()
        self.sand_engine = SandPlotEngine()
        self.log_detector_engine = LogDetectorEngine(self.data_loader)
        self.combine_engine = CombineCorrelationEngine(self)
        self.mrgc_results = {}
        self.rf_results = {} # NEW: Store Random Forest results per well
        self.mrgc_axes_to_well = {}
        self.is_combine_mode = False
        self.las_cache = {}
        self.plot_theme = "Dark Mode" # Default theme (consistent with combo)
        self.show_geo_corr = True # Geology Marker connection flag
        self.depth_multiplier = 1.0 # 1.0 for Meters, 3.28084 for Feet
        self.depth_unit = "m"
        self.use_gpu = False # Default to CPU for stability
        self.is_combine_mode = False
        self.project_chain = []
        
        self.apply_dark_theme()
        self.init_ui()
        self.setup_menu()
        
        # Start Resource Monitor
        self.res_monitor = ResourceMonitorThread()
        self.res_monitor.stats_updated.connect(self.update_resource_stats)
        self.res_monitor.start()
        
    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(DARK_BG))
        palette.setColor(QPalette.WindowText, QColor(SILVER))
        palette.setColor(QPalette.Base, QColor(SIDEBAR_BG))
        palette.setColor(QPalette.AlternateBase, QColor(DARK_BG))
        palette.setColor(QPalette.ToolTipBase, QColor(SILVER))
        palette.setColor(QPalette.ToolTipText, QColor(DARK_BG))
        palette.setColor(QPalette.Text, QColor(SILVER))
        palette.setColor(QPalette.Button, QColor(SIDEBAR_BG))
        palette.setColor(QPalette.ButtonText, QColor(SILVER))
        palette.setColor(QPalette.Highlight, QColor(TURQUOISE))
        palette.setColor(QPalette.HighlightedText, QColor(Qt.black))
        self.setPalette(palette)
        
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {DARK_BG}; }}
            QGroupBox {{ 
                font-weight: bold; border: 1px solid {SILVER}; 
                margin-top: 10px; color: {TURQUOISE}; 
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
            QLabel {{ color: {SILVER}; }}
            QComboBox {{ 
                background-color: {SIDEBAR_BG}; border: 1px solid {SILVER}; 
                padding: 5px; color: {SILVER};
            }}
            QComboBox QAbstractItemView {{
                background-color: {SIDEBAR_BG}; color: {SILVER}; selection-background-color: {TURQUOISE};
            }}
            QPushButton {{ 
                background-color: {TURQUOISE}; color: {DARK_BG}; font-weight: bold; 
                padding: 8px; border-radius: 4px; border: 1px solid {TURQUOISE};
            }}
            QPushButton:hover {{ background-color: #008b8b; }}
            QTabWidget::pane {{ border: 1px solid {SILVER}; }}
            QTabBar::tab {{ 
                background: {SIDEBAR_BG}; color: {SILVER}; padding: 10px; border: 1px solid #555;
            }}
            QTabBar::tab:selected {{ background: {TURQUOISE}; color: {DARK_BG}; font-weight: bold; }}
            QStatusBar {{ background: {SIDEBAR_BG}; color: {TURQUOISE}; }}
        """)

    def init_ui(self):
        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # We use a Vertical Layout to have a distinct Header + Content area
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Custom Status Bar (placed at the top)
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {SIDEBAR_BG};
                color: {TURQUOISE};
                border-bottom: 2px solid {TURQUOISE};
                font-family: 'Segoe UI', sans-serif;
                font-weight: 500;
                min-height: 20px;
                font-size: 11px;
            }}
            QStatusBar::item {{ border: none; }}
        """)
        self.status_bar.showMessage("Initializing system components...")
        
        # --- Header Area (Top-Right Logo) ---
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet(f"background-color: {DARK_BG};")
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(10, 2, 15, 2)
        
        # Elegant 2D Header Title & Subtitle Wrapper
        title_container = QWidget()
        title_vlayout = QVBoxLayout(title_container)
        title_vlayout.setContentsMargins(0, 0, 0, 0)
        title_vlayout.setSpacing(2) # Tight spacing between title and subtitle
        
        logo_label = QLabel()
        logo_label.setText("Integrated Well Sync")
        logo_label.setStyleSheet(f"""
            color: {MILK_WHITE}; 
            font-size: 22px; 
            font-family: 'Segoe UI', 'Garamond', 'serif';
            font-weight: bold;
            letter-spacing: 1px;
        """)
        logo_label.setAlignment(Qt.AlignRight)
        
        subtitle_label = QLabel("Best Solution for Well Correlation")
        subtitle_label.setStyleSheet(f"""
            color: {TURQUOISE}; 
            font-size: 11px; 
            font-family: 'Segoe UI', 'Arial';
            font-style: italic;
            font-weight: 500;
        """)
        subtitle_label.setAlignment(Qt.AlignRight)
        
        title_vlayout.addWidget(logo_label)
        title_vlayout.addWidget(subtitle_label)
        
        # --- Stylized Exit Button in Header ---
        self.exit_btn = QPushButton("⏻")
        self.exit_btn.setFixedSize(30, 30)
        self.exit_btn.setCursor(Qt.PointingHandCursor)
        self.exit_btn.setToolTip("Exit Application")
        
        # Apply Pentagon Shape
        pentagon = QPolygon([
            QPoint(15, 0),   # Top
            QPoint(30, 11),  # Mid Right
            QPoint(24, 30),  # Bottom Right
            QPoint(6, 30),   # Bottom Left
            QPoint(0, 11)    # Mid Left
        ])
        self.exit_btn.setMask(QRegion(pentagon))

        self.exit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SIDEBAR_BG};
                color: {SILVER};
                border: 1px solid #555;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
                border: 2px solid white;
            }}
        """)
        self.exit_btn.clicked.connect(self.close)

        header_layout.addStretch() # Push title to the right
        header_layout.addWidget(title_container)
        header_layout.addSpacing(15)
        header_layout.addWidget(self.exit_btn)
        
        main_layout.addWidget(self.header_frame)
        main_layout.addWidget(self.status_bar)
        
        # --- Content Area ---
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Use Splitter for resizable Sidebar/Content
        splitter = QSplitter(Qt.Horizontal)
        
        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(2)
        
        # Well Selection Group
        well_group = QGroupBox("Well Selection")
        well_form = QFormLayout()
        
        self.ref_well_combo = QComboBox()
        self.off_well_combo = QComboBox()
        wells = [str(w) for w in self.data_loader.get_well_names()]
        self.ref_well_combo.addItems(wells)
        self.off_well_combo.addItems(wells)
        
        # Connect Selection to Map Refresh
        self.ref_well_combo.currentIndexChanged.connect(self.plot_2d_map)
        self.ref_well_combo.currentIndexChanged.connect(self.update_available_logs)
        self.ref_well_combo.currentIndexChanged.connect(self.update_mrgc_features)
        self.ref_well_combo.currentIndexChanged.connect(lambda: self.plot_geology_markers() if self.tabs.currentIndex() == 3 else None)
        self.ref_well_combo.currentIndexChanged.connect(lambda: self.plot_sand_markers() if self.tabs.currentIndex() == 4 else None)
        self.ref_well_combo.currentIndexChanged.connect(self.sync_rf_wells) # NEW: Sync RF well
        
        self.off_well_combo.currentIndexChanged.connect(self.plot_2d_map)
        self.off_well_combo.currentIndexChanged.connect(self.update_available_logs)
        self.off_well_combo.currentIndexChanged.connect(self.update_mrgc_features)
        self.off_well_combo.currentIndexChanged.connect(lambda: self.plot_geology_markers() if self.tabs.currentIndex() == 3 else None)
        self.off_well_combo.currentIndexChanged.connect(lambda: self.plot_sand_markers() if self.tabs.currentIndex() == 4 else None)
        self.off_well_combo.currentIndexChanged.connect(self.sync_rf_wells) # NEW: Sync RF well
        
        well_form.addRow("Reference Well:", self.ref_well_combo)
        well_form.addRow("Offset Well:", self.off_well_combo)
        well_group.setLayout(well_form)
        
        # Parameters Group
        param_group = QGroupBox("Correlation Parameters")
        param_form = QFormLayout()
        
        self.inpefa_combo = QComboBox()
        self.inpefa_combo.addItems(["Long Term", "Mid Term", "Short Term", "Shorter Term"])
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Meters", "Feet"])
        self.unit_combo.currentIndexChanged.connect(self.update_unit_ui)
        
        # Radio Button Group for DTW Boundaries
        self.dtw_group = QButtonGroup(self)
        
        self.rb_sand_dtw = QRadioButton("Top Sand Markers")
        self.rb_geo_dtw = QRadioButton("Sequence Markers")
        self.rb_full_dtw = QRadioButton("Full Overlapping Interval")
        
        self.rb_sand_dtw.setChecked(True)
        self.rb_sand_dtw.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
        self.rb_geo_dtw.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
        self.rb_full_dtw.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
        
        self.dtw_group.addButton(self.rb_sand_dtw)
        self.dtw_group.addButton(self.rb_geo_dtw)
        self.dtw_group.addButton(self.rb_full_dtw)
        
        # Connect toggled signal to auto-run (only when checked)
        self.rb_sand_dtw.toggled.connect(lambda c: self.run_analysis(switch_tab=False) if c else None)
        self.rb_geo_dtw.toggled.connect(lambda c: self.run_analysis(switch_tab=False) if c else None)
        self.rb_full_dtw.toggled.connect(lambda c: self.run_analysis(switch_tab=False) if c else None)
        
        param_form.addRow("INPEFA Term:", self.inpefa_combo)
        param_form.addRow("Depth Unit:", self.unit_combo)
        param_form.addRow("DTW Boundary:", self.rb_sand_dtw)
        param_form.addRow("", self.rb_geo_dtw)
        param_form.addRow("", self.rb_full_dtw)
        param_group.setLayout(param_form)
        
        # Display Settings Group
        disp_group = QGroupBox("Display Settings")
        disp_layout = QFormLayout()
        
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(10, 90)
        self.ratio_slider.setValue(50)
        self.ratio_slider.valueChanged.connect(self.update_plot_layout)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentIndexChanged.connect(self.toggle_plot_theme)
        
        disp_layout.addRow("Panel Width:", self.ratio_slider)
        disp_layout.addRow("Plot Theme:", self.theme_combo)
        disp_group.setLayout(disp_layout)
        
        # Computation Settings
        compute_group = QGroupBox("Compute Engine")
        compute_layout = QVBoxLayout()
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
        self.gpu_check.clicked.connect(self.toggle_gpu_acceleration)
        
        from EngineFor_LithologyClustering import HAS_GPU
        from EngineFor_RandomForestLithology import HAS_XGB
        can_use_gpu = HAS_GPU or HAS_XGB
        
        self.gpu_status_lbl = QLabel(f"Hardware Support: {'CUDA Detected' if can_use_gpu else 'CPU Only'}")
        self.gpu_status_lbl.setStyleSheet(f"font-size: 9px; color: {SILVER if can_use_gpu else '#ff6666'};")
        
        self.gpu_check.setEnabled(can_use_gpu)
        if not can_use_gpu:
            self.gpu_check.setToolTip("GPU libraries (cupy/xgboost) not found. Falling back to optimized CPU.")
        
        compute_layout.addWidget(self.gpu_check)
        compute_layout.addWidget(self.gpu_status_lbl)
        compute_group.setLayout(compute_layout)
        
        # Annotation Group
        anno_group = QGroupBox("Sketch & Annotate")
        anno_layout = QVBoxLayout()
        anno_layout.setSpacing(5)
        anno_layout.setContentsMargins(10, 15, 10, 10)
        
        # Style for annotation buttons - making them more subtle
        anno_btn_style = f"""
            QPushButton {{
                background-color: {SIDEBAR_BG};
                color: {SILVER};
                border: 1px solid #555;
                font-weight: normal;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: #444;
                border: 1px solid {TURQUOISE};
            }}
            QPushButton:checked {{
                background-color: #444;
                border: 1px solid {TURQUOISE};
                color: {TURQUOISE};
                font-weight: bold;
            }}
        """
        
        self.sketch_btn = QPushButton("✎ Sketch") # Changed to greyish pencil
        self.sketch_btn.setCheckable(True)
        self.sketch_btn.setStyleSheet(anno_btn_style)
        self.sketch_btn.clicked.connect(self.toggle_sketching)
        
        self.erase_btn = QPushButton("⌫ Erase") # Subtle erase icon
        self.erase_btn.setCheckable(True)
        self.erase_btn.setStyleSheet(anno_btn_style)
        self.erase_btn.clicked.connect(self.toggle_eraser)
        
        # Row for Sketch/Erase
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(self.sketch_btn)
        tool_layout.addWidget(self.erase_btn)
        
        # Undo/Redo - smaller
        undo_redo_layout = QHBoxLayout()
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setToolTip("Undo")
        self.undo_btn.setStyleSheet(anno_btn_style)
        self.undo_btn.clicked.connect(self.undo_last_sketch)
        
        self.redo_btn = QPushButton("↷")
        self.redo_btn.setToolTip("Redo")
        self.redo_btn.setStyleSheet(anno_btn_style)
        self.redo_btn.clicked.connect(self.redo_last_sketch)
        
        self.clear_sketch_btn = QPushButton("🗑 Clear")
        self.clear_sketch_btn.setStyleSheet(anno_btn_style)
        self.clear_sketch_btn.clicked.connect(self.clear_all_sketches)
        
        undo_redo_layout.addWidget(self.undo_btn)
        undo_redo_layout.addWidget(self.redo_btn)
        undo_redo_layout.addWidget(self.clear_sketch_btn)
        
        anno_layout.addLayout(tool_layout)
        anno_layout.addLayout(undo_redo_layout)
        anno_group.setLayout(anno_layout)
        
        # Hardware Monitor Group
        hw_group = QGroupBox("Hardware Monitor")
        hw_layout = QVBoxLayout()
        
        # Laptop Branding
        self.sys_model_lbl = QLabel("ROG Strix Series") # Default placeholder
        self.sys_model_lbl.setAlignment(Qt.AlignCenter)
        self.sys_model_lbl.setStyleSheet(f"color: {TURQUOISE}; font-size: 13px; font-weight: bold; margin-bottom: 5px;")
        hw_layout.addWidget(self.sys_model_lbl)
        
        dev_layout = QHBoxLayout()
        self.dev_status_icon = QLabel("⚙") # Icon
        self.dev_status_icon.setStyleSheet(f"color: {SILVER}; font-size: 14px;")
        self.dev_status_lbl = QLabel("Mode: CPU")
        self.dev_status_lbl.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
        dev_layout.addWidget(self.dev_status_icon)
        dev_layout.addWidget(self.dev_status_lbl)
        dev_layout.addStretch()
        
        # Hardware identifying info only (Stability mode)
        hw_layout.addLayout(dev_layout)
        self.cpu_name_lbl = QLabel("Processor: Detecting...")
        self.cpu_name_lbl.setStyleSheet(f"color: {SILVER}; font-size: 10px;")
        self.cpu_name_lbl.setWordWrap(True)
        hw_layout.addWidget(self.cpu_name_lbl)
        
        # Spacer for visual separation
        hw_layout.addSpacing(2)
        
        self.gpu_name_lbl = QLabel("Graphics: Detecting...")
        self.gpu_name_lbl.setStyleSheet(f"color: {SILVER}; font-size: 10px;")
        self.gpu_name_lbl.setWordWrap(True)
        hw_layout.addWidget(self.gpu_name_lbl)
        
        hw_layout.addSpacing(2)
        hw_group.setLayout(hw_layout)
        
        # Buttons
        self.run_btn = QPushButton("Calculate Correlation")
        # Style defined in stylesheet, but specific override here if needed
        self.run_btn.clicked.connect(lambda: self.run_analysis(switch_tab=True))
        
        self.export_btn = QPushButton("Export Results (CSV)")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_csv)
        
        sidebar_layout.addWidget(well_group)

        # --- 2D Maps Mode Selection ---
        self.map_mode_group = QGroupBox("2D Maps Mode")
        map_mode_layout = QVBoxLayout()
        map_mode_layout.setSpacing(1)
        map_mode_layout.setContentsMargins(5, 10, 5, 5)
        
        self.rb_chosen_map = QRadioButton("Chosen Well Maps")
        self.rb_route_map = QRadioButton("Route Plotter Maps")
        self.rb_chosen_map.setChecked(True)
        
        rb_style = f"color: {TURQUOISE}; font-weight: bold;"
        self.rb_chosen_map.setStyleSheet(rb_style)
        self.rb_route_map.setStyleSheet(rb_style)
        
        self.map_mode_btn_group = QButtonGroup(self)
        self.map_mode_btn_group.addButton(self.rb_chosen_map)
        self.map_mode_btn_group.addButton(self.rb_route_map)
        
        # Multi-select list for Route Plotter
        self.route_wells_list = QListWidget()
        self.route_wells_list.setSelectionMode(QAbstractItemView.MultiSelection)
        wells_list = self.data_loader.get_well_names()
        self.route_wells_list.addItems(wells_list)
        self.route_wells_list.setFixedHeight(120)
        self.route_wells_list.setVisible(False)
        self.route_wells_list.setStyleSheet(f"""
            QListWidget {{ 
                background-color: {SIDEBAR_BG}; 
                color: {SILVER}; 
                border: 1px solid #555;
                font-size: 11px;
            }}
            QListWidget::item:selected {{ background-color: {TURQUOISE}; color: {DARK_BG}; }}
        """)
        
        map_mode_layout.addWidget(self.rb_chosen_map)
        map_mode_layout.addWidget(self.rb_route_map)
        map_mode_layout.addWidget(self.route_wells_list)
        self.map_mode_group.setLayout(map_mode_layout)
        
        sidebar_layout.addWidget(self.map_mode_group)

        # Connections for Mode Switching
        self.rb_chosen_map.toggled.connect(self.on_map_mode_changed)
        self.rb_route_map.toggled.connect(self.on_map_mode_changed)
        self.route_wells_list.itemSelectionChanged.connect(self.plot_2d_map)

        # Switch Mode Button
        self.sw_mode_btn = QPushButton("Switch Mode")
        self.sw_mode_btn.setCursor(Qt.PointingHandCursor)
        self.sw_mode_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SIDEBAR_BG};
                color: {TURQUOISE};
                border: 1px solid {TURQUOISE};
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                margin-top: 5px;
            }}
            QPushButton:hover {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
            }}
        """)
        self.sw_mode_btn.clicked.connect(self.relaunch_app)

        sidebar_layout.addWidget(param_group)
        sidebar_layout.addWidget(disp_group)
        sidebar_layout.addWidget(compute_group)
        sidebar_layout.addWidget(anno_group)
        sidebar_layout.addWidget(hw_group) # NEW
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.sw_mode_btn)
        sidebar_layout.addWidget(self.run_btn)
        sidebar_layout.addWidget(self.export_btn)
        
        # Central Area (Tabs)
        self.tabs = QTabWidget()
        self.map_tab = QWidget()
        self.corr_tab = QWidget()
        self.log_tab = QWidget()
        
        self.tabs.addTab(self.map_tab, "Well Map (2D)")
        self.tabs.addTab(self.corr_tab, "Automatic Correlation")
        self.tabs.addTab(self.log_tab, "Log Viewer")
        self.geology_tab = QWidget()
        self.tabs.addTab(self.geology_tab, "Sequence Marker")
        self.sand_tab = QWidget()
        self.tabs.addTab(self.sand_tab, "Top Sand Marker")
        self.mrgc_tab = QWidget()
        self.tabs.addTab(self.mrgc_tab, "MRGC Clustering")
        self.rf_tab = QWidget()
        self.tabs.addTab(self.rf_tab, "RF Classification")
        self.detector_tab = QWidget()
        self.tabs.addTab(self.detector_tab, "Log Detector")
        self.compiled_tab = QWidget()
        self.tabs.addTab(self.compiled_tab, "Compiled Plot")
        
        self.init_map_tab()
        self.init_corr_tab()
        self.init_log_tab()
        self.init_mrgc_tab()
        self.init_rf_tab()
        self.init_geology_tab()
        self.init_sand_tab()
        self.init_detector_tab()
        self.init_compiled_tab()
        
        # Tabs connection
        self.tabs.currentChanged.connect(self.refresh_active_tab)
        
        # Add widgets to splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1) # Make tabs expandable
        
        main_layout.addWidget(splitter)
        
        self.status_bar.showMessage("Ready. Select wells and click Calculate.")
        
        # Initial logs/features load
        self.update_available_logs()
        self.update_mrgc_features()

    def _get_bar_style(self, color):
        """Helper to return consistent QProgressBar styling."""
        return f"""
            QProgressBar {{ border: 1px solid #444; border-radius: 3px; background: #1a1a1a; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}
        """

    def update_resource_stats(self, stats):
        """Update the sidebar monitor with latest hardware stats (0.5s polling)."""
        # 0. Update System Model (Personalized Branding)
        if hasattr(self, 'sys_model_lbl'):
            self.sys_model_lbl.setText(stats.get('sys_model', "PC Device"))
            
        # 1. Update identifying text
        if 'cpu_name' in stats: self.cpu_name_lbl.setText(f"Proc: {stats['cpu_name']}")
        if 'gpu_name' in stats: self.gpu_name_lbl.setText(f"Graph: {stats['gpu_name']}")
        
        # 2. Update Device Label based on current mode and availability
        can_use_gpu = stats['gpu_active']
        is_gpu_enabled = self.use_gpu # from computation settings
        
        if is_gpu_enabled and can_use_gpu:
            self.dev_status_lbl.setText("Mode: GPU (Active)")
            self.dev_status_lbl.setStyleSheet(f"color: {SILVER}; font-weight: bold;")
            # Use monochrome lightning + Variation Selector-15 to force color styling
            self.dev_status_icon.setText("\u26A1\ufe0e") 
            self.dev_status_icon.setStyleSheet(f"color: {TURQUOISE}; font-size: 16px; font-weight: bold;")
        else:
            self.dev_status_lbl.setText("Mode: CPU (Active)")
            self.dev_status_lbl.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold;")
            self.dev_status_icon.setText("⚙")
            self.dev_status_icon.setStyleSheet(f"color: {SILVER}; font-size: 14px;")

    def closeEvent(self, event):
        """Ensure background threads are stopped gracefully on exit."""
        if hasattr(self, 'res_monitor'):
            self.res_monitor.stop()
            self.res_monitor.wait()
        event.accept()

    def toggle_sketching(self, checked):
        if checked:
            # Turn off eraser without recursion
            self.erase_btn.blockSignals(True)
            self.erase_btn.setChecked(False)
            self.erase_btn.blockSignals(False)
        
        for mgr in [self.draw_mgr_2d, self.draw_mgr_corr, self.draw_mgr_log, self.draw_mgr_geo, self.draw_mgr_sand, self.draw_mgr_mrgc, self.draw_mgr_compiled]:
            mgr.active = checked
            if checked: mgr.mode = "draw"

    def toggle_eraser(self, checked):
        if checked:
            # Turn off sketching without recursion
            self.sketch_btn.blockSignals(True)
            self.sketch_btn.setChecked(False)
            self.sketch_btn.blockSignals(False)
        
        for mgr in [self.draw_mgr_2d, self.draw_mgr_corr, self.draw_mgr_log, self.draw_mgr_geo, self.draw_mgr_sand, self.draw_mgr_mrgc, self.draw_mgr_compiled]:
            mgr.active = checked
            if checked: mgr.mode = "erase"
        
    def clear_all_sketches(self):
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, 'Clear All Sketches', 
                                   "Are you sure you want to delete all sketches on all tabs?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.draw_mgr_2d.clear()
            self.draw_mgr_corr.clear()
            self.draw_mgr_log.clear()
            self.draw_mgr_geo.clear()
            self.draw_mgr_sand.clear()
            self.draw_mgr_mrgc.clear()
            self.draw_mgr_compiled.clear()
            # Refresh currently visible plot
            self.plot_2d_map()
            if hasattr(self, 'last_plot_args'):
                self.plot_correlation(*self.last_plot_args)
            self.plot_logs()
            self.plot_mrgc_results()
            self.plot_geology_markers()
            self.plot_sand_markers()
            self.plot_compiled()
            self.status_bar.showMessage("All sketches cleared.")

    def undo_last_sketch(self):
        # Determine which manager to use based on active tab
        idx = self.tabs.currentIndex()
        mgr = [self.draw_mgr_2d, self.draw_mgr_corr, self.draw_mgr_log, self.draw_mgr_geo, self.draw_mgr_sand, self.draw_mgr_mrgc, self.draw_mgr_compiled][idx]
        if mgr.undo():
            # Refresh the plot
            self.refresh_active_tab()
            self.status_bar.showMessage("Undo action.")

    def redo_last_sketch(self):
        idx = self.tabs.currentIndex()
        mgr = [self.draw_mgr_2d, self.draw_mgr_corr, self.draw_mgr_log, self.draw_mgr_geo, self.draw_mgr_sand, self.draw_mgr_mrgc, self.draw_mgr_compiled][idx]
        if mgr.redo():
             # Refresh the plot
            self.refresh_active_tab()
            self.status_bar.showMessage("Redo action.")

    def get_plot_colors(self):
        """Returns colors for matplotlib plots based on current theme."""
        if self.plot_theme in ("Dark Mode", "Dark"):
            return {
                'bg': '#1a1a1a', 
                'text': MILK_WHITE, 
                'grid': '#444444',
                'spine': '#666666'
            }
        else: # Light Mode
            return {
                'bg': MILK_WHITE,
                'text': '#000000',
                'grid': '#cccccc',
                'spine': '#333333'
            }

    def toggle_plot_theme(self):
        """Handles plot theme switching and refreshes all tabs."""
        if self.is_combine_mode:
            # Toggle between Dark and Light
            self.plot_theme = "Light Mode" if self.plot_theme == "Dark Mode" else "Dark Mode"
        else:
            self.plot_theme = self.theme_combo.currentText()
        
        # Immediate refresh of current tab
        self.refresh_active_tab()
        
        # Ensure floating buttons match theme in Combine Mode
        if self.is_combine_mode:
            c = self.get_plot_colors()
            btn_style = f"""
                QPushButton {{
                    background-color: {SIDEBAR_BG if self.plot_theme == "Dark Mode" else "#f0f0f0"};
                    color: {TURQUOISE};
                    border: 2px solid {TURQUOISE};
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {TURQUOISE};
                    color: white;
                }}
            """
            if hasattr(self, 'back_btn'): self.back_btn.setStyleSheet(btn_style)
            if hasattr(self, 'combine_theme_btn'): self.combine_theme_btn.setStyleSheet(btn_style)

        # Ensure Geology Marker canvas matches theme
        if hasattr(self, 'fig_geo'):
            c = self.get_plot_colors()
            self.fig_geo.patch.set_facecolor(c['bg'])
            self.canvas_geo.setStyleSheet(f"background-color: {c['bg']};")
            self.scroll_geo.setStyleSheet(f"background-color: {c['bg']}; border: none;")
            
        if hasattr(self, 'fig_sand'):
            c = self.get_plot_colors()
            self.fig_sand.patch.set_facecolor(c['bg'])
            self.canvas_sand.setStyleSheet(f"background-color: {c['bg']};")
            self.scroll_sand.setStyleSheet(f"background-color: {c['bg']}; border: none;")
            
        self.status_bar.showMessage(f"Plot theme changed to {self.plot_theme}.")

    def toggle_gpu_acceleration(self, checked):
        self.use_gpu = checked
        # Propagate to engines
        if hasattr(self, 'mrgc_analyzer'):
            self.mrgc_analyzer.use_gpu = checked
        if hasattr(self, 'rf_engine'):
            self.rf_engine.use_gpu = checked
            
        status = "GPU (CUDA)" if checked else "CPU (Optimized)"
        self.status_bar.showMessage(f"Compute engine switched to {status}")
        
    def refresh_active_tab(self):
        idx = self.tabs.currentIndex()
        if idx == 0: self.plot_2d_map()
        elif idx == 1: 
            if hasattr(self, 'last_plot_args'): self.plot_correlation(*self.last_plot_args)
            else: self.draw_empty_tracks()
        elif idx == 2: self.plot_logs()
        elif idx == 3: self.plot_geology_markers()
        elif idx == 4: self.plot_sand_markers()
        elif idx == 5: self.plot_mrgc_results()
        elif idx == 6: self.plot_rf_results()
        elif idx == 7: self.update_detector_results()
        elif idx == 8: 
            if self.is_combine_mode:
                self.plot_combined_correlation()
            else:
                self.plot_compiled()

    def init_map_tab(self):
        layout = QVBoxLayout(self.map_tab)
        c = self.get_plot_colors()
        
        # 2D Map (Matplotlib)
        self.canvas_2d = FigureCanvas(Figure(figsize=(8, 8), facecolor=c['bg']))
        self.ax_2d = self.canvas_2d.figure.add_subplot(111, label="map")
        self.ax_2d.set_facecolor(c['bg'])
        
        # Initialize Manager
        self.draw_mgr_2d = DrawingManager(self.canvas_2d, on_change=self.refresh_active_tab)
        
        # Toolbar
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self.map_tab)
        
        # Styling for Professional Look
        self.ax_2d.tick_params(colors=c['text'], which='both')
        self.ax_2d.xaxis.label.set_color(c['text'])
        self.ax_2d.yaxis.label.set_color(c['text'])
        self.ax_2d.title.set_color(c['text'])
        
        for spine in self.ax_2d.spines.values():
            spine.set_color(c['spine'])
            spine.set_linewidth(1.0)
        
        layout.addWidget(self.toolbar_2d)
        layout.addWidget(self.canvas_2d)
        
        # Initial Plot
        self.plot_2d_map()

    def find_avoidance_path(self, start_coords, end_coords, obstacles):
        """
        A* pathfinding on a grid to find a path that avoids obstacles.
        """
        # 1. Define Grid
        x_all = [start_coords[0], end_coords[0]] + [o[0] for o in obstacles]
        y_all = [start_coords[1], end_coords[1]] + [o[1] for o in obstacles]
        
        margin = (max(x_all) - min(x_all)) * 0.1 or 100
        xmin, xmax = min(x_all) - margin, max(x_all) + margin
        ymin, ymax = min(y_all) - margin, max(y_all) + margin
        
        res = 60 # Grid resolution
        grid_x = np.linspace(xmin, xmax, res)
        grid_y = np.linspace(ymin, ymax, res)
        
        # 2. Convert coordinates to grid indices
        def to_idx(coord):
            ix = np.argmin(np.abs(grid_x - coord[0]))
            iy = np.argmin(np.abs(grid_y - coord[1]))
            return (int(ix), int(iy))
            
        start_node = to_idx(start_coords)
        end_node = to_idx(end_coords)
        
        # 3. Mark Obstacles
        obstacle_indices = set()
        # Safety radius in grid units (approx 2 grid cells)
        radius_sq = 2.5**2
        
        for obs in obstacles:
            ox, oy = to_idx(obs)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx*dx + dy*dy <= radius_sq:
                        tx, ty = ox + dx, oy + dy
                        if 0 <= tx < res and 0 <= ty < res:
                            # Don't block the actual start/end nodes
                            if (tx, ty) != start_node and (tx, ty) != end_node:
                                obstacle_indices.add((tx, ty))
        
        # 4. A* Algorithm
        def heuristic(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: heuristic(start_node, end_node)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end_node:
                # Reconstruct Path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                # Convert indices back to coordinates
                return [(grid_x[p[0]], grid_y[p[1]]) for p in path[::-1]]
                
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < res and 0 <= neighbor[1] < res:
                    if neighbor in obstacle_indices:
                        continue # Skip obstacles
                        
                    # Diagonal cost is sqrt(2)
                    step_cost = np.sqrt(dx*dx + dy*dy)
                    tentative_g = g_score[current] + step_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, end_node)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Fallback to straight line if path not found
        return [start_coords, end_coords]

    def plot_2d_map(self):
        self.ax_2d.clear()
        wells_df = self.data_loader.wells_info
        c = self.get_plot_colors()
        
        self.canvas_2d.figure.set_facecolor(c['bg'])
        self.ax_2d.set_facecolor(c['bg'])
        
        ref_well = self.ref_well_combo.currentText()
        off_well = self.off_well_combo.currentText()
        
        id_col = "Well identifier (Well name)"
        
        if self.rb_chosen_map.isChecked():
            # Plot connection path if two distinct wells selected
            if ref_well and off_well and ref_well != off_well:
                try:
                    w1 = wells_df[wells_df[id_col] == ref_well].iloc[0]
                    w2 = wells_df[wells_df[id_col] == off_well].iloc[0]
                    
                    start_pt = (w1['X'], w1['Y'])
                    end_pt = (w2['X'], w2['Y'])
                    
                    # Others are obstacles
                    obstacles = []
                    for idx, row in wells_df.iterrows():
                        if row[id_col] != ref_well and row[id_col] != off_well:
                            obstacles.append((row['X'], row['Y']))
                    
                    path = self.find_avoidance_path(start_pt, end_pt, obstacles)
                    
                    # Plot the path
                    px, py = zip(*path)
                    self.ax_2d.plot(px, py, color=TURQUOISE, linestyle='--', linewidth=2.5, 
                                    alpha=0.8, zorder=2, label="Connection Path")
                                    
                except Exception as e:
                    print(f"DEBUG: Path plotting error: {e}")
        else:
            # Route Plotter Mode
            selected_items = self.route_wells_list.selectedItems()
            selected_names = [item.text() for item in selected_items]
            
            if len(selected_names) > 1:
                try:
                    coords = []
                    # Maximum 10 wells
                    for name in selected_names[:10]:
                        w = wells_df[wells_df[id_col] == name].iloc[0]
                        coords.append((w['X'], w['Y']))
                    
                    px, py = zip(*coords)
                    # "Garis rantai" -> dashdot '-.'
                    self.ax_2d.plot(px, py, color="#ffd700", linestyle='-.', linewidth=2.0,
                                    marker='o', markerfacecolor=TURQUOISE, markersize=8,
                                    alpha=0.9, zorder=2, label="Route Path")
                except Exception as e:
                    print(f"DEBUG: Route plotting error: {e}")

        # Plot all wells with same style (turquoise)
        self.ax_2d.scatter(wells_df['X'], wells_df['Y'], 
                           c=TURQUOISE, s=70, edgecolors='black', linewidth=1.2, zorder=4, label="Well Locations")
        
        # Add legend
        leg = self.ax_2d.legend(loc='lower right', frameon=True, fontsize=9)
        if leg:
            frame = leg.get_frame()
            frame.set_facecolor(c['bg'])
            frame.set_edgecolor(c['spine'])
            for text in leg.get_texts():
                text.set_color(c['text'])
        
        # Add labels - simple text without background
        texts = []
        for _, row in wells_df.iterrows():
            t = self.ax_2d.text(row['X'], row['Y'], 
                            str(row[id_col]),
                            fontsize=9, color=c['text'], fontweight='normal', zorder=5)
            texts.append(t)
            
        self.ax_2d.set_title("2D Well Location Map", fontweight="bold", pad=15, fontsize=12, color=c['text'])
        self.ax_2d.set_xlabel("X Coordinate (UTM)", color=c['text'])
        self.ax_2d.set_ylabel("Y Coordinate (UTM)", color=c['text'])
        
        self.ax_2d.tick_params(colors=c['text'], which='both')
        for spine in self.ax_2d.spines.values():
            spine.set_color(c['spine'])
            
        self.ax_2d.grid(True, linestyle='--', alpha=0.3, color=c['grid'])
        self.ax_2d.set_axisbelow(True)
        
        # Use adjustText to prevent label overlap with connecting lines
        if adjust_text and texts:
            adjust_text(texts, ax=self.ax_2d, 
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.6, shrinkA=0, shrinkB=0),
                       expand_points=(1.5, 1.5),
                       force_points=(0.5, 0.5),
                       force_text=(0.5, 0.5))
            
        # Redraw sketches
        self.draw_mgr_2d.redraw([self.ax_2d])
        self.canvas_2d.draw()

    def init_corr_tab(self):
        layout = QVBoxLayout(self.corr_tab)
        c = self.get_plot_colors()
        
        # Matplotlib Figure for Logs
        self.canvas_corr = FigureCanvas(Figure(figsize=(10, 15), facecolor=c['bg']))
        self.fig_corr = self.canvas_corr.figure
        
        self.draw_mgr_corr = DrawingManager(self.canvas_corr, on_change=self.refresh_active_tab)
        
        # Toolbar
        self.toolbar_corr = VerticalToolbar(self.canvas_corr, self.corr_tab)
        layout.addWidget(self.toolbar_corr)
        
        # Create initial empty layout using current slider ratio
        self.draw_empty_tracks()
        
        layout.addWidget(self.canvas_corr)

    def init_log_tab(self):
        layout = QVBoxLayout(self.log_tab)
        
        # Row 1: Well and Curve Selection
        controls1 = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Logs")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self.update_available_logs)
        
        self.log_ref_lbl = QLabel("Ref Well Logs:")
        self.log_ref_lbl.setStyleSheet("font-weight: bold; color: silver;")
        self.ref_curve_list = QListWidget()
        self.ref_curve_list.setFixedHeight(80)
        self.ref_curve_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.ref_curve_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {SIDEBAR_BG};
                color: {SILVER};
                border: 1px solid {SILVER};
            }}
            QListWidget::item:selected {{
                background-color: #4a4a4a;
                border: 1px solid {TURQUOISE};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: #444444;
            }}
        """)
        
        self.ref_curve_list.itemSelectionChanged.connect(self.validate_selection_limit)
        
        self.log_off_lbl = QLabel("Off Well Logs:")
        self.log_off_lbl.setStyleSheet("font-weight: bold; color: silver;")
        self.off_curve_list = QListWidget()
        self.off_curve_list.setFixedHeight(80)
        self.off_curve_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.off_curve_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {SIDEBAR_BG};
                color: {SILVER};
                border: 1px solid {SILVER};
            }}
            QListWidget::item:selected {{
                background-color: #4a4a4a;
                border: 1px solid {TURQUOISE};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: #444444;
            }}
        """)
        self.off_curve_list.itemSelectionChanged.connect(self.validate_selection_limit)
        
        controls1.addWidget(refresh_btn)
        
        ref_vbox = QVBoxLayout(); ref_vbox.addWidget(self.log_ref_lbl); ref_vbox.addWidget(self.ref_curve_list)
        off_vbox = QVBoxLayout(); off_vbox.addWidget(self.log_off_lbl); off_vbox.addWidget(self.off_curve_list)
        
        controls1.addLayout(ref_vbox)
        controls1.addLayout(off_vbox)
        controls1.addStretch()
        
        # Row 2: Lithology Shading Controls
        controls2 = QHBoxLayout()
        
        plot_log_btn = QPushButton("Plot Logs")
        plot_log_btn.setFixedWidth(100)
        plot_log_btn.clicked.connect(self.plot_logs)
        
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["Normal Log Curve", "Vshale Analysis (GR)", "Lithology Analysis"])
        self.plot_mode_combo.setFixedWidth(170)
        
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet("font-weight: bold; color: silver;")
        
        self.log_unit_combo = QComboBox()
        self.log_unit_combo.addItems(["Meters", "Feet"])
        self.log_unit_combo.setFixedWidth(100)
        self.log_unit_combo.currentIndexChanged.connect(self.sync_units_from_log)
        
        log_unit_lbl = QLabel("Unit:")
        log_unit_lbl.setStyleSheet("font-weight: bold; color: silver;")
        
        controls2.addWidget(log_unit_lbl)
        controls2.addWidget(self.log_unit_combo)
        controls2.addSpacing(20)
        controls2.addWidget(mode_lbl)
        controls2.addWidget(self.plot_mode_combo)
        controls2.addSpacing(20)
        controls2.addWidget(plot_log_btn)
        controls2.addStretch()
        
        layout.addLayout(controls1)
        layout.addLayout(controls2)
        
        # Plot Area
        # Plot Area - Very tall aspect ratio for professional elongated logging tracks
        c = self.get_plot_colors()
        self.canvas_log = FigureCanvas(Figure(figsize=(7, 14), facecolor=c['bg']))
        self.fig_log = self.canvas_log.figure
        self.toolbar_log = VerticalToolbar(self.canvas_log, self.log_tab)
        
        self.draw_mgr_log = DrawingManager(self.canvas_log, on_change=self.refresh_active_tab)
        
        layout.addWidget(self.toolbar_log)
        layout.addWidget(self.canvas_log)
    
    def update_unit_ui(self):
        unit_idx = self.unit_combo.currentIndex()
        to_meters = (unit_idx == 0) # 0: Meters, 1: Feet
        
        # 1. Update Internal Multiplier
        if to_meters:
            self.depth_multiplier = 1.0
            self.depth_unit = "m"
        else:
            self.depth_multiplier = 3.28084
            self.depth_unit = "ft"

        # Block signals to avoid infinite loop
        self.log_unit_combo.blockSignals(True)
        self.log_unit_combo.setCurrentIndex(unit_idx)
        self.log_unit_combo.blockSignals(False)
        
        # 2. INSTANT REFRESH: Just trigger redraws with new multiplier
        # No re-calculation, just re-plotting
        self.refresh_active_tab()
        
        self.status_bar.showMessage(f"Units switched to {self.unit_combo.currentText()}. Refresh complete.")

    def sync_units_from_log(self):
        unit_idx = self.log_unit_combo.currentIndex()
        # Sync sidebar combo
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentIndex(unit_idx)
        self.unit_combo.blockSignals(False)
        # Manually trigger the side-effects of unit change
        self.update_unit_ui()

    def init_mrgc_tab(self):
        """Initializes the MRGC Clustering tab with zero-margin layout and professional controls."""
        layout = QVBoxLayout(self.mrgc_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Splitter to divide Controls and Plot
        mrgc_splitter = QSplitter(Qt.Horizontal)
        
        # --- Internal Sidebar for MRGC ---
        mrgc_side = QWidget()
        mrgc_side.setFixedWidth(280)
        mrgc_side_layout = QVBoxLayout(mrgc_side)
        mrgc_side_layout.setContentsMargins(10, 10, 10, 10)
        mrgc_side_layout.setSpacing(10)
        
        mrgc_header = QLabel("MRGC CLUSTERING ENGINE")
        mrgc_header.setAlignment(Qt.AlignCenter)
        mrgc_header.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TURQUOISE};")
        mrgc_side_layout.addWidget(mrgc_header)
        
        # 1. Curve Selection Groups
        for i in [1, 2]:
            well_type = "REFERENCE" if i == 1 else "OFFSET"
            group = QGroupBox(f"{i}. {well_type} LOGS")
            g_layout = QVBoxLayout(group)
            g_layout.setContentsMargins(5, 15, 5, 5)
            
            h_ctrl = QHBoxLayout()
            h_ctrl.addWidget(QLabel("Select Features:"))
            btn_all = QPushButton("ALL")
            btn_all.setFixedHeight(18); btn_all.setFixedWidth(40)
            btn_all.setStyleSheet("font-size: 8px; padding: 0;")
            btn_all.clicked.connect(lambda _, x=i: self.toggle_mrgc_all(x))
            h_ctrl.addWidget(btn_all)
            g_layout.addLayout(h_ctrl)
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(120)
            scroll.setStyleSheet(f"background-color: {SIDEBAR_BG}; border: 1px solid #555;")
            
            feat_widget = QWidget()
            feat_widget.setStyleSheet(f"background-color: {SIDEBAR_BG};")
            feat_layout = QVBoxLayout(feat_widget)
            feat_layout.setContentsMargins(5, 5, 5, 5); feat_layout.setSpacing(2)
            scroll.setWidget(feat_widget)
            
            if i == 1: 
                self.mrgc_w1_feat_layout = feat_layout
                self.mrgc_w1_checks = []
            else: 
                self.mrgc_w2_feat_layout = feat_layout
                self.mrgc_w2_checks = []
                
            g_layout.addWidget(scroll)
            mrgc_side_layout.addWidget(group)
            
        # 2. Parameters
        param_group = QGroupBox("3. CLUSTERING PARAMS")
        p_layout = QVBoxLayout(param_group); p_layout.setContentsMargins(5, 15, 5, 5)
        
        h_alpha = QHBoxLayout()
        h_alpha.addWidget(QLabel("Smoothing (\u03b1):"))
        self.mrgc_alpha_spin = QDoubleSpinBox()
        self.mrgc_alpha_spin.setRange(0.1, 100.0); self.mrgc_alpha_spin.setValue(10.0)
        self.mrgc_alpha_spin.setStyleSheet(f"background-color: {DARK_BG}; color: {SILVER};")
        h_alpha.addWidget(self.mrgc_alpha_spin)
        
        self.mrgc_rec_btn = QPushButton("REC")
        self.mrgc_rec_btn.setFixedHeight(22); self.mrgc_rec_btn.setFixedWidth(42)
        self.mrgc_rec_btn.setStyleSheet(f"font-size: 8px; background-color: {TURQUOISE}; color: {DARK_BG}; font-weight: bold;")
        self.mrgc_rec_btn.clicked.connect(self.suggest_mrgc_alpha)
        h_alpha.addWidget(self.mrgc_rec_btn)
        p_layout.addLayout(h_alpha)
        
        self.mrgc_rec_label = QLabel("Recommended: --")
        self.mrgc_rec_label.setStyleSheet(f"color: {TURQUOISE}; font-style: italic; font-size: 10px; margin-left: 2px;")
        p_layout.addWidget(self.mrgc_rec_label)
        
        h_c = QHBoxLayout()
        h_c.addWidget(QLabel("N Clusters:"))
        self.mrgc_clusters_combo = QComboBox()
        self.mrgc_clusters_combo.addItems(["Auto"] + [str(i) for i in range(2, 11)])
        h_c.addWidget(self.mrgc_clusters_combo)
        p_layout.addLayout(h_c)
        mrgc_side_layout.addWidget(param_group)
        
        # 3. Actions
        self.mrgc_run_btn = QPushButton("RUN MRGC ANALYSIS")
        self.mrgc_run_btn.setFixedHeight(40)
        self.mrgc_run_btn.clicked.connect(self.run_mrgc_analysis)
        mrgc_side_layout.addWidget(self.mrgc_run_btn)
        
        self.mrgc_view_btn = QPushButton("VIEW CLUSTERING DATA")
        self.mrgc_view_btn.setFixedHeight(40)
        self.mrgc_view_btn.setStyleSheet(f"background-color: #f0f0f0; color: #1a1a1a;")
        self.mrgc_view_btn.setEnabled(False)
        self.mrgc_view_btn.clicked.connect(self.open_mrgc_viewer)
        mrgc_side_layout.addWidget(self.mrgc_view_btn)
        
        self.mrgc_pbar = QProgressBar()
        self.mrgc_pbar.setFixedHeight(8); self.mrgc_pbar.hide()
        mrgc_side_layout.addWidget(self.mrgc_pbar)
        
        mrgc_side_layout.addStretch()
        
        # --- Plot Area for MRGC ---
        mrgc_plot_panel = QWidget()
        mrgc_plot_layout = QVBoxLayout(mrgc_plot_panel)
        mrgc_plot_layout.setContentsMargins(0,0,0,0); mrgc_plot_layout.setSpacing(0)
        
        mrgc_top_ctrl = QFrame()
        mrgc_top_ctrl.setStyleSheet(f"background-color: #252525; border-bottom: 1px solid #444;")
        mrgc_top_ctrl_layout = QHBoxLayout(mrgc_top_ctrl)
        mrgc_top_ctrl_layout.setContentsMargins(10, 2, 10, 2)
        
        self.mrgc_hover_info = QLabel("Hover plot for depth & log cluster insight")
        self.mrgc_hover_info.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold; font-size: 11px;")
        mrgc_top_ctrl_layout.addWidget(self.mrgc_hover_info)
        mrgc_top_ctrl_layout.addStretch()
        
        self.canvas_mrgc = FigureCanvas(Figure(figsize=(10, 8), facecolor='#202020'))
        self.toolbar_mrgc = VerticalToolbar(self.canvas_mrgc, self)
        self.toolbar_mrgc.setStyleSheet("background-color: transparent; border: none;")
        mrgc_top_ctrl_layout.addWidget(self.toolbar_mrgc)
        
        mrgc_plot_layout.addWidget(mrgc_top_ctrl)
        mrgc_plot_layout.addWidget(self.canvas_mrgc, 1)
        
        self.draw_mgr_mrgc = DrawingManager(self.canvas_mrgc, on_change=self.refresh_active_tab)
        self.canvas_mrgc.mpl_connect('motion_notify_event', self.on_mrgc_hover)
        
        mrgc_splitter.addWidget(mrgc_side)
        mrgc_splitter.addWidget(mrgc_plot_panel)
        mrgc_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(mrgc_splitter)

    def init_rf_tab(self):
        """Initializes the Random Forest Classification tab with feature selection."""
        layout = QVBoxLayout(self.rf_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        c = self.get_plot_colors()
        
        # Main splitter
        rf_splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT SIDE: Controls ---
        rf_side = QWidget()
        rf_side.setMaximumWidth(350)
        rf_side.setStyleSheet(f"background-color: {SIDEBAR_BG};")
        side_layout = QVBoxLayout(rf_side)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)
        
        # Title
        title_lbl = QLabel("Random Forest Classification")
        title_lbl.setStyleSheet(f"color: {TURQUOISE}; font-size: 14px; font-weight: bold;")
        side_layout.addWidget(title_lbl)
        
        # Well Selection (Reference & Offset like MRGC)
        well_grp = QGroupBox("Well Selection")
        well_grp.setStyleSheet(f"QGroupBox {{ color: {TEXT_COLOR}; font-weight: bold; border: 1px solid {SILVER}; border-radius: 5px; margin-top: 10px; padding-top: 10px; }}")
        well_grp_layout = QFormLayout(well_grp)
        
        # Reference Well
        self.rf_ref_combo = QComboBox()
        self.rf_ref_combo.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COLOR};")
        for well in self.data_loader.get_well_names():
            self.rf_ref_combo.addItem(well)
        well_grp_layout.addRow(QLabel("Reference Well:"), self.rf_ref_combo)
        
        # Offset Well
        self.rf_off_combo = QComboBox()
        self.rf_off_combo.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COLOR};")
        for well in self.data_loader.get_well_names():
            self.rf_off_combo.addItem(well)
        if self.rf_off_combo.count() > 1:
            self.rf_off_combo.setCurrentIndex(1)
        well_grp_layout.addRow(QLabel("Offset Well:"), self.rf_off_combo)
        
        for i in range(well_grp_layout.rowCount()):
            label = well_grp_layout.itemAt(i, QFormLayout.LabelRole)
            if label and label.widget():
                label.widget().setStyleSheet(f"color: {TEXT_COLOR};")
        
        side_layout.addWidget(well_grp)
        
        # Feature Selection
        feature_grp = QGroupBox("Log Features")
        feature_grp.setStyleSheet(f"QGroupBox {{ color: {TEXT_COLOR}; font-weight: bold; border: 1px solid {SILVER}; border-radius: 5px; margin-top: 10px; padding-top: 10px; }}")
        feature_grp_layout = QVBoxLayout(feature_grp)
        
        self.rf_feature_checks = {}
        default_features = ['GR', 'RHOB', 'DT', 'SP', 'ILD', 'NPHI', 'CALI']
        for feat in default_features:
            cb = QCheckBox(feat)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {TEXT_COLOR};")
            self.rf_feature_checks[feat] = cb
            feature_grp_layout.addWidget(cb)
        side_layout.addWidget(feature_grp)
        # Parameters Group
        param_grp = QGroupBox("Parameters")
        param_grp.setStyleSheet(f"QGroupBox {{ color: {TEXT_COLOR}; font-weight: bold; border: 1px solid {SILVER}; border-radius: 5px; margin-top: 10px; padding-top: 10px; }}")
        param_layout = QFormLayout(param_grp)
        
        # Vshale Cutoff
        self.rf_vsh_cutoff = QDoubleSpinBox()
        self.rf_vsh_cutoff.setRange(0.1, 0.9)
        self.rf_vsh_cutoff.setSingleStep(0.05)
        self.rf_vsh_cutoff.setValue(0.4)
        self.rf_vsh_cutoff.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COLOR};")
        param_layout.addRow(QLabel("Vsh Cutoff (Sand):"), self.rf_vsh_cutoff)
        
        # N Estimators
        self.rf_n_estimators = QSpinBox()
        self.rf_n_estimators.setRange(50, 1000)
        self.rf_n_estimators.setValue(100)
        self.rf_n_estimators.setSingleStep(50)
        self.rf_n_estimators.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COLOR};")
        param_layout.addRow(QLabel("N Estimators:"), self.rf_n_estimators)
        
        # Auto Optimize checkbox
        self.rf_auto_optimize = QCheckBox("Auto-Optimize")
        self.rf_auto_optimize.setStyleSheet(f"color: {TEXT_COLOR};")
        param_layout.addRow(self.rf_auto_optimize)
        
        for i in range(param_layout.rowCount()):
            label = param_layout.itemAt(i, QFormLayout.LabelRole)
            if label and label.widget():
                label.widget().setStyleSheet(f"color: {TEXT_COLOR};")
        
        side_layout.addWidget(param_grp)
        
        # Buttons
        btn_style = f"background-color: {TURQUOISE}; color: #000; font-weight: bold; padding: 8px; border-radius: 5px;"
        
        self.rf_train_btn = QPushButton("TRAIN MODEL")
        self.rf_train_btn.setStyleSheet(btn_style)
        self.rf_train_btn.clicked.connect(self.run_rf_training)
        side_layout.addWidget(self.rf_train_btn)
        
        # Results Display
        self.rf_results_text = QTextEdit()
        self.rf_results_text.setReadOnly(True)
        self.rf_results_text.setMaximumHeight(180)
        self.rf_results_text.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COLOR}; font-family: Consolas; font-size: 10px;")
        side_layout.addWidget(self.rf_results_text)
        
        side_layout.addStretch()
        
        # --- RIGHT SIDE: Plot ---
        rf_plot_panel = QWidget()
        rf_plot_layout = QVBoxLayout(rf_plot_panel)
        rf_plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig_rf = Figure(figsize=(12, 8), facecolor=c['bg'])
        self.canvas_rf = FigureCanvas(self.fig_rf)
        self.canvas_rf.setStyleSheet(f"background-color: {c['bg']};")
        
        self.toolbar_rf = VerticalToolbar(self.canvas_rf, self.rf_tab)
        rf_plot_layout.addWidget(self.toolbar_rf)
        rf_plot_layout.addWidget(self.canvas_rf, 1)
        
        self.draw_mgr_rf = DrawingManager(self.canvas_rf, on_change=self.refresh_active_tab)
        
        rf_splitter.addWidget(rf_side)
        rf_splitter.addWidget(rf_plot_panel)
        rf_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(rf_splitter)
        
        # Sync initially
        self.sync_rf_wells()
        
        # Initialize RF Engine
        self.rf_engine = RandomForestLithologyEngine(self.data_loader)
        self.rf_trained = False
        self.rf_predictions_ref = None
        self.rf_predictions_off = None
    
    def run_rf_training(self):
        """Execute Random Forest training and prediction in background thread."""
        from PyQt5.QtWidgets import QMessageBox
        if self.rb_route_map.isChecked():
            QMessageBox.warning(self, "Route Mode Active", 
                                "Random Forest training is disabled while in Route Plotter mode.")
            return

        ref_well = self.rf_ref_combo.currentText()
        off_well = self.rf_off_combo.currentText()
        
        if ref_well == off_well:
            QMessageBox.warning(self, "Selection", "Please select different wells for Reference and Offset.")
            return
        
        selected_features = [feat for feat, cb in self.rf_feature_checks.items() if cb.isChecked()]
        if len(selected_features) < 2:
            QMessageBox.warning(self, "Selection", "Please select at least 2 features.")
            return
        
        # Prepare params
        params = {
            'wells': [ref_well, off_well],
            'features': selected_features,
            'vsh_cutoff': self.rf_vsh_cutoff.value(),
            'n_estimators': None if self.rf_auto_optimize.isChecked() else self.rf_n_estimators.value(),
            'optimize': self.rf_auto_optimize.isChecked(),
            'use_gpu': self.use_gpu
        }
        
        # UI State
        self.rf_train_btn.setEnabled(False)
        self.status_bar.showMessage("Training Random Forest model in background...")
        self.rf_results_text.setText("Training in progress... please wait.")
        
        # Start Thread
        self.rf_thread = RFTrainingThread(self.rf_engine, params)
        self.rf_thread.finished.connect(self.on_rf_training_fin)
        self.rf_thread.error.connect(self.on_rf_training_error)
        self.rf_thread.start()

    def on_rf_training_error(self, err):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Training Error", err)
        self.rf_train_btn.setEnabled(True)
        self.status_bar.showMessage("Training failed.")
        self.rf_results_text.setText(f"Error occurred:\n{err}")

    def on_rf_training_fin(self, output):
        from PyQt5.QtWidgets import QMessageBox
        try:
            rf_metrics = output['rf_metrics']
            svm_metrics = output['svm_metrics']
            features = output['features']
            results = output['results']
            
            # Update results text
            ref_well = self.rf_ref_combo.currentText()
            off_well = self.rf_off_combo.currentText()
            
            result_text = f"=== Training Results ===\n"
            result_text += f"Reference: {ref_well}\n"
            result_text += f"Offset: {off_well}\n"
            result_text += f"Features: {', '.join(features)}\n"
            result_text += f"Samples: Train={output['train_size']}, Test={output['test_size']}\n\n"
            result_text += f"--- Random Forest ---\n"
            result_text += f"Accuracy: {rf_metrics['accuracy']:.2%}\n"
            result_text += f"N Estimators: {rf_metrics.get('n_estimators', 'N/A')}\n"
            result_text += f"OOB Score: {rf_metrics.get('oob_score', 0):.2%}\n\n"
            result_text += f"--- SVM Baseline ---\n"
            result_text += f"Accuracy: {svm_metrics['accuracy']:.2%}\n\n"
            result_text += f"--- Variable Importance ---\n"
            for _, row in self.rf_engine.importance_df.iterrows():
                result_text += f"{row['Feature']}: {row['Gini_Importance']:.1%}\n"
            
            self.rf_results_text.setText(result_text)
            
            # Store results
            self.rf_predictions_ref = results.get(ref_well)
            self.rf_predictions_off = results.get(off_well)
            self.rf_results.update(results)
            
            self.rf_trained = True
            
            # Plot
            self.plot_rf_results_dual(ref_well, off_well)
            self.status_bar.showMessage(f"Training Complete! RF Accuracy: {rf_metrics['accuracy']:.2%}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Plot Error", f"Failed to display results: {str(e)}\n{traceback.format_exc()}")
            
        self.rf_train_btn.setEnabled(True)

    def plot_rf_results(self):
        """Wrapper for refresh_active_tab to use current wells."""
        if self.rf_trained:
            ref_well = self.rf_ref_combo.currentText()
            off_well = self.rf_off_combo.currentText()
            self.plot_rf_results_dual(ref_well, off_well)

    def sync_rf_wells(self):
        """Synchronizes Random Forest well selection with the main sidebar selection."""
        if hasattr(self, 'rf_ref_combo') and hasattr(self, 'rf_off_combo'):
            self.rf_ref_combo.setCurrentText(self.ref_well_combo.currentText())
            self.rf_off_combo.setCurrentText(self.off_well_combo.currentText())
            # If RF is already trained, we might want to refresh the plot, 
            # but usually user triggers training manually.
        else:
            self.plot_rf_results_dual(None, None)

    def plot_rf_results_dual(self, ref_well, off_well):
        """Plot RF results for both Reference and Offset wells side-by-side."""
        c = self.get_plot_colors()
        self.fig_rf.set_facecolor(c['bg'])
        self.fig_rf.clf()
        
        if not self.rf_trained:
            ax = self.fig_rf.add_subplot(111)
            ax.set_facecolor(c['bg'])
            ax.text(0.5, 0.5, "Select Wells and Click 'TRAIN' to See Results", 
                    ha='center', va='center', color=c['text'], fontsize=14, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas_rf.draw()
            return

        # Check if predictions exist
        has_ref = hasattr(self, 'rf_predictions_ref') and self.rf_predictions_ref is not None
        has_off = hasattr(self, 'rf_predictions_off') and self.rf_predictions_off is not None

        # Determine grid
        n_cols = 1 # Always show importance
        if has_ref: n_cols += 2 # Lith + Prob for Ref
        if has_off: n_cols += 2 # Lith + Prob for Off
        
        # Width ratios: Importance (1.2), Lith (0.5), Prob (0.8), Lith (0.5), Prob (0.8)
        w_ratios = [1.2]
        if has_ref: w_ratios.extend([0.5, 0.8])
        if has_off: w_ratios.extend([0.5, 0.8])
        
        gs = self.fig_rf.add_gridspec(1, n_cols, width_ratios=w_ratios, wspace=0.15)
        
        # 1. Variable Importance
        all_axes = []
        ax_imp = self.fig_rf.add_subplot(gs[0])
        ax_imp.set_label("rf_importance")
        if self.rf_engine.importance_df is not None:
            imp_df = self.rf_engine.importance_df.sort_values('Gini_Importance', ascending=True)
            colors = ['#00CED1' if i >= len(imp_df)-2 else '#708090' for i in range(len(imp_df))]
            ax_imp.barh(imp_df['Feature'], imp_df['Gini_Importance'], color=colors[::-1])
            ax_imp.set_title('Importance', fontweight='normal', color=c['text'], fontsize=10)
            ax_imp.set_xlabel('Gini Index', color=c['text'], fontsize=8)
            ax_imp.tick_params(labelsize=8, colors=c['text'])
            ax_imp.set_facecolor(c['bg'])
            for spine in ax_imp.spines.values(): spine.set_color(c['spine'])
        all_axes.append(ax_imp)

        # 2. Reference Well Tracks
        curr_col = 1
        if has_ref:
            ref_axes = self._plot_well_rf_tracks(self.fig_rf, gs, curr_col, ref_well, self.rf_predictions_ref, c, is_ref=True)
            all_axes.extend(ref_axes)
            curr_col += 2
            
        # 3. Offset Well Tracks
        if has_off:
            off_axes = self._plot_well_rf_tracks(self.fig_rf, gs, curr_col, off_well, self.rf_predictions_off, c, is_ref=False)
            all_axes.extend(off_axes)
        
        # Redraw any persistent annotations
        self.draw_mgr_rf.redraw(all_axes)
        
        self.fig_rf.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Shared Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ffd700', edgecolor='k', label='Shaly Sand'),
            Patch(facecolor='#8B4513', edgecolor='k', label='Shale')
        ]
        self.fig_rf.legend(handles=legend_elements, loc='lower center', ncol=2, 
                          title="Legend dari Random forest", title_fontsize=10,
                          fontsize=9, facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'])
        
        self.canvas_rf.draw()

    def _plot_well_rf_tracks(self, fig, gs, start_col, well_name, df, c, is_ref=True):
        """Helper to plot Lithology and Probability for a single well."""
        # Ensure data is sorted from shallow to deep
        df = df.sort_values('Depth')
        
        # Strict Filtering: Filter data to INPEFA overlap if correlation exists
        if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
            grid_overlap = self.last_plot_args[0]
            o_min, o_max = grid_overlap.min(), grid_overlap.max()
            df = df[(df['Depth'] >= o_min) & (df['Depth'] <= o_max)]
        
        if df.empty:
            # Just setup axes for label persistence but skip drawing
            ax_lith = fig.add_subplot(gs[start_col])
            ax_prob = fig.add_subplot(gs[start_col+1], sharey=ax_lith)
            ax_lith.set_label(f"rf_{'ref' if is_ref else 'off'}_lith")
            ax_prob.set_label(f"rf_{'ref' if is_ref else 'off'}_prob")
            return [ax_lith, ax_prob]

        ax_lith = fig.add_subplot(gs[start_col])
        ax_prob = fig.add_subplot(gs[start_col+1], sharey=ax_lith)
        
        depths = df['Depth'].values * self.depth_multiplier
        labels = df['Predicted_Lithology'].values
        
        # Lithology Track (High-Fidelity fill_betweenx loop)
        color_map = {0: '#8B4513', 1: '#ffd700', 2: '#8B4513', 3: '#8B4513'}
        if len(depths) > 0:
            curr_p, start_i = labels[0], 0
            for i in range(1, len(labels)):
                if labels[i] != curr_p:
                    color = color_map.get(int(curr_p), '#808080')
                    ax_lith.fill_betweenx(depths[start_i:i+1], 0, 1, color=color, alpha=0.9)
                    curr_p, start_i = labels[i], i
            color = color_map.get(int(curr_p), '#808080')
            ax_lith.fill_betweenx(depths[start_i:], 0, 1, color=color, alpha=0.9)

        
        ax_lith.set_title(f"{'REF' if is_ref else 'OFF'}\n{well_name}", fontsize=9, fontweight='normal', color=c['text'])
        ax_lith.set_xticks([]); ax_lith.set_facecolor(c['bg'])
        
        # 4. Set Depth Limits (prioritize INPEFA overlap if correlation has been run)
        if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
            grid_overlap = self.last_plot_args[0]
            ax_lith.set_ylim(grid_overlap.max() * self.depth_multiplier, grid_overlap.min() * self.depth_multiplier)
        else:
            ax_lith.set_ylim(depths.max(), depths.min()) # Force inversion: shallow at top
        
        ax_lith.tick_params(labelsize=8, colors=c['text'])
        for spine in ax_lith.spines.values(): spine.set_color(c['spine'])
        if not is_ref: ax_lith.set_yticklabels([]) # Hide depth values for offset to save space
        
        # Probability Track (Show Sand Prob as representative)
        import numpy as np # Added import for np
        p_sand = df['Probability_Sand'].values if 'Probability_Sand' in df.columns else np.zeros(len(df))
        ax_prob.plot(p_sand, depths, color='#FFFF00', linewidth=1)
        ax_prob.fill_betweenx(depths, p_sand, 0, color='#FFFF00', alpha=0.3)
        ax_prob.set_xlim(0, 1)
        ax_prob.set_title('Shaly Sand Prob.', fontsize=8, color=c['text'])
        ax_prob.set_facecolor(c['bg'])
        ax_prob.tick_params(labelsize=8, colors=c['text'], labelleft=False)
        for spine in ax_prob.spines.values(): spine.set_color(c['spine'])
        
        # Set labels for annotation persistence
        ax_lith.set_label(f"rf_{'ref' if is_ref else 'off'}_lith")
        ax_prob.set_label(f"rf_{'ref' if is_ref else 'off'}_prob")

        return [ax_lith, ax_prob]

    def init_geology_tab(self):
        """Initializes the Geology Marker tab with a professional, interactive vertical layout."""
        layout = QVBoxLayout(self.geology_tab)
        layout.setContentsMargins(5, 5, 5, 5) # Tightened margins
        layout.setSpacing(2)
        c = self.get_plot_colors()
        
        # 1. Header/Toolbar Row
        top_ctrl = QFrame()
        top_ctrl.setStyleSheet(f"background-color: {c['bg']}; border-bottom: 1px solid {c['spine']};")
        top_layout = QHBoxLayout(top_ctrl)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        # The single canvas instance (Reduced height for static/zoomable view)
        self.canvas_geo = FigureCanvas(Figure(figsize=(12, 10), facecolor=c['bg']))
        self.fig_geo = self.canvas_geo.figure
        self.draw_mgr_geo = DrawingManager(self.canvas_geo, on_change=self.refresh_active_tab)
        
        self.toolbar_geo = VerticalToolbar(self.canvas_geo, self.geology_tab)
        self.toolbar_geo.setStyleSheet(f"background-color: transparent; border: none;")
        
        self.geo_hover_info = QLabel("Hover for info")
        self.geo_hover_info.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold; font-family: 'Segoe UI', sans-serif;")
        
        self.geo_corr_btn = QPushButton("Show Marker Correlation")
        self.geo_corr_btn.setCheckable(True)
        self.geo_corr_btn.setChecked(True)
        self.geo_corr_btn.setFixedWidth(200)
        self.geo_corr_btn.setCursor(Qt.PointingHandCursor)
        self.geo_corr_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['bg']};
                color: {TURQUOISE};
                border: 2px solid {TURQUOISE};
                border-radius: 6px;
                font-weight: bold;
                padding: 6px;
            }}
            QPushButton:hover {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
            }}
            QPushButton:checked {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
                border: 2px solid #555;
            }}
            QPushButton:pressed {{
                background-color: "#008B8B";
                border: 2px solid #333;
            }}
        """)
        self.geo_corr_btn.clicked.connect(self.toggle_geo_correlation)
        
        top_layout.addWidget(self.toolbar_geo)
        top_layout.addStretch()
        top_layout.addWidget(self.geo_hover_info)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.geo_corr_btn)
        
        layout.addWidget(top_ctrl)
        
        # 2. Scroll Area for the tall plot
        self.scroll_geo = QScrollArea()
        self.scroll_geo.setWidgetResizable(True)
        self.scroll_geo.setFrameShape(QFrame.NoFrame)
        self.scroll_geo.setStyleSheet(f"background-color: {c['bg']};")
        
        self.scroll_geo.setWidget(self.canvas_geo)
        layout.addWidget(self.scroll_geo)
        
        self.canvas_geo.mpl_connect("motion_notify_event", self.on_geo_hover)

    def toggle_geo_correlation(self, checked):
        self.show_geo_corr = checked
        self.plot_geology_markers()

    def on_geo_hover(self, event):
        if event.inaxes is None: return
        try:
            well = event.inaxes.get_label()
            depth = event.ydata # This is already in display units because we use data coords
            self.geo_hover_info.setText(f"Well: {well} | Depth: {depth:.2f}{self.depth_unit}")
        except: pass

    def _draw_stepped_correlation_line(self, fig, y_ref, y_off, r_list, o_list, color, label=None, **kwargs):
        """Helper to draw Petrel-style stepped correlation lines across multiple axes."""
        from matplotlib.patches import ConnectionPatch
        from matplotlib.transforms import Bbox, TransformedBbox
        
        # 0. Boundary Check Preparation
        if not r_list or not o_list: return
        
        # Create a safe vertical clipping box in absolute figure coordinates
        # Using get_position() provides absolute figure fractions (0.0 to 1.0)
        try:
            pos = r_list[0].get_position()
            # Clip vertically to panel bounds but allow full width (0 to 1) for labels
            f_bbox = Bbox.from_extents(0, pos.y0, 1.0, pos.y1)
            safe_bbox = TransformedBbox(f_bbox, fig.transFigure)
        except:
            safe_bbox = fig.bbox # Fallback

        # Get visible vertical limits (data units) for the viewport
        y_min_data, y_max_data = r_list[0].get_ylim()
        limit_min, limit_max = sorted([y_min_data, y_max_data])
        
        # Skip if BOTH points are completely outside the visible vertical range (on the same side)
        if (y_ref < limit_min and y_off < limit_min) or (y_ref > limit_max and y_off > limit_max):
            return

        # 1. Horizontal lines through Ref panels and gaps
        if limit_min <= y_ref <= limit_max:
            for i, ax in enumerate(r_list):
                ax.axhline(y=y_ref, color=color, **kwargs)
                if i < len(r_list) - 1:
                    ax_next = r_list[i+1]
                    con = ConnectionPatch(xyA=(1.0, y_ref), xyB=(0.0, y_ref),
                                          coordsA=ax.get_yaxis_transform(), coordsB=ax_next.get_yaxis_transform(),
                                          axesA=ax, axesB=ax_next,
                                          color=color, **kwargs)
                    con.set_clip_box(safe_bbox) # Robust absolute clipping
                    con.set_clip_on(True)
                    fig.add_artist(con)
        
        # 2. Diagonal line between wells
        if r_list and o_list:
            ax_r_last = r_list[-1]
            ax_o_first = o_list[0]
            con = ConnectionPatch(xyA=(1.0, y_ref), xyB=(0.0, y_off),
                                  coordsA=ax_r_last.get_yaxis_transform(), coordsB=ax_o_first.get_yaxis_transform(),
                                  axesA=ax_r_last, axesB=ax_o_first,
                                  color=color, **kwargs)
            con.set_clip_box(safe_bbox)
            con.set_clip_on(True)
            fig.add_artist(con)
            
        # 3. Horizontal lines through Off panels and gaps
        if limit_min <= y_off <= limit_max:
            for i, ax in enumerate(o_list):
                ax.axhline(y=y_off, color=color, **kwargs)
                if i < len(o_list) - 1:
                    ax_next = o_list[i+1]
                    con = ConnectionPatch(xyA=(1.0, y_off), xyB=(0.0, y_off),
                                          coordsA=ax.get_yaxis_transform(), coordsB=ax_next.get_yaxis_transform(),
                                          axesA=ax, axesB=ax_next,
                                          color=color, **kwargs)
                    con.set_clip_box(safe_bbox)
                    con.set_clip_on(True)
                    fig.add_artist(con)
        
            # 4. Label at the far right
            if label and o_list:
                ax_last = o_list[-1]
                t = ax_last.text(1.02, y_off, label, transform=ax_last.get_yaxis_transform(),
                                 color=color, fontsize=9, fontweight='normal', ha='left', va='center', zorder=100)
                t.set_clip_box(safe_bbox) # Prevents overlap with header/footer
                t.set_clip_on(True)

    def plot_geology_markers(self):
        """Plots GR logs and geological markers with high-quality styling."""
        self.fig_geo.clear()
        c = self.get_plot_colors()
        self.fig_geo.patch.set_facecolor(c['bg'])
        self.fig_geo.set_facecolor(c['bg'])
        self.canvas_geo.setStyleSheet(f"background-color: {c['bg']};")
        
        ref = self.ref_well_combo.currentText()
        off = self.off_well_combo.currentText()
        unit_str = "m" if self.unit_combo.currentText() == "Meters" else "ft"
        
        if not ref or not off: 
            self.canvas_geo.draw()
            return
        
        try:
            # 1. Load Data
            df1 = self.data_loader.load_las_data(ref)
            df2 = self.data_loader.load_las_data(off)
            m1 = self.geology_engine.get_markers(ref, self.data_loader.all_markers).copy()
            m2 = self.geology_engine.get_markers(off, self.data_loader.all_markers).copy()
            
            # Standardize Surface names to strings for reliable mapping
            m1['Surface'] = m1['Surface'].astype(str)
            m2['Surface'] = m2['Surface'].astype(str)
            
            # 2. Setup Persistent Color Palette
            all_surfs = sorted(list(set(m1['Surface'].tolist() + m2['Surface'].tolist())))
            cmap = matplotlib.cm.get_cmap('tab20', len(all_surfs)) if len(all_surfs) > 0 else None
            surf_colors = {str(surf): cmap(i) for i, surf in enumerate(all_surfs)} if cmap else {}
            
            # 3. Setup Figure Dimensions (Professional Thin & Tall)
            self.fig_geo.set_figheight(30) 
            self.fig_geo.set_figwidth(12)  
            
            # GridSpec: optimized for "Thin & Tall" tracks with plenty of room for legend
            # left=0.08, right=0.55 (leaves 45% for labels & legend)
            gs = self.fig_geo.add_gridspec(1, 2, wspace=1.2, top=0.96, bottom=0.03, left=0.08, right=0.55)
            ax1 = self.fig_geo.add_subplot(gs[0], label=ref)
            ax2 = self.fig_geo.add_subplot(gs[1], label=off)
            
            for i, (ax, df, well, markers) in enumerate(zip([ax1, ax2], [df1, df2], [ref, off], [m1, m2])):
                ax.set_facecolor(c['bg'])
                # MATCH LOG VIEWER: Dark Green for GR (#006400)
                curve_color = "#006400"
                plot_df = df.copy()
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    o_min, o_max = grid_overlap.min(), grid_overlap.max()
                    plot_df = plot_df[(plot_df['Depth'] >= o_min) & (plot_df['Depth'] <= o_max)]
                
                ax.plot(plot_df['GR'], plot_df['Depth'] * self.depth_multiplier, color=curve_color, lw=1.2, alpha=0.9, zorder=2)
                
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    ax.set_ylim(grid_overlap.max() * self.depth_multiplier, grid_overlap.min() * self.depth_multiplier)
                else:
                    ax.set_ylim(df['Depth'].max() * self.depth_multiplier, df['Depth'].min() * self.depth_multiplier)
                ax.set_title(well, fontweight='normal', color=c['text'], pad=12, fontsize=14, fontfamily='sans-serif')
                ax.set_xlabel("GR (API)", color=c['text'], fontsize=11, fontfamily='sans-serif', labelpad=5)
                ax.set_ylabel(f"Depth ({self.depth_unit})", color=c['text'], fontsize=11, fontfamily='sans-serif')
                
                # Dual Axis Formatting
                ax.tick_params(colors=c['text'], labelsize=10, which='both', direction='out', length=5)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                
                # Visual Grid
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--', lw=0.6)
                for spine in ax.spines.values():
                    spine.set_color(c['spine'])
                    spine.set_linewidth(1.5)
                
                # 4. Draw Markers
                y_limits = ax.get_ylim()
                l_min, l_max = sorted([y_limits[0], y_limits[1]])
                
                # Strict Filtering: Only draw markers within the overlap range if correlation exists
                plot_markers = markers.copy()
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    o_min, o_max = grid_overlap.min(), grid_overlap.max()
                    plot_markers = plot_markers[(plot_markers['MD'] >= o_min) & (plot_markers['MD'] <= o_max)]
                
                for _, row in plot_markers.iterrows():
                    surf = str(row['Surface'])
                    depth = float(row['MD']) * self.depth_multiplier
                    color = surf_colors.get(surf, 'yellow')
                    # Horizon line - always visible on individual wells
                    if l_min <= depth <= l_max:
                        ax.axhline(y=depth, color=color, ls='-', lw=2.5, alpha=1.0, zorder=10)
                    
                    # Labels on far right of the offset well
                    if i == 1:
                        from matplotlib.transforms import Bbox, TransformedBbox
                        try:
                            pos = ax.get_position()
                            v_clip_bbox = TransformedBbox(Bbox.from_extents(0, pos.y0, 1.0, pos.y1), self.fig_geo.transFigure)
                            t = ax.text(1.08, depth, surf, color=color, transform=ax.get_yaxis_transform(),
                                        verticalalignment='center', fontsize=10, fontweight='normal', 
                                        fontfamily='sans-serif', zorder=12)
                            t.set_clip_box(v_clip_bbox)
                            t.set_clip_on(True)
                        except: pass

            # 5. Draw Robust Connection Lines (Diagonal - Individual Tab Style)
            if self.show_geo_corr and not m1.empty and not m2.empty:
                common_surfaces = set(m1['Surface']).intersection(set(m2['Surface']))
                from matplotlib.patches import ConnectionPatch
                from matplotlib.transforms import Bbox, TransformedBbox
                
                # Clipping box for individual tab
                try:
                    pos1 = ax1.get_position()
                    tab_clip_bbox = TransformedBbox(Bbox.from_extents(0, pos1.y0, 1.0, pos1.y1), self.fig_geo.transFigure)
                except:
                    tab_clip_bbox = self.fig_geo.bbox

                for surf in common_surfaces:
                    d1_list = m1[m1['Surface'] == surf]['MD'].tolist()
                    d2_list = m2[m2['Surface'] == surf]['MD'].tolist()
                    
                    if d1_list and d2_list:
                        d1, d2 = float(d1_list[0]) * self.depth_multiplier, float(d2_list[0]) * self.depth_multiplier
                        color = surf_colors.get(surf, 'yellow')
                        
                        # Use standard diagonal connection for individual tabs as preferred
                        con = ConnectionPatch(xyA=(1.0, d1), xyB=(0.0, d2),
                                            coordsA=ax1.get_yaxis_transform(), coordsB=ax2.get_yaxis_transform(),
                                            axesA=ax1, axesB=ax2, 
                                            color=color, ls="-", lw=1.2, alpha=0.6, zorder=8)
                        con.set_clip_box(tab_clip_bbox) # Robust absolute clipping
                        con.set_clip_on(True) 
                        self.fig_geo.add_artist(con)
            
            # Force transform update to ensure everything stays in place
            self.fig_geo.canvas.draw_idle()
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=surf_colors[s], lw=5, label=s) for s in all_surfs]
            if legend_elements:
                # anchor at 1.8 means way out to the right of the axes
                leg = ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.9, 1.0), 
                                title="SEQUENCE MARKERS", fontsize=10, title_fontsize=11,
                                facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'],
                                ncol=1 if len(legend_elements) < 25 else 2) # Two columns if many markers
                if leg:
                    leg.get_title().set_fontweight('bold')
                    leg.get_title().set_color(c['text'])
                    leg.get_frame().set_linewidth(1.5)

            # Finalize
            self.draw_mgr_geo.redraw([ax1, ax2])
            self.canvas_geo.draw()
            # Force a second draw to ensure ConnectionPatch is correctly rendered and persistent
            self.canvas_geo.draw_idle() 
            
        except Exception as e:
            self.status_bar.showMessage(f"Critical Geology Error: {e}")
            import traceback
            traceback.print_exc()

    def init_sand_tab(self):
        """Initializes the Sand Plot Label tab."""
        layout = QVBoxLayout(self.sand_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        c = self.get_plot_colors()
        
        # 1. Header/Toolbar Row
        top_ctrl = QFrame()
        top_ctrl.setStyleSheet(f"background-color: {c['bg']}; border-bottom: 1px solid {c['spine']};")
        top_layout = QHBoxLayout(top_ctrl)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        self.canvas_sand = FigureCanvas(Figure(figsize=(12, 10), facecolor=c['bg']))
        self.fig_sand = self.canvas_sand.figure
        self.draw_mgr_sand = DrawingManager(self.canvas_sand, on_change=self.refresh_active_tab)
        
        self.toolbar_sand = VerticalToolbar(self.canvas_sand, self.sand_tab)
        self.toolbar_sand.setStyleSheet(f"background-color: transparent; border: none;")
        
        self.sand_hover_info = QLabel("Hover for info")
        self.sand_hover_info.setStyleSheet(f"color: {TURQUOISE}; font-weight: bold; font-family: 'Segoe UI', sans-serif;")
        
        self.sand_corr_btn = QPushButton("Show Sand Correlation")
        self.sand_corr_btn.setCheckable(True)
        self.sand_corr_btn.setChecked(True)
        self.sand_corr_btn.setFixedWidth(200)
        self.sand_corr_btn.setCursor(Qt.PointingHandCursor)
        self.sand_corr_btn.setStyleSheet(self.geo_corr_btn.styleSheet()) # Reuse style
        self.sand_corr_btn.clicked.connect(lambda: self.plot_sand_markers())
        
        top_layout.addWidget(self.toolbar_sand)
        top_layout.addStretch()
        top_layout.addWidget(self.sand_hover_info)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.sand_corr_btn)
        
        layout.addWidget(top_ctrl)
        
        # 2. Scroll Area for the tall plot
        self.scroll_sand = QScrollArea()
        self.scroll_sand.setWidgetResizable(True)
        self.scroll_sand.setFrameShape(QFrame.NoFrame)
        self.scroll_sand.setStyleSheet(f"background-color: {c['bg']};")
        
        self.scroll_sand.setWidget(self.canvas_sand)
        layout.addWidget(self.scroll_sand)
        
        self.canvas_sand.mpl_connect("motion_notify_event", self.on_sand_hover)

    def on_sand_hover(self, event):
        if event.inaxes is None: return
        try:
            well = event.inaxes.get_label()
            depth = event.ydata # Display-ready data coord
            self.sand_hover_info.setText(f"Well: {well} | Depth: {depth:.2f}{self.depth_unit}")
        except: pass

    def plot_sand_markers(self):
        """Plots GR logs and Sand Plot markers."""
        self.fig_sand.clear()
        c = self.get_plot_colors()
        self.fig_sand.patch.set_facecolor(c['bg'])
        self.fig_sand.set_facecolor(c['bg'])
        self.canvas_sand.setStyleSheet(f"background-color: {c['bg']};")
        
        ref = self.ref_well_combo.currentText()
        off = self.off_well_combo.currentText()
        unit_str = "m" if self.unit_combo.currentText() == "Meters" else "ft"
        
        if not ref or not off: 
            self.canvas_sand.draw()
            return
        
        try:
            df1 = self.data_loader.load_las_data(ref)
            df2 = self.data_loader.load_las_data(off)
            m1 = self.sand_engine.get_markers(ref, self.data_loader.sand_markers).copy()
            m2 = self.sand_engine.get_markers(off, self.data_loader.sand_markers).copy()
            
            # Standardize Surface names to strings
            m1['Surface'] = m1['Surface'].astype(str)
            m2['Surface'] = m2['Surface'].astype(str)
            
            all_surfs = sorted(list(set(m1['Surface'].tolist() + m2['Surface'].tolist())))
            cmap = matplotlib.cm.get_cmap('tab20', len(all_surfs)) if len(all_surfs) > 0 else None
            surf_colors = {str(surf): cmap(i) for i, surf in enumerate(all_surfs)} if cmap else {}
            
            self.fig_sand.set_figheight(30) 
            self.fig_sand.set_figwidth(12)  
            
            gs = self.fig_sand.add_gridspec(1, 2, wspace=1.2, top=0.96, bottom=0.03, left=0.08, right=0.55)
            ax1 = self.fig_sand.add_subplot(gs[0], label=ref)
            ax2 = self.fig_sand.add_subplot(gs[1], sharey=ax1, label=off)
            
            for i, (ax, df, well, markers) in enumerate(zip([ax1, ax2], [df1, df2], [ref, off], [m1, m2])):
                ax.set_facecolor(c['bg'])
                curve_color = "#006400"
                plot_df = df.copy()
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    o_min, o_max = grid_overlap.min(), grid_overlap.max()
                    plot_df = plot_df[(plot_df['Depth'] >= o_min) & (plot_df['Depth'] <= o_max)]
                
                ax.plot(plot_df['GR'], plot_df['Depth'] * self.depth_multiplier, color=curve_color, lw=1.2, alpha=0.9, zorder=2)
                
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    ax.set_ylim(grid_overlap.max() * self.depth_multiplier, grid_overlap.min() * self.depth_multiplier)
                else:
                    ax.set_ylim(df['Depth'].max() * self.depth_multiplier, df['Depth'].min() * self.depth_multiplier)
                ax.set_title(well, fontweight='normal', color=c['text'], pad=12, fontsize=14, fontfamily='sans-serif')
                ax.set_xlabel("GR (API)", color=c['text'], fontsize=11, fontfamily='sans-serif', labelpad=5)
                ax.set_ylabel(f"Depth ({self.depth_unit})", color=c['text'], fontsize=11, fontfamily='sans-serif')
                
                ax.tick_params(colors=c['text'], labelsize=10, which='both', direction='out', length=5)
                ax.xaxis.set_label_position('top')
                ax.xaxis.tick_top()
                
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--', lw=0.6)
                for spine in ax.spines.values():
                    spine.set_color(c['spine'])
                    spine.set_linewidth(1.5)
                
                # 4. Draw Markers
                y_limits = ax.get_ylim()
                l_min, l_max = sorted([y_limits[0], y_limits[1]])
                
                # Strict Filtering: Only draw markers within the overlap range if correlation exists
                plot_markers = markers.copy()
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    o_min, o_max = grid_overlap.min(), grid_overlap.max()
                    plot_markers = plot_markers[(plot_markers['MD'] >= o_min) & (plot_markers['MD'] <= o_max)]
                
                for _, row in plot_markers.iterrows():
                    surf = str(row['Surface'])
                    depth = float(row['MD']) * self.depth_multiplier
                    color = surf_colors.get(surf, 'yellow')
                    
                    # Horizon line - always visible on individual wells
                    if l_min <= depth <= l_max:
                        ax.axhline(y=depth, color=color, ls='-', lw=2.5, alpha=1.0, zorder=10)
                        
                    # Labels on far right of the offset well
                    if i == 1:
                        from matplotlib.transforms import Bbox, TransformedBbox
                        try:
                            pos = ax.get_position()
                            v_clip_bbox = TransformedBbox(Bbox.from_extents(0, pos.y0, 1.0, pos.y1), self.fig_sand.transFigure)
                            t = ax.text(1.08, depth, surf, color=color, transform=ax.get_yaxis_transform(),
                                        verticalalignment='center', fontsize=10, fontweight='normal', 
                                        fontfamily='sans-serif', zorder=12)
                            t.set_clip_box(v_clip_bbox)
                            t.set_clip_on(True)
                        except: pass

            # 5. Draw Robust Connection Lines (Diagonal - Individual Tab Style)
            if self.sand_corr_btn.isChecked() and not m1.empty and not m2.empty:
                common_surfaces = set(m1['Surface']).intersection(set(m2['Surface']))
                from matplotlib.patches import ConnectionPatch
                from matplotlib.transforms import Bbox, TransformedBbox
                
                # Clipping box for individual tab
                try:
                    pos1 = ax1.get_position()
                    tab_clip_bbox = TransformedBbox(Bbox.from_extents(0, pos1.y0, 1.0, pos1.y1), self.fig_sand.transFigure)
                except:
                    tab_clip_bbox = self.fig_sand.bbox

                for surf in common_surfaces:
                    d1_list = m1[m1['Surface'] == surf]['MD'].tolist()
                    d2_list = m2[m2['Surface'] == surf]['MD'].tolist()
                    if d1_list and d2_list:
                        d1, d2 = float(d1_list[0]) * self.depth_multiplier, float(d2_list[0]) * self.depth_multiplier
                        color = surf_colors.get(str(surf), 'gold') 
                        
                        # Use standard diagonal connection for individual tabs as preferred
                        con = ConnectionPatch(xyA=(1.0, d1), xyB=(0.0, d2),
                                            coordsA=ax1.get_yaxis_transform(), coordsB=ax2.get_yaxis_transform(),
                                            axesA=ax1, axesB=ax2, 
                                            color=color, ls="-", lw=1.2, alpha=0.6, zorder=8)
                        con.set_clip_box(tab_clip_bbox)
                        con.set_clip_on(True) 
                        self.fig_sand.add_artist(con)
            
            self.fig_sand.canvas.draw_idle()
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=surf_colors[s], lw=5, label=s) for s in all_surfs]
            if legend_elements:
                leg = ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.9, 1.0), 
                                title="TOP SAND MARKERS", fontsize=10, title_fontsize=11,
                                facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'],
                                ncol=1 if len(legend_elements) < 25 else 2)
                if leg:
                    leg.get_title().set_fontweight('bold')
                    leg.get_title().set_color(c['text'])
                    leg.get_frame().set_linewidth(1.5)

            self.draw_mgr_sand.redraw([ax1, ax2])
            self.canvas_sand.draw()
            self.canvas_sand.draw_idle() 
            
        except Exception as e:
            self.status_bar.showMessage(f"Critical Sand Plot Error: {e}")
            import traceback
            traceback.print_exc()
        
    def update_mrgc_features(self):
        """Updates the MRGC feature lists based on currently selected wells in the main sidebar."""
        w1 = self.ref_well_combo.currentText()
        w2 = self.off_well_combo.currentText()
        self.mrgc_rec_label.setText("Recommended: --")
        
        for idx, well_name in enumerate([w1, w2]):
            layout = self.mrgc_w1_feat_layout if idx == 0 else self.mrgc_w2_feat_layout
            checks = self.mrgc_w1_checks if idx == 0 else self.mrgc_w2_checks
            
            # Clear layout
            while layout.count():
                item = layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            checks.clear()
            
            if not well_name: continue
            
            try:
                curves = self.data_loader.get_available_curves(well_name)
                # Auto-select common Petrophysical curves
                common = ['GR', 'RHOB', 'NPHI', 'DT', 'VSH', 'ILD', 'RT']
                for c in curves:
                    if any(x in c.upper() for x in ['DEPT', 'DEPTH']): continue
                    cb = QCheckBox(c)
                    cb.setStyleSheet(f"color: {SILVER};")
                    if any(x in c.upper() for x in common): cb.setChecked(True)
                    layout.addWidget(cb)
                    checks.append(cb)
                layout.addStretch()
            except Exception as e:
                print(f"MRGC Error loading curves: {e}")

    def toggle_mrgc_all(self, well_num):
        checks = self.mrgc_w1_checks if well_num == 1 else self.mrgc_w2_checks
        if not checks: return
        state = not all(cb.isChecked() for cb in checks)
        for cb in checks: cb.setChecked(state)

    def suggest_mrgc_alpha(self):
        w1 = self.ref_well_combo.currentText()
        if not w1: return
        self.status_bar.showMessage(f"Calculating recommended alpha for {w1}...")
        try:
            feats = [cb.text() for cb in self.mrgc_w1_checks if cb.isChecked()]
            rec = self.mrgc_analyzer.suggest_alpha(w1, feats)
            self.mrgc_alpha_spin.setValue(rec)
            self.mrgc_rec_label.setText(f"Recommended: {rec:.1f}")
            self.status_bar.showMessage("Recommended Alpha applied.", 3000)
        except Exception as e:
            self.mrgc_rec_label.setText("Recommended: Error")
            print(f"DEBUG: MRGC Alpha suggestion failed: {e}")

    def run_mrgc_analysis(self):
        if self.rb_route_map.isChecked():
            QMessageBox.warning(self, "Route Mode Active", 
                                "MRGC Analysis is disabled while in Route Plotter mode.")
            return

        w1 = self.ref_well_combo.currentText()
        w2 = self.off_well_combo.currentText()
        if w1 == w2:
            QMessageBox.warning(self, "Selection Error", "Please select two different wells for side-by-side analysis.")
            return
            
        f1 = [cb.text() for cb in self.mrgc_w1_checks if cb.isChecked()]
        f2 = [cb.text() for cb in self.mrgc_w2_checks if cb.isChecked()]
        
        if not f1 or not f2:
            QMessageBox.warning(self, "Selection", "Please select feature logs for both wells.")
            return
            
        # Use intersection of features for combined clustering to ensure consistency
        shared_features = list(set(f1) & set(f2))
        if not shared_features:
            QMessageBox.warning(self, "Feature Mismatch", "No shared feature logs selected. Combined clustering requires common logs between wells.")
            return
            
        self.mrgc_run_btn.setEnabled(False)
        self.unit_combo.setEnabled(False)
        self.mrgc_pbar.show()
        self.mrgc_pbar.setRange(0, 0)
        self.mrgc_results = {}
        
        n_c = None if self.mrgc_clusters_combo.currentText() == "Auto" else int(self.mrgc_clusters_combo.currentText())
        unit = "m" if self.unit_combo.currentText() == "Meters" else "ft"
        params = {'alpha': self.mrgc_alpha_spin.value(), 'n_clusters': n_c, 'unit': unit, 'feature_curves': shared_features}
        
        self.mrgc_thread = MRGCClusteringThread(self.mrgc_analyzer, [w1, w2], params)
        self.mrgc_thread.finished.connect(self.on_combined_mrgc_fin)
        self.mrgc_thread.error.connect(self.on_mrgc_error)
        self.mrgc_thread.start()

    def on_mrgc_error(self, err):
        QMessageBox.critical(self, "Analysis Error", err)
        self.mrgc_run_btn.setEnabled(True)
        self.unit_combo.setEnabled(True)
        self.mrgc_pbar.hide()

    def on_combined_mrgc_fin(self, results):
        self.mrgc_results = results
        try:
            self.plot_mrgc_results()
            self.mrgc_view_btn.setEnabled(True)
            self.status_bar.showMessage("Combined MRGC Analysis Complete.")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Plot Error", f"Failed to plot MRGC results: {e}\n{traceback.format_exc()}")
        self.mrgc_run_btn.setEnabled(True)
        self.unit_combo.setEnabled(True)
        self.mrgc_pbar.hide()

    def plot_mrgc_results(self):
        """Plot MRGC clustering results with comprehensive error handling"""
        try:
            if not self.mrgc_results: return
            self.canvas_mrgc.figure.clear()
            c = self.get_plot_colors()
            self.canvas_mrgc.figure.set_facecolor(c['bg'])
            
            w1, w2 = self.ref_well_combo.currentText(), self.off_well_combo.currentText()
            if w1 not in self.mrgc_results or w2 not in self.mrgc_results:
                self.canvas_mrgc.figure.text(0.5, 0.5, "Analysis required for selected wells.\\nPlease run MRGC Analysis.", 
                                             ha='center', va='center', color='red', fontsize=12)
                self.canvas_mrgc.draw()
                return

            d1, d2 = self.mrgc_results[w1], self.mrgc_results[w2]
            
            # Verify required columns exist
            required_cols = ['Depth', 'NI', 'KRI', 'Cluster']
            for well_name, data in [(w1, d1), (w2, d2)]:
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    self.canvas_mrgc.figure.text(0.5, 0.5, 
                                                f"Error: Missing columns in {well_name}: {missing_cols}\\nPlease re-run MRGC Analysis.", 
                                                ha='center', va='center', color='red', fontsize=12)
                    self.canvas_mrgc.draw()
                    return
            
            self.mrgc_axes_to_well = {}
            
            # Calculate dynamic width ratios based on 'Panel Width' slider
            ratio = self.ratio_slider.value()
            # Each well has 3 tracks. We split the 100% total according to the ratio.
            # r1 is the weight for each track of Well 1, r2 for Well 2.
            r1, r2 = ratio / 3.0, (100 - ratio) / 3.0
            w_ratios = [r1, r1, r1, r2, r2, r2]
            
            gs = gridspec.GridSpec(1, 6, width_ratios=w_ratios, wspace=0.1, left=0.06, right=0.98, top=0.94, bottom=0.08)
            # Use colormap properly
            cmap_obj = plt.get_cmap('tab20')
            cmap_colors = cmap_obj(np.linspace(0, 1, 20))
            all_facies_axes = []
            master_ax = None
            
            # Calculate global depth limits for synchronization
            # Use 'Depth' (not 'DEPTH') to match engine output
            all_min = min(d1['Depth'].min(), d2['Depth'].min()) * self.depth_multiplier
            all_max = max(d1['Depth'].max(), d2['Depth'].max()) * self.depth_multiplier
            
            for i, (name, data) in enumerate([(w1, d1), (w2, d2)]):
                base = i * 3; depth = data['Depth'].values * self.depth_multiplier
                
                # NI Track
                ax_ni = self.canvas_mrgc.figure.add_subplot(gs[base], facecolor=c['bg'], sharey=master_ax)
                if master_ax is None: master_ax = ax_ni
                ax_ni.plot(data['NI'], depth, color=c['text'], lw=0.8)
                ax_ni.set_title(f"NI - {name}", color=c['text'], size=7, weight='normal')
                ax_ni.grid(True, alpha=0.1, color=c['grid'], ls=':')
                self.mrgc_axes_to_well[ax_ni] = name
                
                # KRI Track
                ax_kri = self.canvas_mrgc.figure.add_subplot(gs[base+1], facecolor=c['bg'], sharey=master_ax)
                ax_kri.plot(data['KRI'], depth, color='#FF4500', lw=0.8)
                ax_kri.set_title(f"KRI - {name}", color='#FF4500', size=7, weight='normal')
                ax_kri.grid(True, alpha=0.1, color=c['grid'], ls=':')
                self.mrgc_axes_to_well[ax_kri] = name
                
                # Facies Track
                ax_fac = self.canvas_mrgc.figure.add_subplot(gs[base+2], facecolor=c['bg'], sharey=master_ax)
                ax_fac.set_label(f"mrgc_log_cluster_{name}") # For drawing manager
                
                # Robust Facies Plotting - Use 'Depth' not 'DEPTH'
                df_sorted = data.sort_values('Depth')
                
                # Strict Filtering: Filter data to INPEFA overlap if correlation exists
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    o_min, o_max = grid_overlap.min(), grid_overlap.max()
                    df_sorted = df_sorted[(df_sorted['Depth'] >= o_min) & (df_sorted['Depth'] <= o_max)]

                depths_plot = df_sorted['Depth'].values
                clusters_plot = df_sorted['Cluster'].values
                
                if len(depths_plot) > 0:
                    current_c = clusters_plot[0]
                    start_idx = 0
                    for idx_p in range(1, len(clusters_plot)):
                        if clusters_plot[idx_p] != current_c:
                            ax_fac.fill_betweenx(depths_plot[start_idx:idx_p+1] * self.depth_multiplier, 0, 1, 
                                              color=cmap_colors[int(current_c)%20], alpha=0.9)
                            current_c = clusters_plot[idx_p]
                            start_idx = idx_p
                    ax_fac.fill_betweenx(depths_plot[start_idx:] * self.depth_multiplier, 0, 1, 
                                      color=cmap_colors[int(current_c)%20], alpha=0.9)
                
                for cid in np.unique(clusters_plot):
                    ax_fac.plot([], [], color=cmap_colors[int(cid)%20], label=f"Clust {cid}", linewidth=10)

                ax_fac.set_title(f"LOG CLUSTER - {name}", color=c['text'], size=7, weight='normal')
                ax_fac.set_xticks([])
                all_facies_axes.append(ax_fac); self.mrgc_axes_to_well[ax_fac] = name
                
                for ax in [ax_ni, ax_kri, ax_fac]:
                    ax.tick_params(axis='both', colors=c['text'], labelsize=6.5)
                    for s in ax.spines.values(): s.set_color(c['spine'])
                    
                    # Show labels only for the first track of the first well
                    if base == 0 and ax == ax_ni:
                        ax.tick_params(labelleft=True)
                        ax.set_ylabel(f"Depth ({self.depth_unit})", color=c['text'], size=7)
                    else:
                        ax.tick_params(labelleft=False)
            
            # Set global limits once on master axis
            # Set global limits once on master axis
            if master_ax:
                if hasattr(self, 'last_plot_args') and self.last_plot_args is not None:
                    grid_overlap = self.last_plot_args[0]
                    master_ax.set_ylim(grid_overlap.max() * self.depth_multiplier, grid_overlap.min() * self.depth_multiplier)
                else:
                    master_ax.set_ylim(all_max, all_min)

            # Legend
            handles, labels = [], []
            for ax in all_facies_axes:
                h, l = ax.get_legend_handles_labels()
                for hi, li in zip(h, l):
                    if li not in labels: handles.append(hi); labels.append(li)
            if labels:
                comb = sorted(zip(labels, handles), key=lambda x: int(x[0].split()[-1]))
                ls, hs = zip(*comb)
                self.canvas_mrgc.figure.legend(hs, ls, loc='lower center', ncol=min(10, len(ls)), 
                                        facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'], fontsize=7, frameon=True)

            self.canvas_mrgc.figure.suptitle("MRGC ANALYSIS: LOG DATA CLUSTERING", color=c['text'], weight='normal', size=11)
            
            # Redraw sketches if any
            self.draw_mgr_mrgc.redraw(all_facies_axes + list(self.mrgc_axes_to_well.keys()))
            self.canvas_mrgc.draw()
        
        except Exception as e:
            import traceback
            print(f"Error in plot_mrgc_results: {e}")
            print(traceback.format_exc())
            self.canvas_mrgc.figure.clear()
            self.canvas_mrgc.figure.text(0.5, 0.5, f"Error plotting MRGC results:\\n{str(e)}", 
                                        ha='center', va='center', color='red', fontsize=10)
            self.canvas_mrgc.draw()

    def on_mrgc_hover(self, event):
        if not event.inaxes or not self.mrgc_results: return
        well = self.mrgc_axes_to_well.get(event.inaxes)
        if not well: return
        try:
            d = self.mrgc_results[well]
            idx = np.abs(d['Depth'].values * self.depth_multiplier - event.ydata).argmin()
            depth = d['Depth'].values[idx] * self.depth_multiplier
            cluster = d['Cluster'].values[idx]
            
            # Dynamic Label Color matching the cluster
            cmap_obj = plt.get_cmap('tab20')
            rgb = cmap_obj(int(cluster)%20)
            hex_color = matplotlib.colors.to_hex(rgb)
            
            self.mrgc_hover_info.setText(f"Well: {well} | Depth: {depth:.2f}{self.depth_unit} | Cluster: {cluster}")
            self.mrgc_hover_info.setStyleSheet(f"color: {hex_color}; font-weight: bold; font-size: 11px;")
        except: pass

    def open_mrgc_viewer(self):
        try:
            if not self.mrgc_results: return
            viewer = ClusterDataViewer(self.mrgc_results, self)
            viewer.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Viewer Error", f"Could not open data viewer: {e}")
            print(f"DEBUG: Viewer Error: {e}")

    def update_available_logs(self):
        ref = self.ref_well_combo.currentText()
        off = self.off_well_combo.currentText()
        
        # Get curves
        c1 = self.data_loader.get_available_curves(ref)
        c2 = self.data_loader.get_available_curves(off)
        
        self.ref_curve_list.clear()
        self.off_curve_list.clear()
        
        for c in c1:
            item = QListWidgetItem(c)
            self.ref_curve_list.addItem(item)
        for c in c2:
            item = QListWidgetItem(c)
            self.off_curve_list.addItem(item)
            
        self.log_ref_lbl.setText(f"Ref Well Logs: {ref}")
        self.log_off_lbl.setText(f"Off Well Logs: {off}")
        self.status_bar.showMessage(f"Logs loaded for {ref} and {off}")

    def validate_selection_limit(self):
        """Enforces a maximum of 4 selected curves per well."""
        sender = self.sender()
        if not isinstance(sender, QListWidget): return
        
        selected = sender.selectedItems()
        if len(selected) > 4:
            sender.blockSignals(True)
            # Keep only the first 4 selected items, deselect the rest
            for i in range(4, len(selected)):
                selected[i].setSelected(False)
            sender.blockSignals(False)
            self.status_bar.showMessage("Selection Limit: Maximum 4 curves per well!", 3000)

    def get_curve_style(self, curve_name):
        """Returns (color, linestyle, unit) based on curve mnemonic using industry standards."""
        cn = curve_name.upper()
        
        # Gamma Ray
        if any(x in cn for x in ['GR', 'GAMMA']):
            return '#006400', '-', 'API' # Dark Green
            
        # Resistivity Logs
        if any(x in cn for x in ['LLS', 'SFL', 'SHAL', 'SN', 'MSFL']):
            return '#FF8C00', '-', 'ohm.m' # Shallow - Dark Orange
        if any(x in cn for x in ['ILM', 'LLM', 'MED', 'LN']):
            return '#4169E1', '-', 'ohm.m' # Medium - Royal Blue
        if any(x in cn for x in ['ILD', 'LLD', 'RT', 'DEEP']):
            return '#FF0000', '-', 'ohm.m' # Deep - Bright Red
            
        # Density Log (RHOB)
        if 'RHOB' in cn or 'DENS' in cn:
            return '#B22222', '-', 'g/cm3' # Firebrick (Distinct from Red)
            
        # Neutron Log (NPHI)
        if 'NPHI' in cn or 'NEUT' in cn:
            return '#1E90FF', '-', 'v/v' # Dodger Blue (Distinct from Royal Blue)
            
        # Sonic/DT Log
        if 'DT' in cn or 'SONI' in cn:
            return '#800080', '-', 'us/ft' # Purple
            
        # Caliper Log
        if 'CAL' in cn:
            return '#8B4513', '--', 'in' # Brown Dash
            
        # Spontaneous Potential
        if 'SP' in cn:
            return '#000000', '-', 'mV' # Black
            
        # Photoelectric (PEF)
        if 'PEF' in cn:
            return '#FF00FF', '-', 'b/e' # Magenta
            
        # Default
        return '#555555', '-', '-' # Grey

    def plot_logs(self):
        """Plots logs for both wells in the Log Viewer tab."""
        ref = self.ref_well_combo.currentText()
        off = self.off_well_combo.currentText()
        
        c1_items = [item.text() for item in self.ref_curve_list.selectedItems()]
        c2_items = [item.text() for item in self.off_curve_list.selectedItems()]
        
        if not (c1_items or c2_items):
            self.status_bar.showMessage("Please select at least one curve from the lists.")
            return
            
        try:
            unit_str = "m" if self.unit_combo.currentText() == "Meters" else "ft"
            mode = self.plot_mode_combo.currentText()
            c = self.get_plot_colors()
            self.fig_log.clf()
            self.fig_log.set_facecolor(c['bg'])
            
            # Setup Layout: 2 Fixed Panels (Reference | Offset)
            # Accommodate up to 4 headers (approx 0.06 height each -> 0.24 total)
            top_margin = 0.79 if mode == "Normal Log Curve" else 0.92
            self.fig_log.subplots_adjust(top=top_margin, bottom=0.08, left=0.12, right=0.88, wspace=0.2)
            
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
            ax1 = self.fig_log.add_subplot(gs[0], label="ax_left", facecolor=c['bg'])
            ax2 = self.fig_log.add_subplot(gs[1], sharey=ax1, label="ax_right", facecolor=c['bg'])
            
            # Calculate common depth range for all selected curves
            all_depths = []
            for w, curves in [(ref, c1_items), (off, c2_items)]:
                for c_name in curves:
                    try:
                        df_tmp = self.data_loader.load_curve_data(w, c_name)
                        if not df_tmp.empty:
                            all_depths.extend([df_tmp["Depth"].min(), df_tmp["Depth"].max()])
                    except: continue
            
            curr_y_min = (min(all_depths) if all_depths else 0) * self.depth_multiplier
            curr_y_max = (max(all_depths) if all_depths else 1000) * self.depth_multiplier
            ax1.set_ylim(curr_y_max, curr_y_min) 
            
            # Shared Axis Styling
            for ax in [ax1, ax2]:
                ax.tick_params(colors=c['text'], labelsize=8)
                ax.xaxis.label.set_color(c['text'])
                ax.yaxis.label.set_color(c['text'])
                ax.title.set_color(c['text'])
                ax.grid(True, alpha=0.2, color=c['grid'])
                for spine in ax.spines.values(): spine.set_color(c['spine'])

            # --- MODE: Lithology Analysis (Expert Labelling) ---
            if mode == "Lithology Analysis":
                for ax, well_name, curves in zip([ax1, ax2], [ref, off], [c1_items, c2_items]):
                    curve_name = curves[0] if curves else ""
                    if not curve_name or not any(x in curve_name.upper() for x in ['GR', 'GAMMA']):
                        ax.text(0.5, 0.5, "Requires GR log", ha='center', va='center', transform=ax.transAxes, color='red')
                        continue

                    df = self.data_loader.load_curve_data(well_name, curve_name)
                    res_df = None
                    try:
                        avail = self.data_loader.get_available_curves(well_name)
                        res_n = next((cn for cn in avail if any(x in cn.upper() for x in ['ILD', 'LLD', 'RT', 'DEEP'])), None)
                        if res_n: res_df = self.data_loader.load_curve_data(well_name, res_n)
                    except: pass

                    depths, gr = df["Depth"].values, df["Value"].values
                    if res_df is not None and not res_df.empty:
                        res = interp1d(res_df["Depth"].values, res_df["Value"].values, bounds_error=False, fill_value=np.nan)(depths)
                    else:
                        res = np.full(len(depths), np.nan)

                    idx = self.log_plot_engine.classify_lithology(gr, res)

                    cmap_lith = ListedColormap(["#000000", "#FFFF00", "#BDB76B", "#8B4513"])
                    data = np.array(idx).reshape(-1, 1)
                    ax.imshow(data, aspect='auto', cmap=cmap_lith, vmin=0, vmax=3, 
                                 extent=[0, 1, depths.max() * self.depth_multiplier, depths.min() * self.depth_multiplier], interpolation='nearest')
                    ax.set_title(f"{well_name} (Lith)", fontweight='normal', pad=10)
                    ax.set_xticks([]); ax.set_xlabel("Lithology")
                    ax.grid(False)

                patches = [
                    Patch(facecolor="#ffd700", edgecolor='k', label='Shaly Sand'), 
                    Patch(facecolor="#8B4513", edgecolor='k', label='Shale')
                ]
                ax1.set_ylabel(f"Depth ({unit_str})")
                for ax in [ax1, ax2]:
                    leg = ax.legend(handles=patches, loc='upper right', fontsize=8, facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'])

            # --- MODE: Vshale Analysis (GR) ---
            elif mode == "Vshale Analysis (GR)":
                for ax, curves, well_name in zip([ax1, ax2], [c1_items, c2_items], [ref, off]):
                    # Priority 1: Load pre-calculated VSH from 'grn vsh'
                    vsh_loaded = self.data_loader.load_vshale_data(well_name)
                    
                    if not vsh_loaded.empty:
                        ax.plot(vsh_loaded['Vsh'], vsh_loaded["Depth"] * self.depth_multiplier, label="VSH (Loaded)", color="#7FFF00", linewidth=1.5, zorder=5)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Vshale (v/v)")
                        ax.set_title(f"{well_name} (Loaded Vsh)", fontweight='normal', pad=10)
                        ax.legend(loc='upper right', fontsize=7, facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'])
                        continue

                    # Fallback to manual calculation if no loaded data
                    c_name = curves[0] if curves else ""
                    if not c_name: 
                        ax.text(0.5, 0.5, "No GR curve for calc", ha='center', va='center', transform=ax.transAxes)
                        continue
                    
                    df = self.data_loader.load_curve_data(well_name, c_name)
                    if df.empty: continue
                    
                    gr = df["Value"].values
                    vsh = self.log_plot_engine.calculate_vshale(gr)
                    
                    # ax.plot(vsh['ish'], df["Depth"] * self.depth_multiplier, label="Linear", color="#00CED1", linewidth=1.2)
                    ax.plot(vsh['clavier'], df["Depth"] * self.depth_multiplier, label="Clavier", color="#FFD700", linewidth=1.2)
                    ax.plot(vsh['steiber'], df["Depth"] * self.depth_multiplier, label="Steiber", color="#FF4500", linewidth=1.2)
                    
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Vshale (v/v)")
                    ax.set_title(f"{well_name} (Calc Vsh)", fontweight='normal', pad=10)
                    ax.legend(loc='upper right', fontsize=7, facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'])

                ax1.set_ylabel(f"Depth ({self.depth_unit})")

            # --- MODE: Normal Log Curve (Multi-Curve) ---
            else:
                for ax, curves, well_name in zip([ax1, ax2], [c1_items, c2_items], [ref, off]):
                    if not curves: 
                        ax.text(0.5, 0.5, "No curves selected", ha='center', va='center', transform=ax.transAxes, color='gray')
                        continue
                    
                    ax.curve_info = [] 
                    sorted_cs = sorted(curves, key=lambda x: 0 if any(m in x.upper() for m in ['GR', 'GAMMA']) else 1)
                    
                    for i, c_name in enumerate(sorted_cs):
                        try:
                            df = self.data_loader.load_curve_data(well_name, c_name)
                            if df.empty: continue
                            
                            color, style, unit = self.get_curve_style(c_name)
                            v_min, v_max = df["Value"].min(), df["Value"].max()
                            norm_vals = (df["Value"] - v_min) / (v_max - v_min) if v_max > v_min else np.full(len(df["Value"]), 0.5)
                                
                            ax.plot(norm_vals, df["Depth"] * self.depth_multiplier, color=color, linestyle=style, linewidth=1.1, zorder=3, label=c_name)
                            ax.curve_info.append({'name': c_name, 'min': v_min, 'max': v_max, 'unit': unit})
                            
                            # COMPACT HEADERS (User Request)
                            # Start low (1.005) and step small (0.055) to fit 4 headers in ~0.24 space
                            h_pos = 1.005 + (i * 0.055)
                            
                            # Min Value (Left)
                            ax.text(0.01, h_pos, f"{v_min:.1f}", transform=ax.transAxes, ha='left', va='bottom', 
                                    color=c['text'], fontsize=7, fontfamily='sans-serif')
                            
                            # Name & Unit (Center) - Bold
                            ax.text(0.5, h_pos, f"{c_name}", transform=ax.transAxes, ha='center', va='bottom', 
                                    color=color, fontweight='normal', fontsize=8, fontfamily='sans-serif')
                                    
                            # Max Value (Right)
                            ax.text(0.99, h_pos, f"{v_max:.1f}", transform=ax.transAxes, ha='right', va='bottom', 
                                    color=c['text'], fontsize=7, fontfamily='sans-serif')
                                    
                            # Divider Line (Subtle)
                            ax.plot([0.0, 1.0], [h_pos-0.002, h_pos-0.002], color=color, lw=0.8, alpha=0.6, transform=ax.transAxes, clip_on=False)
                            
                        except: continue

                    ax.set_xlim(0, 1); ax.set_xticks([])
                    # Removed Well Title per user request -> "nama sumur di atas panel gausha ada gapapa"
                    
                    
                    def log_formatter(x, y, ax_obj=ax):
                        if not hasattr(ax_obj, 'curve_info'): return f"Depth: {y:.1f}"
                        res = [f"Depth: {y:.1f} {self.depth_unit}"]
                        for info in ax_obj.curve_info:
                            val = info['min'] + x * (info['max'] - info['min'])
                            res.append(f"{info['name']}: {val:.2f}")
                        return " | ".join(res)
                    ax.format_coord = log_formatter

                ax1.set_ylabel(f"Depth ({self.depth_unit})")
            
            self.draw_mgr_log.redraw([ax1, ax2])
            self.canvas_log.draw()
            self.status_bar.showMessage(f"Logs plotted in {mode} mode.")
            
        except Exception as e:
            import traceback; traceback.print_exc()
            self.status_bar.showMessage(f"Plot Error: {str(e)}")

    def draw_empty_tracks(self):
        self.fig_corr.clf()
        c = self.get_plot_colors()
        self.fig_corr.set_facecolor(c['bg'])
        
        ratio = self.ratio_slider.value()
        gs = gridspec.GridSpec(1, 2, width_ratios=[ratio, 100-ratio], wspace=0.1)
        
        self.ax_ref = self.fig_corr.add_subplot(gs[0], label="corr_left")
        self.ax_off = self.fig_corr.add_subplot(gs[1], sharey=self.ax_ref, label="corr_right")
        
        # Apply Styling
        for ax in [self.ax_ref, self.ax_off]:
            ax.set_facecolor(c['bg'])
            ax.tick_params(colors=c['text'])
            ax.xaxis.label.set_color(c['text'])
            ax.yaxis.label.set_color(c['text'])
            ax.title.set_color(c['text'])
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_color(c['spine'])
                spine.set_linewidth(1.0)
                
        self.ax_ref.set_title("Reference Well (Select above)", color=c['text'])
        self.ax_off.set_title("Offset Well (Select above)", color=c['text'])
        
        # Redraw sketches
        self.draw_mgr_corr.redraw([self.ax_ref, self.ax_off])
        
        self.canvas_corr.draw()

    def run_analysis(self, switch_tab=True):
        ref_name = self.ref_well_combo.currentText()
        off_name = self.off_well_combo.currentText()
        term_text = self.inpefa_combo.currentText().split()[0].lower()
        self._switch_tab_after_corr = switch_tab
        
        if self.rb_route_map.isChecked():
            QMessageBox.warning(self, "Route Mode Active", 
                                "Analysis features are disabled while in Route Plotter mode.\n"
                                "Please switch back to 'Chosen Well Maps' to perform correlation.")
            return

        self.status_bar.showMessage(f"Processing {ref_name} ↔ {off_name} ({term_text} term) in background...")
        self.run_btn.setEnabled(False)
        
        if self.rb_sand_dtw.isChecked():
            mode = "sand"
        elif self.rb_geo_dtw.isChecked():
            mode = "geo"
        else:
            mode = "full"

        # Start background thread
        self.corr_thread = CorrelationThread(
            ref_name, off_name, term_text, mode,
            self.data_loader, AnalysisEngine, self.depth_multiplier
        )
        self.corr_thread.finished.connect(self.on_correlation_fin)
        self.corr_thread.error.connect(self.on_correlation_error)
        self.corr_thread.start()

    def on_correlation_error(self, err):
        self.status_bar.showMessage(f"Error: {err}")
        QMessageBox.critical(self, "Correlation Error", f"Failed to compute correlation: {err}")
        self.run_btn.setEnabled(True)

    def on_correlation_fin(self, res):
        try:
            # 2. Plotting
            self.plot_correlation(
                res['grid'], res['log_A'], res['log_B'], 
                res['pi'], res['pj'], res['ref_name'], res['off_name']
            )
            
            # 3. Store for export and global use
            self.current_export_data = pd.DataFrame({
                "Reference_Well": res['ref_name'],
                "Offset_Well": res['off_name'],
                "Reference_Depth": res['grid'][res['pi']],
                "Offset_Depth": res['grid'][res['pj']],
                "Reference_INPEFA": res['log_A'][res['pi']],
                "Offset_INPEFA": res['log_B'][res['pj']]
            })
            self.export_btn.setEnabled(True)
            self.last_plot_args = (res['grid'], res['log_A'], res['log_B'], res['pi'], res['pj'], res['ref_name'], res['off_name'])
            
            self.status_bar.showMessage(f"Success! Cost: {res['cost']:.2f}. Overlap: {res['grid'][0] * self.depth_multiplier:.0f}{self.depth_unit} - {res['grid'][-1] * self.depth_multiplier:.0f}{self.depth_unit}")
            
            if self._switch_tab_after_corr:
                self.tabs.setCurrentIndex(1)
                
        except Exception as e:
            import traceback; traceback.print_exc()
            self.status_bar.showMessage(f"Plotting failure: {str(e)}")
        
        self.run_btn.setEnabled(True)

    def export_csv(self):
        if not hasattr(self, 'current_export_data') or self.current_export_data is None:
            return
            
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Correlation CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.current_export_data.to_csv(path, index=False)
                self.status_bar.showMessage(f"Exported to {os.path.basename(path)}")
            except Exception as e:
                self.status_bar.showMessage(f"Export Failed: {str(e)}")

    def update_plot_layout(self):
        # Re-plot if data exists, otherwise redraw empty tracks with new ratio
        if hasattr(self, 'last_plot_args'):
            self.plot_correlation(*self.last_plot_args)
        else:
            self.draw_empty_tracks()
            
        # Also refresh MRGC if results exist to respect 'Panel Width'
        if self.mrgc_results:
            self.plot_mrgc_results()

    def plot_correlation(self, grid, log_A, log_B, pi, pj, ref_name, off_name):
        from matplotlib.patches import ConnectionPatch
        unit_str = "m" if self.unit_combo.currentText() == "Meters" else "ft"
        
        # Store args for re-plotting on resize
        self.last_plot_args = (grid, log_A, log_B, pi, pj, ref_name, off_name)
        
        self.fig_corr.clf() # Clear entire figure to reset gridspec
        c = self.get_plot_colors()
        self.fig_corr.set_facecolor(c['bg'])
        
        # GridSpec for resizing
        ratio = self.ratio_slider.value()
        gs = gridspec.GridSpec(1, 2, width_ratios=[ratio, 100-ratio], wspace=0.1)
        
        self.ax_ref = self.fig_corr.add_subplot(gs[0], label="corr_left")
        self.ax_off = self.fig_corr.add_subplot(gs[1], sharey=self.ax_ref, label="corr_right")
        
        # Apply Styling (Moved from init)
        for ax in [self.ax_ref, self.ax_off]:
            ax.set_facecolor(c['bg'])
            ax.tick_params(colors=c['text'])
            ax.xaxis.label.set_color(c['text'])
            ax.yaxis.label.set_color(c['text'])
            ax.title.set_color(c['text'])
            # ax.invert_yaxis() # Handle individually
            for spine in ax.spines.values():
                spine.set_color(c['spine'])
                spine.set_linewidth(1.0)
        
        # Normalize INPEFA values for gradient coloring
        comb_val = np.concatenate([log_A, log_B])
        vmin, vmax = np.nanmin(comb_val), np.nanmax(comb_val)
        
        # Plot with Gradient Fill
        for ax, log_data, name, orientation in zip([self.ax_ref, self.ax_off], [log_A, log_B], [ref_name, off_name], ["Reference", "Offset"]):
            ax.plot(log_data, grid * self.depth_multiplier, color="black", linewidth=1.1, zorder=5) # Curves now black as requested
            
            # Create Gradient Fill using imshow clipped by fill_betweenx
            # We create a horizontal gradient [vmin, vmax]
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            im = ax.imshow(grad, aspect='auto', cmap='coolwarm', 
                          extent=[vmin, vmax, grid.max() * self.depth_multiplier, grid.min() * self.depth_multiplier],
                          alpha=0.8, zorder=1)
            
            # Use fill_betweenx to create the clipping polygon
            path = ax.fill_betweenx(grid * self.depth_multiplier, log_data, 0, color='none', zorder=2)
            # Clip using the paths (handle potential multiple segments if NaNs exist)
            from matplotlib.path import Path
            combined_path = Path.make_compound_path(*path.get_paths())
            im.set_clip_path(combined_path, transform=ax.transData)
            
            ax.set_title(f"{orientation} Well: {name}", fontweight='normal', fontsize=11, color=c['text'])
            ax.grid(True, alpha=0.3, color=c['grid'], linestyle='--')
            ax.set_xlabel("INPEFA Value", fontweight='normal', color=c['text'])
            if orientation == "Reference":
                ax.set_ylabel(f"Depth ({self.depth_unit})", fontweight='normal', color=c['text'])
        
        # Add a shared colorbar for INPEFA values
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb_ax = self.fig_corr.add_axes([0.45, 0.02, 0.1, 0.02]) # Bottom center
        cbar = self.fig_corr.colorbar(sm, cax=cb_ax, orientation='horizontal')
        cbar.set_label('INPEFA Intensity', color=c['text'], fontsize=8)
        cbar.ax.xaxis.set_tick_params(color=c['text'], labelcolor=c['text'], labelsize=7)
        cbar.outline.set_edgecolor(c['spine'])
        # Add Low/High indicators
        cb_ax.text(-0.05, 0.5, 'LOW', transform=cb_ax.transAxes, va='center', ha='right', fontsize=7, color=c['text'], fontweight='normal')
        cb_ax.text(1.05, 0.5, 'HIGH', transform=cb_ax.transAxes, va='center', ha='left', fontsize=7, color=c['text'], fontweight='normal')

        # Custom Hover Formatter
        def corr_formatter(x, y):
            idx = self.tabs.currentIndex() # Double check context
            return f"Depth: {y:.1f} {self.depth_unit} | INPEFA: {x:.4f}"
        
        self.ax_ref.format_coord = corr_formatter
        self.ax_off.format_coord = corr_formatter

        # Shared Legend Logic moved to bottom section


        # Correlation Lines (Subsampled like notebook) 
        sub = max(1, len(pi) // 50)
        # Correlation Lines (Stepped Style for professional consistency)
        sub = max(1, len(pi) // 50)
        corr_line_color = '#00008b' # Dark Blue
        
        # Prepare axis lists for the stepped line helper
        # In this tab, there is only 1 panel per well
        r_list = [self.ax_ref]
        o_list = [self.ax_off]
        
        for i, j in zip(pi[::sub], pj[::sub]):
            self._draw_stepped_correlation_line(self.fig_corr, 
                                               grid[i] * self.depth_multiplier, 
                                               grid[j] * self.depth_multiplier, 
                                               r_list, o_list, 
                                               color=corr_line_color, 
                                               linewidth=0.8, alpha=0.7, linestyle='-', zorder=10)
        
        # Add proxy artist for correlation lines in legend
        from matplotlib.lines import Line2D
        corr_proxy = Line2D([0], [0], color=corr_line_color, linewidth=0.8, alpha=0.7, label='Correlation Path')
        
        for ax in [self.ax_ref, self.ax_off]:
            # Re-fetch handles/labels to include proxy if ref
            h, l = ax.get_legend_handles_labels()
            if ax == self.ax_ref:
                h.append(corr_proxy)
                l.append('Correlation Path')
            
            leg = ax.legend(h, l, loc='upper right', fontsize=8)
            if leg:
                frame = leg.get_frame()
                frame.set_facecolor(c['bg'])
                frame.set_edgecolor(c['spine'])
                for text in leg.get_texts():
                    text.set_color(c['text'])
        
        # Redraw sketches
        self.draw_mgr_corr.redraw([self.ax_ref, self.ax_off])
        self.canvas_corr.draw()

    def relaunch_app(self):
        """Signals the main loop to destroy this window and show the mode selection dialog again."""
        self.should_relaunch = True
        self.close()

    def init_compiled_tab(self):
        """Initializes the Compiled Plot tab with a comprehensive 4-panel view."""
        layout = QVBoxLayout(self.compiled_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        c = self.get_plot_colors()
        
        # Create scroll area for large plot
        self.scroll_compiled = QScrollArea()
        self.scroll_compiled.setWidgetResizable(True)
        self.scroll_compiled.setStyleSheet(f"background-color: {c['bg']}; border: none;")
        
        # Create canvas widget
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib Figure - Large size for comprehensive view
        self.fig_compiled = Figure(figsize=(24, 30), facecolor=c['bg'])
        self.canvas_compiled = FigureCanvas(self.fig_compiled)
        self.canvas_compiled.setStyleSheet(f"background-color: {c['bg']};")
        
        # Initialize Drawing Manager
        self.draw_mgr_compiled = DrawingManager(self.canvas_compiled, on_change=self.refresh_active_tab)
        
        # Toolbar (hidden in combine mode)
        self.toolbar_compiled = VerticalToolbar(self.canvas_compiled, self.compiled_tab)
        layout.addWidget(self.toolbar_compiled)

        # -----------------------------------------------------------------------
        # COMBINE MODE CONTROL BAR  (hidden by default, shown in combine mode)
        # Fix 4 & 5: Proper docked toolbar -- no more floating buttons
        # -----------------------------------------------------------------------
        combine_btn_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {TURQUOISE};
                border: 1px solid {TURQUOISE};
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 18px;
            }}
            QPushButton:hover {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
            }}
        """
        self.combine_controls_bar = QFrame()
        self.combine_controls_bar.setFrameShape(QFrame.NoFrame)
        self.combine_controls_bar.setFixedHeight(54)
        self.combine_controls_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {SIDEBAR_BG};
                border-bottom: 2px solid {TURQUOISE};
            }}
        """)
        cbar_layout = QHBoxLayout(self.combine_controls_bar)
        cbar_layout.setContentsMargins(12, 6, 12, 6)
        cbar_layout.setSpacing(10)

        self.back_to_menu_btn = QPushButton("Back to Menu")
        self.back_to_menu_btn.setStyleSheet(combine_btn_style)
        self.back_to_menu_btn.setCursor(Qt.PointingHandCursor)
        self.back_to_menu_btn.clicked.connect(self.exit_combine_mode)

        self.combine_theme_toggle_btn = QPushButton("Theme")
        self.combine_theme_toggle_btn.setStyleSheet(combine_btn_style)
        self.combine_theme_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.combine_theme_toggle_btn.clicked.connect(self.toggle_plot_theme)

        vsep = QFrame(); vsep.setFrameShape(QFrame.VLine)
        vsep.setStyleSheet("color: #444;")

        panels_lbl = QLabel("Display Panels:")
        panels_lbl.setStyleSheet(f"color: {SILVER}; font-weight: bold; font-size: 11px;")

        cbar_layout.addWidget(self.back_to_menu_btn)
        cbar_layout.addWidget(self.combine_theme_toggle_btn)
        cbar_layout.addWidget(vsep)
        cbar_layout.addWidget(panels_lbl)
        # The actual checkboxes are added here in combine mode; they are removed from the
        # normal_controls_bar and re-parented into this layout by setup_combine_ui / exit_combine_mode.
        cbar_layout.addStretch()
        self.combine_controls_bar.hide()
        layout.addWidget(self.combine_controls_bar)

        # -----------------------------------------------------------------------
        # NORMAL MODE PANEL SELECTOR  (shown in normal mode, hidden in combine mode)
        # -----------------------------------------------------------------------
        self.normal_controls_bar = QWidget()
        controls_layout = QHBoxLayout(self.normal_controls_bar)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(20)
        
        lbl_select = QLabel("Display Panels:")
        lbl_select.setStyleSheet(f"color: {c['text']}; font-weight: bold;")
        controls_layout.addWidget(lbl_select)
        
        checkbox_style = f"color: {c['text']};"
        
        self.check_inpefa = QCheckBox("Automatic")
        self.check_inpefa.setChecked(True)
        self.check_inpefa.setStyleSheet(checkbox_style)
        self.check_inpefa.stateChanged.connect(self.plot_compiled)
        controls_layout.addWidget(self.check_inpefa)
        
        self.check_geology = QCheckBox("Sequence")
        self.check_geology.setChecked(True)
        self.check_geology.setStyleSheet(checkbox_style)
        self.check_geology.stateChanged.connect(self.plot_compiled)
        controls_layout.addWidget(self.check_geology)
        
        self.check_sand = QCheckBox("Top Sand")
        self.check_sand.setChecked(True)
        self.check_sand.setStyleSheet(checkbox_style)
        self.check_sand.stateChanged.connect(self.plot_compiled)
        controls_layout.addWidget(self.check_sand)
        
        self.check_cluster = QCheckBox("MRGC")
        self.check_cluster.setChecked(True)
        self.check_cluster.setStyleSheet(checkbox_style)
        self.check_cluster.stateChanged.connect(self.plot_compiled)
        controls_layout.addWidget(self.check_cluster)
        
        self.check_rf = QCheckBox("Random Forest")
        self.check_rf.setChecked(True)
        self.check_rf.setStyleSheet(checkbox_style)
        self.check_rf.stateChanged.connect(self.plot_compiled)
        controls_layout.addWidget(self.check_rf)
        
        controls_layout.addStretch()
        layout.addWidget(self.normal_controls_bar)
        
        canvas_layout.addWidget(self.canvas_compiled)
        self.scroll_compiled.setWidget(canvas_widget)
        layout.addWidget(self.scroll_compiled)
        
        # Reentrancy guard for synchronized scrolling (Fix 2)
        self._combine_sync_in_progress = False

    # --- Log Detector Feature ---
    def init_detector_tab(self):
        """Initializes the Log Detector tab with a checklist of unique logs and a matching well list."""
        layout = QHBoxLayout(self.detector_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Left Panel: Log Selection
        left_panel = QGroupBox("1. Select Logs to Detect")
        left_layout = QVBoxLayout(left_panel)
        
        self.detector_log_list = QListWidget()
        self.detector_log_list.setStyleSheet(f"""
            QListWidget {{ 
                background-color: {SIDEBAR_BG}; 
                color: {SILVER}; 
                border: 1px solid #555;
                font-size: 13px;
            }}
            QListWidget::item {{ padding: 5px; }}
            QListWidget::item:hover {{ background-color: #444; }}
        """)
        # We will populate items with checkboxes manually
        self.detector_log_list.itemChanged.connect(self.update_detector_results)
        
        refresh_btn = QPushButton("Refresh Log List")
        refresh_btn.clicked.connect(self.refresh_detector_logs)
        
        left_layout.addWidget(refresh_btn)
        left_layout.addWidget(self.detector_log_list)
        
        # Right Panel: Results
        right_panel = QGroupBox("2. Wells Containing Selected Logs")
        right_layout = QVBoxLayout(right_panel)
        
        self.detector_well_result = QListWidget()
        self.detector_well_result.setStyleSheet(f"""
            QListWidget {{ 
                background-color: {DARK_BG}; 
                color: {TURQUOISE}; 
                border: 1px solid {TURQUOISE};
                font-weight: bold;
                font-size: 14px;
            }}
        """)
        
        self.detector_summary_lbl = QLabel("Detected: 0 wells")
        self.detector_summary_lbl.setStyleSheet(f"color: {SILVER}; font-style: italic;")
        
        self.export_detector_btn = QPushButton("Export Log Matrix to Excel")
        self.export_detector_btn.clicked.connect(self.export_detector_matrix)
        self.export_detector_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {TURQUOISE};
                color: {DARK_BG};
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #00b5b5; }}
        """)
        
        right_layout.addWidget(self.detector_well_result)
        right_layout.addWidget(self.detector_summary_lbl)
        right_layout.addWidget(self.export_detector_btn)
        
        # Add to main tab layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 1)
        
        # Initial population
        self.refresh_detector_logs()

    def refresh_detector_logs(self):
        """Fetches unique logs from all wells and populates the checklist."""
        self.detector_log_list.clear()
        try:
            unique_logs = self.log_detector_engine.get_all_unique_logs()
            for log in unique_logs:
                item = QListWidgetItem(log)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.detector_log_list.addItem(item)
            self.status_bar.showMessage(f"Found {len(unique_logs)} unique logs across all wells.")
        except Exception as e:
            self.status_bar.showMessage(f"Error refreshing logs: {e}")

    def update_detector_results(self):
        """Triggered when log checkboxes change. Filters wells based on selection."""
        selected_logs = []
        for i in range(self.detector_log_list.count()):
            item = self.detector_log_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_logs.append(item.text())
        
        self.detector_well_result.clear()
        
        if not selected_logs:
            self.detector_summary_lbl.setText("Selected: 0 logs | Detected: 0 wells")
            return
            
        try:
            matching_wells = self.log_detector_engine.find_wells_with_logs(selected_logs)
            
            for well in matching_wells:
                self.detector_well_result.addItem(well)
            
            self.detector_summary_lbl.setText(f"Selected: {len(selected_logs)} logs | Detected: {len(matching_wells)} wells")
        except Exception as e:
            self.status_bar.showMessage(f"Log Detector Error: {e}")

    def export_detector_matrix(self):
        """Creates an Excel overview of log availability based on selection."""
        from PyQt5.QtWidgets import QFileDialog
        
        selected_logs = []
        for i in range(self.detector_log_list.count()):
            item = self.detector_log_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_logs.append(item.text())
        
        if not selected_logs:
            QMessageBox.warning(self, "No Selection", "Please check at least one log curve to export.")
            return
            
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Export Log Availability", "Log_Availability_Matrix.xlsx", "Excel Files (*.xlsx)")
            if path:
                self.status_bar.showMessage("Generating matrix...")
                df = self.log_detector_engine.export_availability_matrix(selected_logs)
                df.to_excel(path, index=False)
                self.status_bar.showMessage(f"Successfully exported log matrix to {os.path.basename(path)}")
                QMessageBox.information(self, "Export Success", f"Log availability matrix saved to:\n{path}")
        except Exception as e:
            self.status_bar.showMessage(f"Export Error: {e}")
            QMessageBox.critical(self, "Export Failed", f"Failed to save Excel file:\n{str(e)}")

    def plot_compiled(self):
        """
        Creates a comprehensive compiled plot showing Reference and Offset wells.
        Panel selection is dynamic based on UI checkboxes.
        """
        self.fig_compiled.clear()
        c = self.get_plot_colors()
        self.fig_compiled.set_facecolor(c['bg'])
        
        ref = self.ref_well_combo.currentText()
        off = self.off_well_combo.currentText()
        unit_str = "m" if self.unit_combo.currentText() == "Meters" else "ft"
        
        if not ref or not off:
            self.canvas_compiled.draw()
            return
        
        try:
            # ===== Check Panel Visibility =====
            show_inpefa = self.check_inpefa.isChecked()
            show_geology = self.check_geology.isChecked()
            show_sand = self.check_sand.isChecked()
            show_cluster = self.check_cluster.isChecked()
            
            active_panels = []
            if show_inpefa: active_panels.append("INPEFA")
            if show_geology: active_panels.append("Sequence")
            if show_sand: active_panels.append("Top Sand")
            if show_cluster: active_panels.append("Cluster")
            if self.check_rf.isChecked(): active_panels.append("RF")
            
            num_active = len(active_panels)
            if num_active == 0:
                self.canvas_compiled.draw()
                return

            # ===== Load data =====
            has_corr_data = hasattr(self, 'last_plot_args') and self.last_plot_args is not None
            df1_geo = self.data_loader.load_las_data(ref)
            df2_geo = self.data_loader.load_las_data(off)
            
            # Check availability of MRGC and RF results
            has_mrgc = ref in self.mrgc_results and off in self.mrgc_results
            has_rf = ref in self.rf_results and off in self.rf_results

            # --- Global Depth Range Calculation ---
            if has_corr_data:
                # Priority 1: Use strictly the INPEFA overlapping interval
                grid_inpefa, _, _, _, _, _, _ = self.last_plot_args
                global_min = grid_inpefa.min() * self.depth_multiplier
                global_max = grid_inpefa.max() * self.depth_multiplier
            else:
                # Priority 2: Fallback to full range of all displayed data
                depth_list = []
                if not df1_geo.empty: depth_list.extend([df1_geo['Depth'].min(), df1_geo['Depth'].max()])
                if not df2_geo.empty: depth_list.extend([df2_geo['Depth'].min(), df2_geo['Depth'].max()])
                
                if has_mrgc:
                    depth_list.extend([self.mrgc_results[ref]['Depth'].min(), self.mrgc_results[ref]['Depth'].max()])
                    depth_list.extend([self.mrgc_results[off]['Depth'].min(), self.mrgc_results[off]['Depth'].max()])
                    
                if has_rf:
                    depth_list.extend([self.rf_results[ref]['Depth'].min(), self.rf_results[ref]['Depth'].max()])
                    depth_list.extend([self.rf_results[off]['Depth'].min(), self.rf_results[off]['Depth'].max()])
                
                if not depth_list:
                    self.canvas_compiled.draw()
                    return
                    
                global_min = min(depth_list) * self.depth_multiplier
                global_max = max(depth_list) * self.depth_multiplier
            
            # Add small padding for better visibility
            pad = (global_max - global_min) * 0.02
            global_min -= pad
            global_max += pad


            m1_geo = self.geology_engine.get_markers(ref, self.data_loader.all_markers)
            m2_geo = self.geology_engine.get_markers(off, self.data_loader.all_markers)
            
            m1_sand = self.sand_engine.get_markers(ref, self.data_loader.sand_markers)
            m2_sand = self.sand_engine.get_markers(off, self.data_loader.sand_markers)

            # Strict Filtering: Filter markers and LAS data if correlation exists
            if has_corr_data:
                grid_overlap, _, _, _, _, _, _ = self.last_plot_args
                o_min, o_max = grid_overlap.min(), grid_overlap.max()
                m1_geo = m1_geo[(m1_geo['MD'] >= o_min) & (m1_geo['MD'] <= o_max)]
                m2_geo = m2_geo[(m2_geo['MD'] >= o_min) & (m2_geo['MD'] <= o_max)]
                m1_sand = m1_sand[(m1_sand['MD'] >= o_min) & (m1_sand['MD'] <= o_max)]
                m2_sand = m2_sand[(m2_sand['MD'] >= o_min) & (m2_sand['MD'] <= o_max)]
                df1_geo = df1_geo[(df1_geo['Depth'] >= o_min) & (df1_geo['Depth'] <= o_max)]
                df2_geo = df2_geo[(df2_geo['Depth'] >= o_min) & (df2_geo['Depth'] <= o_max)]

            all_geo_surfs = sorted(list(set(m1_geo['Surface'].tolist() + m2_geo['Surface'].tolist())))
            geo_cmap = matplotlib.cm.get_cmap('tab20', len(all_geo_surfs)) if len(all_geo_surfs) > 0 else None
            geo_surf_colors = {surf: geo_cmap(i) for i, surf in enumerate(all_geo_surfs)} if geo_cmap else {}
            
            all_sand_surfs = sorted(list(set(m1_sand['Surface'].tolist() + m2_sand['Surface'].tolist())))
            sand_cmap = matplotlib.cm.get_cmap('tab20', len(all_sand_surfs)) if len(all_sand_surfs) > 0 else None
            sand_surf_colors = {surf: sand_cmap(i) for i, surf in enumerate(all_sand_surfs)} if sand_cmap else {}

            # ===== Setup Nested GridSpec =====
            # Primary grid: 2 columns (Reference Well block, Offset Well block)
            # wspace=0.4 provides a moderate gap between the two wells for connection lines
            gs0 = gridspec.GridSpec(1, 2, figure=self.fig_compiled, 
                                   wspace=0.4, 
                                   left=0.08, right=0.94, top=0.92, bottom=0.10)
            
            # Map width ratios for panels (Narrow specific ones)
            # INPEFA (1.0) vs Others (0.5 - 0.7)
            panel_ratios = {
                "INPEFA": 1.0,
                "Sequence": 0.5,
                "Top Sand": 0.5,
                "Cluster": 0.6,
                "RF": 0.6
            }
            
            ref_ratios = [panel_ratios[p] for p in active_panels]
            # Offset well has mirror/custom order: Cluster, RF, INPEFA, Sequence, Top Sand
            off_panels_ordered = []
            if show_cluster: off_panels_ordered.append("Cluster")
            if self.check_rf.isChecked(): off_panels_ordered.append("RF")
            if show_inpefa: off_panels_ordered.append("INPEFA")
            if show_geology: off_panels_ordered.append("Sequence")
            if show_sand: off_panels_ordered.append("Top Sand")
            
            off_ratios = [panel_ratios[p] for p in off_panels_ordered]
            
            # Nested grids for each well block
            # small wspace (0.04) makes panels within the same well tight
            gs_ref = gridspec.GridSpecFromSubplotSpec(1, len(active_panels), 
                                                     subplot_spec=gs0[0], 
                                                     wspace=0.04, 
                                                     width_ratios=ref_ratios)
                                                     
            gs_off = gridspec.GridSpecFromSubplotSpec(1, len(off_panels_ordered), 
                                                     subplot_spec=gs0[1], 
                                                     wspace=0.04, 
                                                     width_ratios=off_ratios)
            
            all_axes = []
            master_ax = None
            
            # Map of axes for connection lines
            ref_axes = {}
            off_axes = {}
            
            # --- REFERENCE WELL ---
            ref_idx = 0
            
            # Reference: INPEFA
            if show_inpefa:
                if has_corr_data:
                    grid, log_A, log_B, pi, pj, ref_name, off_name = self.last_plot_args
                    ax = self.fig_compiled.add_subplot(gs_ref[0, ref_idx], label="ref_inpefa")
                    if master_ax is None: master_ax = ax
                    
                    ax.set_facecolor(c['bg'])
                    ax.plot(log_A, grid * self.depth_multiplier, color="black", linewidth=1.1, zorder=5)
                    
                    vmin, vmax = np.nanmin(np.concatenate([log_A, log_B])), np.nanmax(np.concatenate([log_A, log_B]))
                    grad = np.linspace(0, 1, 256).reshape(1, -1)
                    im = ax.imshow(grad, aspect='auto', cmap='coolwarm', 
                                  extent=[vmin, vmax, grid.max() * self.depth_multiplier, grid.min() * self.depth_multiplier],
                                  alpha=0.8, zorder=1)
                    path = ax.fill_betweenx(grid * self.depth_multiplier, log_A, 0, color='none', zorder=2)
                    from matplotlib.path import Path
                    combined_path = Path.make_compound_path(*path.get_paths())
                    im.set_clip_path(combined_path, transform=ax.transData)
                    
                    ax.set_title("Automatic", fontweight='normal', fontsize=9, color=c['text'])
                    ax.set_xlabel("Value", fontsize=7, color=c['text'])
                    ax.set_ylabel(f"Depth ({self.depth_unit})", fontsize=9, color=c['text'])
                    ax.set_ylim(global_max, global_min)
                    ax.tick_params(colors=c['text'], labelsize=7)
                    ax.grid(True, alpha=0.2, color=c['grid'], linestyle='--')
                    for spine in ax.spines.values(): spine.set_color(c['spine'])
                    ref_axes['inpefa'] = ax
                    all_axes.append(ax)
                ref_idx += 1
            
            # Reference: Geology
            if show_geology:
                ax = self.fig_compiled.add_subplot(gs_ref[0, ref_idx], sharey=master_ax, label="ref_geology")
                if master_ax is None: master_ax = ax
                ax.set_facecolor(c['bg'])
                ax.plot(df1_geo['GR'], df1_geo['Depth'] * self.depth_multiplier, color="#006400", lw=1.0, alpha=0.9, zorder=2)
                ax.set_ylim(global_max, global_min)
                ax.set_title("Seq", fontweight='normal', fontsize=9, color=c['text'])
                ax.set_xlabel("GR", fontsize=7, color=c['text'])
                ax.tick_params(labelleft=(ref_idx == 0), colors=c['text'], labelsize=7)
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--')
                for spine in ax.spines.values(): spine.set_color(c['spine'])
                for _, row in m1_geo.iterrows():
                    color = geo_surf_colors.get(row['Surface'], '#000000')
                    ax.axhline(y=float(row['MD']) * self.depth_multiplier, color=color, ls='-', lw=0.8, alpha=0.9, zorder=10)
                ref_axes['geology'] = ax
                all_axes.append(ax)
                ref_idx += 1
                
            # Reference: Sand
            if show_sand:
                ax = self.fig_compiled.add_subplot(gs_ref[0, ref_idx], sharey=master_ax, label="ref_sand")
                if master_ax is None: master_ax = ax
                ax.set_facecolor(c['bg'])
                ax.plot(df1_geo['GR'], df1_geo['Depth'] * self.depth_multiplier, color="#006400", lw=1.0, alpha=0.9, zorder=2)
                ax.set_ylim(global_max, global_min)
                ax.set_title("Top Sand", fontweight='normal', fontsize=9, color=c['text'])
                ax.set_xlabel("GR", fontsize=7, color=c['text'])
                ax.tick_params(labelleft=(ref_idx == 0), colors=c['text'], labelsize=7)
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--')
                for spine in ax.spines.values(): spine.set_color(c['spine'])
                for _, row in m1_sand.iterrows():
                    color = sand_surf_colors.get(row['Surface'], '#000000')
                    ax.axhline(y=float(row['MD']) * self.depth_multiplier, color=color, ls='-', lw=0.8, alpha=0.9, zorder=10)
                ref_axes['sand'] = ax
                all_axes.append(ax)
                ref_idx += 1

            # Reference: Cluster
            if show_cluster:
                if has_mrgc:
                    try:
                        d_mrgc = self.mrgc_results[ref]
                        # Strict Filtering: Filter MRGC data to INPEFA overlap if correlation exists
                        if has_corr_data:
                            grid_overlap, _, _, _, _, _, _ = self.last_plot_args
                            o_min, o_max = grid_overlap.min(), grid_overlap.max()
                            d_mrgc = d_mrgc[(d_mrgc['Depth'] >= o_min) & (d_mrgc['Depth'] <= o_max)]

                        ax = self.fig_compiled.add_subplot(gs_ref[0, ref_idx], sharey=master_ax, label="ref_cluster")
                        if master_ax is None: master_ax = ax
                        ax.set_facecolor(c['bg'])
                        df_sorted = d_mrgc.sort_values('Depth')
                        depths, clusters = df_sorted['Depth'].values * self.depth_multiplier, df_sorted['Cluster'].values
                        cmap_obj = plt.get_cmap('tab20')
                        cmap_colors = cmap_obj(np.linspace(0, 1, 20))
                        if len(depths) > 0:
                            curr_c, start_i = clusters[0], 0
                            for i in range(1, len(clusters)):
                                if clusters[i] != curr_c:
                                    ax.fill_betweenx(depths[start_i:i+1], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                                    curr_c, start_i = clusters[i], i
                            ax.fill_betweenx(depths[start_i:], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                        ax.set_title("MRGC", color=c['text'], size=8, weight='normal')
                        ax.set_xticks([])
                        ax.set_ylim(global_max, global_min)
                        ax.tick_params(labelleft=(ref_idx == 0), colors=c['text'], labelsize=7)
                        for s in ax.spines.values(): s.set_color(c['spine'])
                        ref_axes['cluster'] = ax
                        all_axes.append(ax)
                    except Exception as e:
                        print(f"Error plotting MRGC cluster for {ref}: {e}")
                ref_idx += 1

            # Reference: RF
            if self.check_rf.isChecked():
                if has_rf:
                    d_rf = self.rf_results[ref]
                    # Strict Filtering: Filter RF data to INPEFA overlap if correlation exists
                    if has_corr_data:
                        grid_overlap, _, _, _, _, _, _ = self.last_plot_args
                        o_min, o_max = grid_overlap.min(), grid_overlap.max()
                        d_rf = d_rf[(d_rf['Depth'] >= o_min) & (d_rf['Depth'] <= o_max)]

                    ax = self.fig_compiled.add_subplot(gs_ref[0, ref_idx], sharey=master_ax, label="ref_rf")
                    if master_ax is None: master_ax = ax
                    ax.set_facecolor(c['bg'])
                    df_sorted = d_rf.sort_values('Depth')
                    depths = df_sorted['Depth'].values * self.depth_multiplier
                    preds = df_sorted['Predicted_Lithology'].values
                    if len(depths) > 0:
                        curr_p, start_i = preds[0], 0
                        color_map = {0: '#8B4513', 1: '#ffd700', 2: '#8B4513', 3: '#8B4513'}
                        for i in range(1, len(preds)):
                            if preds[i] != curr_p:
                                color = color_map.get(int(curr_p), '#808080')
                                ax.fill_betweenx(depths[start_i:i+1], 0, 1, color=color, alpha=0.9)
                                curr_p, start_i = preds[i], i
                        color = color_map.get(int(curr_p), '#808080')
                        ax.fill_betweenx(depths[start_i:], 0, 1, color=color, alpha=0.9)
                    ax.set_title("RF", color=c['text'], size=8, weight='normal')
                    ax.set_xticks([])
                    ax.set_ylim(global_max, global_min)
                    ax.tick_params(labelleft=(ref_idx == 0), colors=c['text'], labelsize=7)
                    ref_axes['rf'] = ax
                    all_axes.append(ax)
                ref_idx += 1
            
            # --- OFFSET WELL ---
            off_idx = 0
            
            # Offset: Cluster
            if show_cluster:
                if has_mrgc:
                    try:
                        d_mrgc = self.mrgc_results[off]
                        # Strict Filtering: Filter MRGC data to INPEFA overlap if correlation exists
                        if has_corr_data:
                            grid_overlap, _, _, _, _, _, _ = self.last_plot_args
                            o_min, o_max = grid_overlap.min(), grid_overlap.max()
                            d_mrgc = d_mrgc[(d_mrgc['Depth'] >= o_min) & (d_mrgc['Depth'] <= o_max)]

                        ax = self.fig_compiled.add_subplot(gs_off[0, off_idx], sharey=master_ax, label="off_cluster")
                        ax.set_facecolor(c['bg'])
                        df_sorted = d_mrgc.sort_values('Depth')
                        depths, clusters = df_sorted['Depth'].values * self.depth_multiplier, df_sorted['Cluster'].values
                        if len(depths) > 0:
                            curr_c, start_i = clusters[0], 0
                            for i in range(1, len(clusters)):
                                if clusters[i] != curr_c:
                                    ax.fill_betweenx(depths[start_i:i+1], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                                    curr_c, start_i = clusters[i], i
                            ax.fill_betweenx(depths[start_i:], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                        ax.set_title("MRGC", color=c['text'], size=8, weight='normal')
                        ax.set_xticks([])
                        ax.set_ylim(global_max, global_min)
                        ax.tick_params(labelleft=False)
                        for s in ax.spines.values(): s.set_color(c['spine'])
                        off_axes['cluster'] = ax
                        all_axes.append(ax)
                    except Exception as e:
                        print(f"Error plotting MRGC cluster for {off}: {e}")
                off_idx += 1

            # Offset: RF
            if self.check_rf.isChecked():
                if has_rf:
                    d_rf = self.rf_results[off]
                    # Strict Filtering: Filter RF data to INPEFA overlap if correlation exists
                    if has_corr_data:
                        grid_overlap, _, _, _, _, _, _ = self.last_plot_args
                        o_min, o_max = grid_overlap.min(), grid_overlap.max()
                        d_rf = d_rf[(d_rf['Depth'] >= o_min) & (d_rf['Depth'] <= o_max)]

                    ax = self.fig_compiled.add_subplot(gs_off[0, off_idx], sharey=master_ax, label="off_rf")
                    ax.set_facecolor(c['bg'])
                    df_sorted = d_rf.sort_values('Depth')
                    depths = df_sorted['Depth'].values * self.depth_multiplier
                    preds = df_sorted['Predicted_Lithology'].values
                    if len(depths) > 0:
                        curr_p, start_i = preds[0], 0
                        color_map = {0: '#8B4513', 1: '#ffd700', 2: '#8B4513', 3: '#8B4513'}
                        for i in range(1, len(preds)):
                            if preds[i] != curr_p:
                                color = color_map.get(int(curr_p), '#808080')
                                ax.fill_betweenx(depths[start_i:i+1], 0, 1, color=color, alpha=0.9)
                                curr_p, start_i = preds[i], i
                        color = color_map.get(int(curr_p), '#808080')
                        ax.fill_betweenx(depths[start_i:], 0, 1, color=color, alpha=0.9)
                    ax.set_title("RF", color=c['text'], size=8, weight='normal')
                    ax.set_xticks([])
                    ax.set_ylim(global_max, global_min)
                    ax.tick_params(labelleft=False)
                    off_axes['rf'] = ax
                    all_axes.append(ax)
                off_idx += 1
            
            # Offset: INPEFA
            if show_inpefa:
                if has_corr_data:
                    ax = self.fig_compiled.add_subplot(gs_off[0, off_idx], sharey=master_ax, label="off_inpefa")
                    ax.set_facecolor(c['bg'])
                    ax.plot(log_B, grid * self.depth_multiplier, color="black", linewidth=1.1, zorder=5) 
                    
                    im = ax.imshow(grad, aspect='auto', cmap='coolwarm', 
                                  extent=[vmin, vmax, grid.max() * self.depth_multiplier, grid.min() * self.depth_multiplier],
                                  alpha=0.8, zorder=1)
                    path = ax.fill_betweenx(grid * self.depth_multiplier, log_B, 0, color='none', zorder=2)
                    combined_path = Path.make_compound_path(*path.get_paths())
                    im.set_clip_path(combined_path, transform=ax.transData)
                    
                    ax.set_title("Automatic", fontweight='normal', fontsize=9, color=c['text'])
                    ax.set_xlabel("Value", fontsize=7, color=c['text'])
                    ax.tick_params(labelleft=False, colors=c['text'], labelsize=7)
                    ax.grid(True, alpha=0.2, color=c['grid'], linestyle='--')
                    for spine in ax.spines.values(): spine.set_color(c['spine'])
                    off_axes['inpefa'] = ax
                    all_axes.append(ax)
                off_idx += 1
            
            # Offset: Geology
            if show_geology:
                ax = self.fig_compiled.add_subplot(gs_off[0, off_idx], sharey=master_ax, label="off_geology")
                ax.set_facecolor(c['bg'])
                ax.plot(df2_geo['GR'], df2_geo['Depth'] * self.depth_multiplier, color="#006400", lw=1.0, alpha=0.9, zorder=2)
                ax.set_ylim(global_max, global_min)
                ax.set_title("Seq", fontweight='normal', fontsize=9, color=c['text'])
                ax.set_xlabel("GR", fontsize=7, color=c['text'])
                ax.tick_params(labelleft=False, colors=c['text'], labelsize=7)
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--')
                for spine in ax.spines.values(): spine.set_color(c['spine'])
                for _, row in m2_geo.iterrows():
                    color = geo_surf_colors.get(row['Surface'], '#000000')
                    ax.axhline(y=float(row['MD']) * self.depth_multiplier, color=color, ls='-', lw=0.8, alpha=0.9, zorder=10)
                off_axes['geology'] = ax
                all_axes.append(ax)
                off_idx += 1
            
            # Offset: Sand
            if show_sand:
                ax = self.fig_compiled.add_subplot(gs_off[0, off_idx], sharey=master_ax, label="off_sand")
                ax.set_facecolor(c['bg'])
                ax.plot(df2_geo['GR'], df2_geo['Depth'] * self.depth_multiplier, color="#006400", lw=1.0, alpha=0.9, zorder=2)
                ax.set_ylim(global_max, global_min)
                ax.set_title("Top Sand", fontweight='normal', fontsize=9, color=c['text'])
                ax.set_xlabel("GR", fontsize=7, color=c['text'])
                ax.tick_params(labelleft=False, colors=c['text'], labelsize=7)
                ax.grid(True, alpha=0.2, color=c['grid'], ls='--')
                for spine in ax.spines.values(): spine.set_color(c['spine'])
                for _, row in m2_sand.iterrows():
                    color = sand_surf_colors.get(row['Surface'], '#000000')
                    ax.axhline(y=float(row['MD']) * self.depth_multiplier, color=color, ls='-', lw=0.8, alpha=0.9, zorder=10)
                off_axes['sand'] = ax
                all_axes.append(ax)
                off_idx += 1

            # ===== CONNECTION LINES (Stepped Style) =====
            ref_axes_list = [ref_axes[p] for p in ["inpefa", "geology", "sand", "cluster", "rf"] if p in ref_axes]
            off_axes_list = [off_axes[p] for p in ["cluster", "rf", "inpefa", "geology", "sand"] if p in off_axes]
            
            # Use the shared class method for stepped lines
            def draw_stepped_line(y_ref, y_off, r_list, o_list, color, label=None, **kwargs):
                self._draw_stepped_correlation_line(self.fig_compiled, y_ref, y_off, r_list, o_list, color, label, **kwargs)

            # DTW Correlation
            if 'inpefa' in ref_axes and 'inpefa' in off_axes:
                grid, log_A, log_B, pi, pj, _, _ = self.last_plot_args
                # Performance Fix: Re-implement subsampling (match individual tab) to prevent "Not Responding"
                sub = max(1, len(pi) // 50)
                
                for idx, (i, j) in enumerate(zip(pi[::sub], pj[::sub])):
                    draw_stepped_line(grid[i] * self.depth_multiplier, grid[j] * self.depth_multiplier, ref_axes_list, off_axes_list, 
                                     color='#00008b', label=f"AT-{idx+1}",   # Dark Blue as requested 
                                     linewidth=0.8, alpha=0.7, linestyle='-', zorder=5)
            
            # Geology Markers
            if 'geology' in ref_axes and 'geology' in off_axes and self.show_geo_corr:
                common_geo = set(m1_geo['Surface']).intersection(set(m2_geo['Surface']))
                for surf in common_geo:
                    d1 = float(m1_geo[m1_geo['Surface'] == surf]['MD'].iloc[0])
                    d2 = float(m2_geo[m2_geo['Surface'] == surf]['MD'].iloc[0])
                    color = geo_surf_colors.get(surf, '#000000')
                    draw_stepped_line(d1 * self.depth_multiplier, d2 * self.depth_multiplier, ref_axes_list, off_axes_list, 
                                     color=color, label=str(surf),
                                     linewidth=0.8, alpha=0.8, linestyle='-', zorder=8)

            # Sand Markers
            if 'sand' in ref_axes and 'sand' in off_axes and hasattr(self, 'sand_corr_btn') and self.sand_corr_btn.isChecked():
                common_sand = set(m1_sand['Surface']).intersection(set(m2_sand['Surface']))
                for surf in common_sand:
                    d1 = float(m1_sand[m1_sand['Surface'] == surf]['MD'].iloc[0])
                    d2 = float(m2_sand[m2_sand['Surface'] == surf]['MD'].iloc[0])
                    color = sand_surf_colors.get(surf, '#000000')
                    draw_stepped_line(d1 * self.depth_multiplier, d2 * self.depth_multiplier, ref_axes_list, off_axes_list, 
                                     color=color, label=str(surf),
                                     linewidth=0.8, alpha=0.8, linestyle='-', zorder=8)

            # Labels and Final Redraw
            self.fig_compiled.text(0.25, 0.985, f"REFERENCE: {ref}", ha='center', va='top', fontsize=14, fontweight='normal', color=c['text'])
            self.fig_compiled.text(0.75, 0.985, f"OFFSET: {off}", ha='center', va='top', fontsize=14, fontweight='normal', color=c['text'])
            
            # Add shared colorbar for INPEFA in Compiled Plot
            if show_inpefa and has_corr_data:
                sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
                cb_ax = self.fig_compiled.add_axes([0.08, 0.02, 0.15, 0.012]) # Bottom left area (Fix overlap)
                cbar = self.fig_compiled.colorbar(sm, cax=cb_ax, orientation='horizontal')
                cbar.set_label('Integrated Prediction Error Filter Analysis (INPEFA)', color=c['text'], fontsize=8)
                cbar.ax.xaxis.set_tick_params(color=c['text'], labelcolor=c['text'], labelsize=7)
                cbar.outline.set_edgecolor(c['spine'])
                # Add Low/High indicators
                cb_ax.text(-0.05, 0.5, 'LOW', transform=cb_ax.transAxes, va='center', ha='right', fontsize=7, color=c['text'], fontweight='normal')
                cb_ax.text(1.05, 0.5, 'HIGH', transform=cb_ax.transAxes, va='center', ha='left', fontsize=7, color=c['text'], fontweight='normal')
            
            # Add RF Legend if shown
            if self.check_rf.isChecked() and has_rf:
                from matplotlib.patches import Patch
                rf_elements = [
                    Patch(facecolor='#ffd700', edgecolor='k', label='Shaly Sand'),
                    Patch(facecolor='#8B4513', edgecolor='k', label='Shale')
                ]
                self.fig_compiled.legend(handles=rf_elements, loc='lower right', 
                                       title="Legend of Random Forest", fontsize=8, title_fontsize=9,
                                       facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'],
                                       bbox_to_anchor=(0.98, 0.01), ncol=2)
            
            # Add MRGC Legend if shown
            if show_cluster and has_mrgc:
                cmap_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
                mrgc_elements = [Patch(facecolor=cmap_colors[i%20], label=f"Cluster {i+1}") for i in range(10)]
                self.fig_compiled.legend(handles=mrgc_elements, loc='lower center', 
                                       title="Legend of MRGC", fontsize=6.5, title_fontsize=7.5,
                                       facecolor=c['bg'], edgecolor=c['spine'], labelcolor=c['text'],
                                       bbox_to_anchor=(0.5, 0.005), ncol=10,
                                       handlelength=0.7, handletextpad=0.2, columnspacing=0.5)
            
            self.draw_mgr_compiled.redraw(all_axes)
            self.canvas_compiled.draw()
            self.status_bar.showMessage("Compiled Plot updated.")
            
        except Exception as e:
            import traceback
            self.status_bar.showMessage(f"Error: {str(e)}")
            print(traceback.format_exc())
            self.canvas_compiled.draw()

    def setup_menu(self):
        """Adds project management menus."""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{ background-color: {DARK_BG}; color: {SILVER}; }}
            QMenuBar::item:selected {{ background-color: {TURQUOISE}; color: {DARK_BG}; }}
            QMenu {{ background-color: {SIDEBAR_BG}; color: {SILVER}; border: 1px solid #555; }}
            QMenu::item:selected {{ background-color: {TURQUOISE}; color: {DARK_BG}; }}
        """)
        
        file_menu = menubar.addMenu("Project")
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        
        open_action = file_menu.addAction("Open Project")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_project)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def save_project(self):
        """Serializes current application state to a file."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Well Sync Project (*.wcp)")
        if not path: return
        
        try:
            self.status_bar.showMessage("Saving project... please wait.")
            
            # 1. Collect UI Settings
            # Determine DTW mode from radio buttons
            if self.rb_sand_dtw.isChecked(): dtw_mode = "sand"
            elif self.rb_geo_dtw.isChecked(): dtw_mode = "geo"
            else: dtw_mode = "full"
            
            ui_state = {
                'ref_well': self.ref_well_combo.currentText(),
                'off_well': self.off_well_combo.currentText(),
                'inpefa_term': self.inpefa_combo.currentText(),
                'unit_idx': self.unit_combo.currentIndex(),
                'dtw_mode': dtw_mode,
                'panel_ratio': self.ratio_slider.value(),
                'checks_compiled': {
                    'inpefa': self.check_inpefa.isChecked(),
                    'geology': self.check_geology.isChecked(),
                    'sand': self.check_sand.isChecked(),
                    'cluster': self.check_cluster.isChecked(),
                    'rf': self.check_rf.isChecked()
                },
                # Extended Settings
                'mrgc_alpha': self.mrgc_alpha_spin.value(),
                'mrgc_clusters': self.mrgc_clusters_combo.currentText(),
                'mrgc_w1_selected': [cb.text().strip() for cb in self.mrgc_w1_checks if cb.isChecked()],
                'mrgc_w2_selected': [cb.text().strip() for cb in self.mrgc_w2_checks if cb.isChecked()],
                'rf_vsh_cutoff': self.rf_vsh_cutoff.value(),
                'rf_n_estimators': self.rf_n_estimators.value(),
                'rf_auto_optimize': self.rf_auto_optimize.isChecked(),
                'rf_selected_features': [feat.strip() for feat, cb in self.rf_feature_checks.items() if cb.isChecked()],
                'rf_trained': getattr(self, 'rf_trained', False),
                'gpu_enabled': self.gpu_check.isChecked(),
                'plot_theme': self.theme_combo.currentText(),
                'show_geo_corr': getattr(self, 'show_geo_corr', True)
            }
            
            # 2. Collect Results
            state = {
                'ui_state': ui_state,
                'mrgc_results': self.mrgc_results,
                'rf_results': self.rf_results,
                'last_plot_args': getattr(self, 'last_plot_args', None),
                'current_export_data': getattr(self, 'current_export_data', None),
                # Drawings from all managers
                'drawings': {
                    'log': self.draw_mgr_log.drawings,
                    'mrgc': self.draw_mgr_mrgc.drawings,
                    'rf': self.draw_mgr_rf.drawings,
                    'geo': self.draw_mgr_geo.drawings,
                    'sand': self.draw_mgr_sand.drawings,
                    'corr': self.draw_mgr_corr.drawings,
                    'compiled': self.draw_mgr_compiled.drawings
                }
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            
            self.status_bar.showMessage(f"Project saved to {os.path.basename(path)}", 5000)
            QMessageBox.information(self, "Success", f"Project successfully saved to:\n{path}")
            
        except Exception as e:
            import traceback
            self.status_bar.showMessage("Save failed.")
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}\n{traceback.format_exc()}")

    def load_project(self, filename=None):
        """Deserializes and restores application state."""
        from PyQt5.QtWidgets import QFileDialog
        if not filename:
            path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Well Sync Project (*.wcp)")
        else:
            path = filename
            
        if not path: return
        
        try:
            self.status_bar.showMessage("Loading project... please wait.")
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # 1. Restore Results Data
            self.mrgc_results = state.get('mrgc_results', {})
            self.rf_results = state.get('rf_results', {})
            self.last_plot_args = state.get('last_plot_args')
            self.current_export_data = state.get('current_export_data')
            
            # 2. Restore UI Settings
            ui = state.get('ui_state', {})
            # Block signals to prevent triggering redraws too early
            self.unit_combo.blockSignals(True)
            self.unit_combo.setCurrentIndex(ui.get('unit_idx', 0))
            self.unit_combo.blockSignals(False)
            
            self.ref_well_combo.blockSignals(True)
            self.ref_well_combo.setCurrentText(ui.get('ref_well', ''))
            self.ref_well_combo.blockSignals(False)
            
            self.off_well_combo.blockSignals(True)
            self.off_well_combo.setCurrentText(ui.get('off_well', ''))
            self.off_well_combo.blockSignals(False)
            
            self.inpefa_combo.setCurrentText(ui.get('inpefa_term', 'Mid Term'))
            
            dtw_mode = ui.get('dtw_mode', 'sand')
            if dtw_mode == "sand": self.rb_sand_dtw.setChecked(True)
            elif dtw_mode == "geo": self.rb_geo_dtw.setChecked(True)
            else: self.rb_full_dtw.setChecked(True)
            
            self.ratio_slider.setValue(ui.get('panel_ratio', 50))
            
            checks = ui.get('checks_compiled', {})
            self.check_inpefa.setChecked(checks.get('inpefa', True))
            self.check_geology.setChecked(checks.get('geology', True))
            self.check_sand.setChecked(checks.get('sand', True))
            self.check_cluster.setChecked(checks.get('cluster', True))
            self.check_rf.setChecked(checks.get('rf', True))
            
            # 3. Restore Drawings
            draws = state.get('drawings', {})
            self.draw_mgr_log.drawings = draws.get('log', {})
            self.draw_mgr_mrgc.drawings = draws.get('mrgc', {})
            self.draw_mgr_rf.drawings = draws.get('rf', {})
            self.draw_mgr_geo.drawings = draws.get('geo', {})
            self.draw_mgr_sand.drawings = draws.get('sand', {})
            self.draw_mgr_corr.drawings = draws.get('corr', {})
            self.draw_mgr_compiled.drawings = draws.get('compiled', {})
            
            # 4. Trigger Global Refresh
            self.status_bar.showMessage("Restoring visualizations...")
            self.update_available_logs()
            self.update_mrgc_features()
            self.sync_rf_wells()
            
            # --- 5. Restore Analystical Settings & Feature Selections ---
            # These must be set AFTER update_mrgc_features and sync_rf_wells
            
            # MRGC Features (Only restore if present in saved file)
            if 'mrgc_w1_selected' in ui:
                m1_sel = ui.get('mrgc_w1_selected', [])
                for cb in self.mrgc_w1_checks:
                    cb.setChecked(cb.text().strip() in m1_sel)
            if 'mrgc_w2_selected' in ui:
                m2_sel = ui.get('mrgc_w2_selected', [])
                for cb in self.mrgc_w2_checks:
                    cb.setChecked(cb.text().strip() in m2_sel)
                
            # MRGC Params
            self.mrgc_alpha_spin.setValue(ui.get('mrgc_alpha', 10.0))
            self.mrgc_clusters_combo.setCurrentText(str(ui.get('mrgc_clusters', 'Auto')))
            
            # RF Features
            if 'rf_selected_features' in ui:
                rf_sel = ui.get('rf_selected_features', [])
                for feat, cb in self.rf_feature_checks.items():
                    cb.setChecked(feat.strip() in rf_sel)
                
            # RF Params
            self.rf_vsh_cutoff.setValue(ui.get('rf_vsh_cutoff', 0.4))
            self.rf_n_estimators.setValue(ui.get('rf_n_estimators', 100))
            self.rf_auto_optimize.setChecked(ui.get('rf_auto_optimize', False))
            self.rf_trained = ui.get('rf_trained', bool(self.rf_results))
            
            # Restore specific result references for current wells
            ref_w = ui.get('ref_well', '')
            off_w = ui.get('off_well', '')
            if self.rf_results:
                self.rf_predictions_ref = self.rf_results.get(ref_w)
                self.rf_predictions_off = self.rf_results.get(off_w)
            
            # Global Settings
            self.gpu_check.setChecked(ui.get('gpu_enabled', False))
            if 'plot_theme' in ui:
                self.theme_combo.setCurrentText(ui.get('plot_theme', 'Dark'))
            self.show_geo_corr = ui.get('show_geo_corr', True)
            if hasattr(self, 'geo_corr_btn'):
                self.geo_corr_btn.setChecked(self.show_geo_corr)

            self.refresh_active_tab()
            
            self.status_bar.showMessage(f"Project loaded: {os.path.basename(path)}", 5000)
            
        except Exception as e:
            import traceback
            self.status_bar.showMessage("Load failed.")
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}\n{traceback.format_exc()}")

    def load_combined_projects(self):
        """Opens multiple project files, validates the chain, and prepares for combined plotting."""
        from PyQt5.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Multiple Projects to Combine", "", "Well Sync Project (*.wcp)")
        if not files or len(files) < 1:
            QMessageBox.warning(self, "Selection Canceled", "Please select at least one project file.")
            return

        self.status_bar.showMessage("Loading and chaining projects...")
        loaded_projects = []
        
        try:
            for f_path in files:
                with open(f_path, 'rb') as f:
                    state = pickle.load(f)
                    loaded_projects.append(state)
            
            # Chain validation and sorting
            # We want to form a chain like A-B, B-C, C-D
            # To do this, we'll try to order them based on well names
            if len(loaded_projects) > 1:
                # Simple greedy chaining: find a start, then find the next
                # For now, let's assume the user selects them in order, or we can try to sort
                # A better way: find the project where the Offset Well is not a Reference Well in any other project, 
                # or find the Reference Well that is not an Offset Well in any other project.
                
                all_refs = [p['ui_state']['ref_well'] for p in loaded_projects]
                all_offs = [p['ui_state']['off_well'] for p in loaded_projects]
                
                # Find the starting project (ref_well is not in all_offs)
                start_p = None
                for p in loaded_projects:
                    if p['ui_state']['ref_well'] not in all_offs:
                        start_p = p
                        break
                
                if start_p is None: start_p = loaded_projects[0] # fallback
                
                chain = [start_p]
                remaining = [p for p in loaded_projects if p != start_p]
                
                while remaining:
                    last_well = chain[-1]['ui_state']['off_well']
                    found_next = False
                    for i, p in enumerate(remaining):
                        if p['ui_state']['ref_well'] == last_well:
                            chain.append(remaining.pop(i))
                            found_next = True
                            break
                    if not found_next:
                        # Chain broken!
                        QMessageBox.critical(self, "Chain Error", 
                            f"Proyek tidak membentuk rantai yang kontinu.\n"
                            f"Sumur terakhir: {last_well} tidak ditemukan sebagai 'Reference Well' di proyek lainnya.")
                        return

                self.project_chain = chain
            else:
                self.project_chain = loaded_projects

            # NEW: Cache LAS data to prevent repeated file reads and improve performance
            self.las_cache = {}
            wells_to_cache = []
            for p in self.project_chain:
                wells_to_cache.extend([p['ui_state']['ref_well'], p['ui_state']['off_well']])
            
            self.status_bar.showMessage("Caching well logs...")
            for well in set(wells_to_cache):
                df = self.data_loader.load_las_data(well)
                if not df.empty:
                    self.las_cache[well] = df

            self.is_combine_mode = True
            self.setup_combine_ui()
            self.plot_combined_correlation()
            self.status_bar.showMessage("Combine Correlation successful!", 5000)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Load Error", f"Failed to load projects:\n{str(e)}\n{traceback.format_exc()}")

    def setup_combine_ui(self):
        """Reconfigures the UI for full-screen combined correlation plotting (Fix 4/5)."""
        # Hide Sidebar and Header
        self.sidebar.hide()
        self.header_frame.hide()
        
        # Switch to Compiled Plot tab and hide the tab bar
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.compiled_tab))
        self.tabs.tabBar().hide()
        
        # Hide normal-mode widgets; show combine-mode control bar
        # Keep toolbar_compiled visible so user can still zoom/pan (Fix: issue 1)
        self.toolbar_compiled.show()
        self.normal_controls_bar.hide()
        self.combine_controls_bar.show()
        
        # Move the 5 panel checkboxes into the combine_controls_bar layout (Fix 4/5)
        # Get the QHBoxLayout of combine_controls_bar
        cbar_layout = self.combine_controls_bar.layout()
        cb_style = f"color: {SILVER}; font-size: 11px;"
        for cb in [self.check_inpefa, self.check_geology, self.check_sand,
                   self.check_cluster, self.check_rf]:
            cb.setStyleSheet(cb_style)
            cbar_layout.insertWidget(cbar_layout.count() - 1, cb)  # insert before stretch
        
        # Fix 3: Reconnect stateChanged to plot_combined_correlation (not plot_compiled)
        for cb in [self.check_inpefa, self.check_geology, self.check_sand,
                   self.check_cluster, self.check_rf]:
            try:
                cb.stateChanged.disconnect(self.plot_compiled)
            except TypeError:
                pass
            cb.stateChanged.connect(self.plot_combined_correlation)
        
        # Maximize window
        self.showMaximized()

    def exit_combine_mode(self):
        """Restores the UI and returns to the initial state (restarts app logic)."""
        # Move the checkboxes back to normal_controls_bar layout (Fix 4/5)
        normal_layout = self.normal_controls_bar.layout()
        stretch_index = normal_layout.count() - 1  # last item is stretch
        normal_cb_style = f"color: {TEXT_COLOR};"
        for cb in [self.check_inpefa, self.check_geology, self.check_sand,
                   self.check_cluster, self.check_rf]:
            cb.setStyleSheet(normal_cb_style)
            normal_layout.insertWidget(stretch_index, cb)
            stretch_index += 1
        
        # Fix 3: Reconnect signals back to plot_compiled
        for cb in [self.check_inpefa, self.check_geology, self.check_sand,
                   self.check_cluster, self.check_rf]:
            try:
                cb.stateChanged.disconnect(self.plot_combined_correlation)
            except TypeError:
                pass
            cb.stateChanged.connect(self.plot_compiled)
        
        # Restore normal-mode UI widgets
        self.combine_controls_bar.hide()
        self.normal_controls_bar.show()
        self.toolbar_compiled.show()
        
        # Re-show sidebar, header, tab bar
        self.sidebar.show()
        self.header_frame.show()
        self.tabs.tabBar().show()
        
        self.is_combine_mode = False
        self.showNormal()
        
        # Restart the app to show mode selection again
        import subprocess
        try:
            subprocess.Popen([sys.executable] + sys.argv)
            os._exit(0)
        except Exception:
            QApplication.quit()






    def plot_combined_correlation(self):
        """Creates a multi-well continuous compiled plot using the Combine Engine.
        Fix 1: Markers filtered to overlap interval per panel pair.
        Fix 2: All well panels share a single master y-axis for synchronized scrolling.
        """
        if not self.project_chain: return
        
        self.fig_compiled.clear()
        self.fig_compiled.suptitle("")
        c = self.get_plot_colors()
        self.fig_compiled.set_facecolor(c['bg'])
        
        num_wells = len(self.project_chain) + 1
        num_projects = len(self.project_chain)
        
        # 1. Calculate Global Range and Setup Grid
        g_min, g_max = self.combine_engine.get_global_depth_range(self.project_chain)
        if g_min is None:
            self.canvas_compiled.draw()
            return
        
        # Better figure layout: more top space for labels, tighter bottom
        gs0 = gridspec.GridSpec(1, num_wells, figure=self.fig_compiled,
                               wspace=0.7,
                               left=0.06, right=0.96, top=0.91, bottom=0.10)

        # 2. Build well list for plotting
        wells_to_plot = []
        for i, p in enumerate(self.project_chain):
            if i == 0: wells_to_plot.append(p['ui_state']['ref_well'])
            wells_to_plot.append(p['ui_state']['off_well'])
        
        active_panels = []
        if self.check_inpefa.isChecked(): active_panels.append("INPEFA")
        if self.check_geology.isChecked(): active_panels.append("Sequence")
        if self.check_sand.isChecked(): active_panels.append("Top Sand")
        if self.check_cluster.isChecked(): active_panels.append("Cluster")
        if self.check_rf.isChecked(): active_panels.append("RF")
        
        if not active_panels: active_panels = ["INPEFA"]
        
        panel_ratios = {"INPEFA": 1.0, "Sequence": 0.5, "Top Sand": 0.5, "Cluster": 0.6, "RF": 0.6}
        ratios = [panel_ratios[p] for p in active_panels]
        
        well_axes = []
        
        # Calculate Global INPEFA Intensity Range
        g_ivmin, g_ivmax = float('inf'), float('-inf')
        if "INPEFA" in active_panels:
            for well_name in wells_to_plot:
                d = self.combine_engine.get_well_data(self.project_chain, well_name)
                if d['inpefa']:
                    g_ivmin = min(g_ivmin, np.nanmin(d['inpefa'][1]))
                    g_ivmax = max(g_ivmax, np.nanmax(d['inpefa'][1]))

        # Fix 1: Pre-compute per-pair overlap intervals for marker filtering
        # overlap_for_well[well_name] = (overlap_min_raw, overlap_max_raw)
        overlap_for_well = {}
        for p in self.project_chain:
            ref_name = p['ui_state']['ref_well']
            off_name = p['ui_state']['off_well']
            ref_df = self.las_cache.get(ref_name, None)
            off_df = self.las_cache.get(off_name, None)
            if ref_df is not None and off_df is not None and not ref_df.empty and not off_df.empty:
                ov_min = max(ref_df['Depth'].min(), off_df['Depth'].min())
                ov_max = min(ref_df['Depth'].max(), off_df['Depth'].max())
            else:
                ov_min, ov_max = None, None
            # Store overlap for both wells in this pair
            key = (ref_name, off_name)
            overlap_for_well[key] = (ov_min, ov_max)

        def get_well_overlap(well_name):
            """Returns (min, max) raw depth range that this well overlaps with its neighbours."""
            ranges = []
            for p in self.project_chain:
                ref, off = p['ui_state']['ref_well'], p['ui_state']['off_well']
                if ref == well_name or off == well_name:
                    ov = overlap_for_well.get((ref, off))
                    if ov and ov[0] is not None:
                        ranges.append(ov)
            if not ranges:
                return None, None
            return min(r[0] for r in ranges), max(r[1] for r in ranges)

        # Fix 2: Use a single GLOBAL master axis so all well panels share y-axis
        global_master_ax = None

        for i, well_name in enumerate(wells_to_plot):
            data = self.combine_engine.get_well_data(self.project_chain, well_name)
            gs_well = gridspec.GridSpecFromSubplotSpec(1, len(active_panels),
                                                      subplot_spec=gs0[0, i],
                                                      wspace=0.04,
                                                      width_ratios=ratios)
            axes_dict = {}

            # Fix 1: Get overlap range for this well
            ov_min_raw, ov_max_raw = get_well_overlap(well_name)
            m = self.depth_multiplier
            ov_min = ov_min_raw * m if ov_min_raw is not None else None
            ov_max = ov_max_raw * m if ov_max_raw is not None else None

            for j, panel in enumerate(active_panels):
                # Fix 2: all axes share the global master
                ax = self.fig_compiled.add_subplot(gs_well[0, j],
                                                   sharey=global_master_ax,
                                                   label=f"well_{i}_{panel}")
                if global_master_ax is None:
                    global_master_ax = ax
                
                ax.set_facecolor(c['bg'])
                ax.set_ylim(g_max, g_min)
                ax.tick_params(labelleft=(i == 0 and j == 0), colors=c['text'], labelsize=7)
                for spine in ax.spines.values(): spine.set_color(c['spine'])
                ax.grid(True, axis='y', alpha=0.15, color=c.get('grid', '#555'), linestyle='--')

                # --- Panel Plotting ---
                if panel == "INPEFA" and data['inpefa']:
                    grid, val = data['inpefa']
                    vmin, vmax = g_ivmin, g_ivmax
                    grad = np.linspace(0, 1, 256).reshape(1, -1)
                    im = ax.imshow(grad, aspect='auto', cmap='coolwarm',
                                  extent=[vmin, vmax, grid.max()*m, grid.min()*m],
                                  alpha=0.8, zorder=1)
                    path = ax.fill_betweenx(grid*m, val, 0, color='none', zorder=2)
                    combined_path = Path.make_compound_path(*path.get_paths())
                    im.set_clip_path(combined_path, transform=ax.transData)
                    ax.plot(val, grid*m, color=c['text'], lw=0.8, zorder=5)
                    ax.set_title("INP", fontsize=8, color=c['text'], fontweight='normal')

                elif panel == "Sequence" and not data['las'].empty:
                    ax.plot(data['las']['GR'], data['las']['Depth']*m, color="#006400", lw=0.8)
                    # Fix 1: Only draw markers within the overlap interval
                    if data['markers_seq'] is not None:
                        for _, row in data['markers_seq'].iterrows():
                            md_raw = float(row['MD'])
                            md = md_raw * m
                            if ov_min is not None and not (ov_min <= md <= ov_max):
                                continue  # Skip markers outside overlap
                            ax.axhline(y=md, color="blue", ls='-', lw=0.7, alpha=0.75)
                    ax.set_title("Seq", fontsize=8, color=c['text'], fontweight='normal')

                elif panel == "Top Sand" and not data['las'].empty:
                    ax.plot(data['las']['GR'], data['las']['Depth']*m, color="#8b4513", lw=0.8)
                    # Fix 1: Only draw markers within the overlap interval
                    if data['markers_sand'] is not None:
                        for _, row in data['markers_sand'].iterrows():
                            md_raw = float(row['MD'])
                            md = md_raw * m
                            if ov_min is not None and not (ov_min <= md <= ov_max):
                                continue  # Skip markers outside overlap
                            ax.axhline(y=md, color="red", ls='-', lw=0.7, alpha=0.75)
                    ax.set_title("Sand", fontsize=8, color=c['text'], fontweight='normal')

                elif panel == "Cluster" and data['mrgc'] is not None:
                    df = data['mrgc'].sort_values('Depth')
                    depths, clusters = df['Depth'].values * m, df['Cluster'].values
                    cmap_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
                    if len(depths) > 0:
                        curr_c, start_i = clusters[0], 0
                        for idx_p in range(1, len(clusters)):
                            if clusters[idx_p] != curr_c:
                                ax.fill_betweenx(depths[start_i:idx_p+1], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                                curr_c, start_i = clusters[idx_p], idx_p
                        ax.fill_betweenx(depths[start_i:], 0, 1, color=cmap_colors[int(curr_c)%20], alpha=0.9)
                    ax.set_title("MRGC", fontsize=8, color=c['text'], fontweight='normal')
                    ax.set_xticks([])

                elif panel == "RF" and data['rf'] is not None:
                    df = data['rf'].sort_values('Depth')
                    depths, preds = df['Depth'].values * m, df['Predicted_Lithology'].values
                    color_map = {0: '#8B4513', 1: '#ffd700', 2: '#8B4513', 3: '#8B4513'}
                    if len(depths) > 0:
                        curr_p, start_i = preds[0], 0
                        for idx_p in range(1, len(preds)):
                            if preds[idx_p] != curr_p:
                                ax.fill_betweenx(depths[start_i:idx_p+1], 0, 1, color=color_map.get(int(curr_p), 'grey'), alpha=0.9)
                                curr_p, start_i = preds[idx_p], idx_p
                        ax.fill_betweenx(depths[start_i:], 0, 1, color=color_map.get(int(curr_p), 'grey'), alpha=0.9)
                    ax.set_title("RF", fontsize=8, color=c['text'], fontweight='normal')
                    ax.set_xticks([])

                axes_dict[panel] = ax
            well_axes.append(axes_dict)
            
            # Well Name Labels — top only, clean single line
            pos = gs0[0, i].get_position(self.fig_compiled)
            mid_x = pos.x0 + pos.width / 2
            self.fig_compiled.text(mid_x, 0.935, well_name, ha='center',
                                  color=c['text'], fontweight='bold', fontsize=11)

        # 3. Draw Connections Between Wells
        for i in range(num_projects):
            conn = self.combine_engine.get_connection_data(self.project_chain, i)
            if not conn: continue
            
            l_axes = well_axes[i]
            r_axes = well_axes[i+1]
            ref_list = [l_axes[p] for p in active_panels if p in l_axes]
            off_list = [r_axes[p] for p in active_panels if p in r_axes]

            # Helper: convert data-y to figure-fraction y using the first ref axis
            def data_to_fig_y(y_data, ax):
                """Convert a data coordinate to figure-fraction y."""
                try:
                    y_disp = ax.transData.transform((0, y_data))[1]
                    y_fig = self.fig_compiled.transFigure.inverted().transform((0, y_disp))[1]
                    return y_fig
                except Exception:
                    return 0.5

            # Figure-fraction x midpoint between left-well right edge and right-well left edge
            try:
                pos_l = gs0[0, i].get_position(self.fig_compiled)
                pos_r = gs0[0, i+1].get_position(self.fig_compiled)
                mid_fig_x = (pos_l.x1 + pos_r.x0) / 2.0
            except Exception:
                mid_fig_x = 0.5

            label_ax = ref_list[0] if ref_list else None  # axis used for y-conversion

            # Connect INPEFA (Auto Correlation) — draw DTW label once per pair
            if "INPEFA" in l_axes and "INPEFA" in r_axes:
                grid, pi, pj = conn['grid'], conn['pi'], conn['pj']
                sub = max(1, len(pi) // 50)
                for p_i, p_j in zip(pi[::sub], pj[::sub]):
                    self._draw_stepped_correlation_line(self.fig_compiled, grid[p_i]*m, grid[p_j]*m,
                                                     ref_list, off_list, color='#00008b', alpha=0.5, lw=0.8)

            # Fix 1: Connect Sequence — only common markers within pair overlap
            if "Sequence" in l_axes and "Sequence" in r_axes:
                mk1 = self.combine_engine.get_well_data(self.project_chain, conn['ref_name'])['markers_seq']
                mk2 = self.combine_engine.get_well_data(self.project_chain, conn['off_name'])['markers_seq']
                if mk1 is not None and mk2 is not None:
                    common = set(mk1['Surface']) & set(mk2['Surface'])
                    for surf in common:
                        y1_raw = float(mk1[mk1['Surface']==surf]['MD'].iloc[0])
                        y2_raw = float(mk2[mk2['Surface']==surf]['MD'].iloc[0])
                        
                        self._draw_stepped_correlation_line(
                            self.fig_compiled, y1_raw*m, y2_raw*m,
                            ref_list, off_list, color='blue', alpha=0.65, lw=0.9)
                        # Marker name label at connection midpoint
                        if label_ax is not None:
                            y_mid_fig = data_to_fig_y((y1_raw + y2_raw) / 2.0 * m, label_ax)
                            if 0.05 < y_mid_fig < 0.95:
                                self.fig_compiled.text(
                                    mid_fig_x, y_mid_fig, str(surf),
                                    ha='center', va='center', fontsize=6,
                                    color='blue', alpha=0.85,
                                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.5, ec='none'))

            # Fix 1: Connect Top Sand — only common markers within pair overlap
            if "Top Sand" in l_axes and "Top Sand" in r_axes:
                mk1 = self.combine_engine.get_well_data(self.project_chain, conn['ref_name'])['markers_sand']
                mk2 = self.combine_engine.get_well_data(self.project_chain, conn['off_name'])['markers_sand']
                if mk1 is not None and mk2 is not None:
                    common = set(mk1['Surface']) & set(mk2['Surface'])
                    for surf in common:
                        y1_raw = float(mk1[mk1['Surface']==surf]['MD'].iloc[0])
                        y2_raw = float(mk2[mk2['Surface']==surf]['MD'].iloc[0])

                        self._draw_stepped_correlation_line(
                            self.fig_compiled, y1_raw*m, y2_raw*m,
                            ref_list, off_list, color='red', alpha=0.65, lw=0.9)
                        # Marker name label at connection midpoint
                        if label_ax is not None:
                            y_mid_fig = data_to_fig_y((y1_raw + y2_raw) / 2.0 * m, label_ax)
                            if 0.05 < y_mid_fig < 0.95:
                                self.fig_compiled.text(
                                    mid_fig_x, y_mid_fig, str(surf),
                                    ha='center', va='center', fontsize=6,
                                    color='red', alpha=0.85,
                                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.5, ec='none'))

        # 4. Colorbar and Legends
        if "INPEFA" in active_panels and g_ivmin != float('inf'):
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=g_ivmin, vmax=g_ivmax))
            cb_ax = self.fig_compiled.add_axes([0.08, 0.02, 0.15, 0.012]) # Bottom left area (Fix overlap)
            cbar = self.fig_compiled.colorbar(sm, cax=cb_ax, orientation='horizontal')
            cbar.set_label('Integrated Prediction Error Filter Analysis (INPEFA)', color=c['text'], fontsize=8)
            cbar.ax.xaxis.set_tick_params(color=c['text'], labelcolor=c['text'], labelsize=7)
            cbar.outline.set_edgecolor(c['spine'])
            cb_ax.text(-0.05, 0.5, 'LOW', transform=cb_ax.transAxes, va='center', ha='right', fontsize=7, color=c['text'])
            cb_ax.text(1.05, 0.5, 'HIGH', transform=cb_ax.transAxes, va='center', ha='left', fontsize=7, color=c['text'])

        # Gather MRGC elements for Compiled Plot
        mrgc_elements = []
        if "Cluster" in active_panels:
            found_clusters = set()
            for well_axes_dict in well_axes:
                if "Cluster" in well_axes_dict:
                    ax = well_axes_dict["Cluster"]
                    h, l = ax.get_legend_handles_labels()
                    for hi, li in zip(h, l):
                        if li not in [e.get_label() for e in mrgc_elements]:
                            mrgc_elements.append(hi)
            # If no manual handles, try to find from color map used in plotting
            if not mrgc_elements:
                cmap_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
                # We don't know exactly which clusters are present, but we can show a representative legend if needed
                # However, the user wants the MRGC clusters legend. 
                # Let's try to harvest from axes if possible, or recreate patches.
                # Re-generating patches from used colors:
                for i in range(10): # assuming max 10 clusters as per init
                     mrgc_elements.append(Patch(facecolor=cmap_colors[i%20], label=f"Cluster {i+1}"))

        if "RF" in active_panels:
            rf_elements = [Patch(facecolor='#ffd700', edgecolor='k', label='Shaly Sand'),
                           Patch(facecolor='#8B4513', edgecolor='k', label='Shale')]
            self.fig_compiled.legend(handles=rf_elements, loc='lower right', title="Legend of Random Forest",
                                   fontsize=8, title_fontsize=9, facecolor=c['bg'],
                                   edgecolor=c['spine'], labelcolor=c['text'], ncol=2)

        if mrgc_elements:
            self.fig_compiled.legend(handles=mrgc_elements, loc='lower center', title="Legend of MRGC",
                                   fontsize=6.5, title_fontsize=7.5, facecolor=c['bg'],
                                   edgecolor=c['spine'], labelcolor=c['text'], ncol=min(10, len(mrgc_elements)),
                                   handlelength=0.7, handletextpad=0.2, columnspacing=0.5,
                                   bbox_to_anchor=(0.5, 0.005))

        self.canvas_compiled.draw()

    def on_map_mode_changed(self):
        """Handles switching between 'Chosen Well' and 'Route Plotter' modes."""
        is_route = self.rb_route_map.isChecked()
        self.route_wells_list.setVisible(is_route)
        
        # Disable/Enable analysis buttons based on mode
        self.run_btn.setEnabled(not is_route)
        if hasattr(self, 'mrgc_run_btn'): self.mrgc_run_btn.setEnabled(not is_route)
        if hasattr(self, 'rf_train_btn'): self.rf_train_btn.setEnabled(not is_route)
        
        # Refresh the 2D map
        self.plot_2d_map()
        
        if is_route:
            self.status_bar.showMessage("Mode: Route Plotter (Analysis disabled, Max 10 wells)")
            self.tabs.setCurrentIndex(0) # Automatically switch to Map tab
        else:
            self.status_bar.showMessage("Mode: Chosen Well Maps (Analysis enabled)")

        
def main():
    app = QApplication(sys.argv)
    
    while True:
        # 1. Show Mode Selection Dialog
        selection = ProjectSelectionDialog()
        if selection.exec_() != QDialog.Accepted:
            break
            
        window = WellCorrelationApp()
        
        # 2. Handle Load Mode
        if selection.selected_mode == "load":
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, window.load_project)
        elif selection.selected_mode == "combine":
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, window.load_combined_projects)
            
        window.show()
        
        # Run app and wait for exit or relaunch signal
        res = app.exec_()
        
        # Check if we should relaunch or exit
        if hasattr(window, 'should_relaunch') and window.should_relaunch:
            window.deleteLater()
            continue
        else:
            sys.exit(res)

if __name__ == "__main__":
    main()
