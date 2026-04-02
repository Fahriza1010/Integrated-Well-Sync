"""
EngineFor_RandomForestLithology.py

Random Forest Lithology Classification Engine
Based on: "Application of random forest algorithm in classification of 
          logging lithology" (Kang & Lu, 2020)

Features:
- Supervised classification using well log data
- Vshale-based label generation with configurable cutoff
- Random Forest with optimized n_estimators
- SVM baseline for comparison
- Variable importance analysis (Gini & OOB)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# GPU/XGBoost Support detection
HAS_XGB = False
xgb = None

def _init_rf_gpu():
    global HAS_XGB, xgb
    import sys
    import os
    
    # Windows DLL loading fix for nvidia-* pip packages
    if sys.platform == 'win32':
        # Search site-packages for nvidia libraries (required for newer pip-installed CUDA)
        for path in sys.path:
            if 'site-packages' in path and os.path.exists(os.path.join(path, 'nvidia')):
                nvidia_root = os.path.join(path, 'nvidia')
                for root, dirs, files in os.walk(nvidia_root):
                    if 'bin' in dirs:
                        bin_path = os.path.join(root, 'bin')
                        if any(f.endswith('.dll') for f in os.listdir(bin_path)):
                            try:
                                # This is critical for Windows to find the CUDA DLLs
                                os.add_dll_directory(bin_path)
                            except: pass
    try:
        import xgboost
        xgb = xgboost
        return True
    except ImportError:
        return False
    except Exception:
        return False

HAS_XGB = _init_rf_gpu()


class RandomForestLithologyEngine:
    """
    Engine for lithology classification using Random Forest algorithm.
    
    Follows the methodology from Kang & Lu (2020):
    - Uses well log curves as input features
    - Generates labels from Vshale (steiber) with configurable cutoff
    - Optimizes n_estimators for best OOB score
    - Provides variable importance analysis
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize the engine.
        
        Parameters:
        -----------
        data_loader : DataLoader, optional
            Instance of EngineFor_DataLoader for loading well data
        """
        self.data_loader = data_loader
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.svm_model = None
        self.feature_names = []
        self.importance_df = None
        self.training_metrics = {}
        self.svm_metrics = {}
        self.has_xgb = HAS_XGB
        self._use_gpu = False  # Default to CPU for stability
        
        # Print GPU/XGBoost status
        if HAS_XGB:
            print(f"RF: XGBoost available but GPU disabled by default. Enable via GUI if needed.")
        else:
            print(f"RF: Using standard RandomForest (XGBoost not available)")
        
    @property
    def use_gpu(self):
        return self._use_gpu
    
    @property
    def has_gpu(self):
        try:
            import cupy as cp
            return True
        except:
            return False
        
    @use_gpu.setter
    def use_gpu(self, value):
        # Force False if hardware/library not available
        self._use_gpu = value if HAS_XGB else False
        
    def calculate_vshale_steiber(self, gr_values):
        """
        Calculate Vshale using Steiber method.
        
        Parameters:
        -----------
        gr_values : array-like
            Gamma Ray log values
            
        Returns:
        --------
        np.ndarray
            Vshale values (0-1 range)
        """
        gr = np.array(gr_values)
        gr_min = np.nanmin(gr)
        gr_max = np.nanmax(gr)
        
        if gr_max <= gr_min:
            return np.zeros(len(gr))
            
        # Linear shale index (Ish)
        ish = np.clip((gr - gr_min) / (gr_max - gr_min), 0, 1)
        
        # Steiber formula
        vsh_steiber = 0.5 * ish / (1.5 - ish)
        
        return np.clip(vsh_steiber, 0, 1)
    
    
    def generate_expert_labels_v2(self, well_name, depths, vsh_values, vsh_cutoff=0.3):
        """
        Generate labels based on specialized sand marker logic.
        
        Logic:
        - Top Sand: marker ending in 'sd'
        - Bottom Sand: depth below Top Sand where Vshale >= cutoff
        - Range [Top, Bottom] = Sand (1)
        - Others = Shale (3)
        """
        labels = np.full(len(depths), 3) # Default to Shale
        
        if self.data_loader is None:
            return labels
            
        from EngineFor_SandPlot import SandPlotEngine
        sand_markers = SandPlotEngine.get_markers(well_name, self.data_loader.sand_markers)
        
        if sand_markers.empty:
            return labels
            
        # Top sands are those with 'Surface' ending in 'sd'
        top_sands = sand_markers[sand_markers['Surface'].str.endswith('sd', na=False)].sort_values('MD')
        
        for _, row in top_sands.iterrows():
            top_depth = row['MD']
            
            # Find Bottom Sand: first depth below top_depth where Vshale >= cutoff
            # Or if Vshale never reaches cutoff, use the bottom of the well or next marker
            mask_below = (depths > top_depth)
            if not any(mask_below):
                continue
                
            idx_below = np.where(mask_below)[0]
            bottom_idx = idx_below[-1] # Default to bottom of well
            
            for i in idx_below:
                if vsh_values[i] >= vsh_cutoff:
                    bottom_idx = i
                    break
            
            bottom_depth = depths[bottom_idx]
            
            # Label the range
            labels[(depths >= top_depth) & (depths <= bottom_depth)] = 1
            
        return labels

    def preprocess_data(self, well_names, feature_curves=None, 
                        vsh_cutoff=0.3, test_size=0.3, random_state=42):
        """
        Preprocess well log data for training.
        
        Parameters:
        -----------
        well_names : list
            List of well names to include
        feature_curves : list, optional
            Log curves to use as features. If None, uses all available.
        vsh_cutoff : float
            Vshale cutoff for sand/non-sand classification
        test_size : float
            Fraction of data for testing (default 0.3)
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, feature_names, full_df)
        """
        all_data = []
        
        for well_name in well_names:
            try:
                # Get available curves
                available = self.data_loader.get_available_curves(well_name)
                
                # Default features if not specified
                if feature_curves is None:
                    # Common petrophysical curves
                    default_curves = ['GR', 'RHOB', 'DT', 'SP', 'ILD', 'NPHI', 'CALI']
                    curves_to_use = [c for c in default_curves if c in available]
                else:
                    curves_to_use = [c for c in feature_curves if c in available]
                
                if not curves_to_use:
                    continue
                    
                # Load each curve
                well_df = None
                for curve in curves_to_use:
                    try:
                        curve_data = self.data_loader.load_curve_data(
                            well_name, curve
                        )
                        if well_df is None:
                            well_df = curve_data.rename(columns={'Value': curve})
                        else:
                            # Merge on Depth
                            temp_df = curve_data.rename(columns={'Value': curve})
                            well_df = pd.merge(well_df, temp_df, on='Depth', how='outer')
                    except Exception as e:
                        continue
                
                if well_df is None or well_df.empty:
                    continue
                
                # Load curves for Expert Labeling (Vshale)
                try:
                    # Attempt to load pre-calculated VSH data
                    vsh_data = self.data_loader.load_vshale_data(well_name)
                    if vsh_data.empty:
                        # Fallback to calculation
                        gr_data = self.data_loader.load_curve_data(well_name, 'GR')
                        vsh_vals = self.calculate_vshale_steiber(gr_data['Value'].values)
                        vsh_data = pd.DataFrame({'Depth': gr_data['Depth'], 'Vsh': vsh_vals})
                    
                    from scipy.interpolate import interp1d
                    f_vsh = interp1d(vsh_data['Depth'], vsh_data['Vsh'], bounds_error=False, fill_value=np.nan)
                    well_df['Vsh_val'] = f_vsh(well_df['Depth'])
                    
                    # Calculate Expert Labels v2
                    well_df['Label'] = self.generate_expert_labels_v2(well_name, well_df['Depth'].values, well_df['Vsh_val'].values, vsh_cutoff)
                    well_df['Well'] = well_name
                    
                except Exception as e:
                    print(f"Error generating labels for {well_name}: {e}")
                    import traceback; traceback.print_exc()
                    continue
                
                all_data.append(well_df)
                
            except Exception as e:
                print(f"Error processing {well_name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data found for any well")
        
        # Combine all wells
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Identify feature columns
        self.feature_names = [c for c in combined_df.columns 
                             if c not in ['Depth', 'Well', 'GR_val', 'Vsh_val', 'Label', 'Vshale']]
        
        # Drop rows with missing values in features or labels
        clean_df = combined_df.dropna(subset=self.feature_names + ['Label'])
        
        if clean_df.empty:
            raise ValueError("No valid samples after removing NaN values")
        
        # Extract features and labels
        X = clean_df[self.feature_names].values
        y = clean_df['Label'].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels (Crucial for XGBoost which requires 0..N-1)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, self.feature_names, clean_df
    
    def optimize_n_estimators(self, X_train, y_train, 
                              n_range=(100, 600), step=50):
        """
        Find optimal n_estimators using OOB error.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        n_range : tuple
            Range of n_estimators to test (min, max)
        step : int
            Step size for search
            
        Returns:
        --------
        tuple : (optimal_n, oob_scores)
        """
        oob_scores = {}
        best_n = n_range[0]
        best_score = 0
        
        for n in range(n_range[0], n_range[1] + 1, step):
            if self.use_gpu and self.has_xgb:
                # XGBoost variant - use 'hist' tree_method (gpu_hist not available in this version)
                rf = xgb.XGBRFClassifier(
                    n_estimators=n,
                    tree_method='hist',
                    random_state=42,
                    n_jobs=-1
                )
            else:
                rf = RandomForestClassifier(
                    n_estimators=n,
                    max_features='sqrt',
                    criterion='gini',
                    bootstrap=True,
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1
                )
            
            # XGBoost handles GPU internally, no need for manual CuPy conversion
            rf.fit(X_train, y_train)
            
            # OOB Score fallback for XGBoost (XGB doesn't have .oob_score_)
            if hasattr(rf, 'oob_score_'):
                score = rf.oob_score_
            else:
                # For XGBoost we use the training accuracy as a proxy for n_estimators optimization 
                # or we could use cross-validation (slower)
                score = rf.score(X_train, y_train)
                
            oob_scores[n] = score
            
            if score > best_score:
                best_score = score
                best_n = n
        
        return best_n, oob_scores
    
    def train_random_forest(self, X_train, y_train, X_test, y_test,
                            n_estimators=None, optimize=True):
        """
        Train Random Forest classifier.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Testing data
        n_estimators : int, optional
            Number of trees. If None and optimize=True, auto-optimize.
        optimize : bool
            Whether to optimize n_estimators
            
        Returns:
        --------
        dict : Training metrics including accuracy, confusion matrix
        """
        # Optimize n_estimators if needed
        if n_estimators is None and optimize:
            print("Optimizing n_estimators...")
            n_estimators, oob_scores = self.optimize_n_estimators(X_train, y_train)
            print(f"Optimal n_estimators: {n_estimators}")
        elif n_estimators is None:
            n_estimators = 100
        
        # Build Random Forest (following paper parameters)
        if self.use_gpu and self.has_xgb:
            # Use 'hist' tree_method (more compatible across XGBoost versions)
            self.model = xgb.XGBRFClassifier(
                n_estimators=n_estimators,
                tree_method='hist',
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features='sqrt',  # sqrt(n_features) as per paper
                criterion='gini',     # Gini index for splits
                bootstrap=True,       # Bagging with replacement
                oob_score=True,       # For importance calculation
                random_state=42,
                n_jobs=-1
            )
        
        # XGBoost handles GPU acceleration internally - no need for manual data conversion
        # Train
        try:
            self.model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)
            
        except Exception as e:
            print(f"RF Training Error with GPU={self.use_gpu}: {str(e)}")
            if self.use_gpu and self.has_xgb:
                print("Falling back to CPU Random Forest...")
                # Rebuild model with CPU-only RandomForest
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features='sqrt',
                    criterion='gini',
                    bootstrap=True,
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                y_prob = self.model.predict_proba(X_test)
            else:
                raise
        
        # Evaluate
        self.training_metrics = self.evaluate_model(y_test, y_pred, "Random Forest")
        self.training_metrics['n_estimators'] = n_estimators
        self.training_metrics['oob_score'] = getattr(self.model, 'oob_score_', 0.0)
        
        # Compute variable importance
        self.compute_variable_importance()
        
        return self.training_metrics
    
    def train_svm_baseline(self, X_train, y_train, X_test, y_test):
        """
        Train SVM classifier for comparison.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Testing data
            
        Returns:
        --------
        dict : SVM metrics for comparison
        """
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.svm_model.fit(X_train, y_train)
        y_pred = self.svm_model.predict(X_test)
        
        self.svm_metrics = self.evaluate_model(y_test, y_pred, "SVM")
        
        return self.svm_metrics
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate model using confusion matrix metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name for reporting
            
        Returns:
        --------
        dict : Metrics including overall accuracy, UA, PA
        """
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate User's Accuracy (UA) and Producer's Accuracy (PA)
        # UA = diagonal / row sum (precision per class)
        # PA = diagonal / column sum (recall per class)
        
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        
        ua = np.diag(cm) / np.where(row_sums > 0, row_sums, 1)
        pa = np.diag(cm) / np.where(col_sums > 0, col_sums, 1)
        
        target_names = ['Coal', 'Sand', 'Shaly sand', 'Shale']
        # For labeling v2, we only use Sand (1) and Shale (3) mostly, but keep indices for compatibility
        # Map existing labels to target names present in y_true/y_pred
        present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        present_names = [target_names[int(i)] for i in present_labels]
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'users_accuracy': ua,
            'producers_accuracy': pa,
            'classification_report': classification_report(
                y_true, y_pred, 
                labels=present_labels,
                target_names=present_names,
                output_dict=True
            )
        }
        
        return metrics
    
    def compute_variable_importance(self):
        """
        Compute variable importance using Gini and OOB methods.
        
        Returns:
        --------
        pd.DataFrame : Importance ranking for each feature
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest first.")
        
        # Gini importance (mean decrease in impurity)
        gini_importance = self.model.feature_importances_
        
        # OOB-based importance (permutation importance approximation)
        # Using Gini as proxy since sklearn's RF uses impurity-based importance
        
        self.importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Gini_Importance': gini_importance,
            'Rank': np.argsort(-gini_importance) + 1
        }).sort_values('Gini_Importance', ascending=False)
        
        return self.importance_df
    
    def predict_lithology(self, well_name, feature_curves=None):
        """
        Predict lithology for a well using trained model.
        
        Parameters:
        -----------
        well_name : str
            Well name to predict
        feature_curves : list, optional
            Features to use (must match training features)
            
        Returns:
        --------
        pd.DataFrame : Predictions with depth, features, label, probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest first.")
        
        # Load well data
        available = self.data_loader.get_available_curves(well_name)
        
        if feature_curves is None:
            feature_curves = self.feature_names
        
        # Load each curve
        well_df = None
        for curve in feature_curves:
            if curve not in available:
                continue
            try:
                curve_data = self.data_loader.load_curve_data(
                    well_name, curve
                )
                if well_df is None:
                    well_df = curve_data.rename(columns={'Value': curve})
                else:
                    temp_df = curve_data.rename(columns={'Value': curve})
                    well_df = pd.merge(well_df, temp_df, on='Depth', how='outer')
            except:
                continue
        
        if well_df is None or well_df.empty:
            raise ValueError(f"Could not load data for {well_name}")
        
        # Ensure all required features exist
        missing = [f for f in self.feature_names if f not in well_df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Drop NaN rows
        clean_df = well_df.dropna(subset=self.feature_names)
        
        if clean_df.empty:
            raise ValueError("No valid samples after removing NaN values")
        
        # Extract and scale features
        X = clean_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        # XGBoost handles GPU data internally - no manual conversion needed
        # Predict
        y_encoded = self.model.predict(X_scaled)
        # Decode labels back to original (e.g., 0->1, 1->3)
        predictions = self.label_encoder.inverse_transform(y_encoded)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Build result DataFrame
        result_df = clean_df.copy()
        result_df['Predicted_Lithology'] = predictions
        
        lith_map = {0: 'Coal', 1: 'Sand', 2: 'Shaly sand', 3: 'Shale'}
        result_df['Lithology_Label'] = [lith_map.get(p, f'Unknown ({p})') for p in predictions]
        
        # Probabilities - mapping encoded classes back to labels
        for encoded_idx, original_label in enumerate(self.label_encoder.classes_):
            name = lith_map.get(original_label, f"Class_{original_label}")
            if encoded_idx < probabilities.shape[1]:
                result_df[f'Probability_{name.replace(" ", "")}'] = probabilities[:, encoded_idx]
        
        result_df['Well'] = well_name
        
        # Ensure deep depths are at the bottom (shallow to deep sorting)
        return result_df.sort_values('Depth')
    
    def get_comparison_summary(self):
        """
        Get comparison summary between RF and SVM.
        
        Returns:
        --------
        pd.DataFrame : Side-by-side comparison
        """
        if not self.training_metrics or not self.svm_metrics:
            return None
        
        summary = pd.DataFrame({
            'Metric': ['Overall Accuracy'],
            'Random_Forest': [f"{self.training_metrics['accuracy']:.2%}"],
            'SVM': [f"{self.svm_metrics['accuracy']:.2%}"]
        })
        
        return summary
