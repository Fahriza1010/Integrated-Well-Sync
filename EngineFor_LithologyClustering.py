"""
Lithology Clustering for Well Log Data

This module implements depth-wise lithology clustering using Multi-Resolution Graph-Based
Clustering (MRGC) to identify different rock types/facies at each depth interval.

Author: Generated for Well Correlation Analysis
"""

import os
import numpy as np
import pandas as pd
import lasio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# GPU Acceleration Support (Optional)
HAS_GPU = False
cp = None

def _init_gpu_engine():
    global HAS_GPU, cp
    import sys
    import os
    
    # Windows DLL loading fix for nvidia-* pip packages
    if sys.platform == 'win32':
        # Search site-packages for nvidia libraries (required for newer pip-installed CUDA)
        found_dlls = False
        for path in sys.path:
            # Be very thorough in searching for nvidia/bin
            if 'site-packages' in path:
                nvidia_root = os.path.join(path, 'nvidia')
                if os.path.exists(nvidia_root):
                    for root, dirs, files in os.walk(nvidia_root):
                        if 'bin' in dirs:
                            bin_path = os.path.join(root, 'bin')
                            # Check for the specific DLL cupy is looking for
                            if any(f.startswith('nvrtc64') for f in os.listdir(bin_path)):
                                try:
                                    os.add_dll_directory(bin_path)
                                    # Also add to PATH as fallback for some versions
                                    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                                    # Set CUDA_PATH to help CuPy find its own libs
                                    if 'CUDA_PATH' not in os.environ:
                                        os.environ['CUDA_PATH'] = os.path.dirname(bin_path)
                                    found_dlls = True
                                except: pass
                                
    try:
        import cupy
        # Perform a real GPU operation to test driver/DLL stability
        a = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
        b = cupy.array([4.0, 5.0, 6.0], dtype=cupy.float64)
        c = a + b
        res = cupy.asnumpy(c)
        if np.allclose(res, [5.0, 7.0, 9.0]):
            cp = cupy 
            HAS_GPU = True
        else:
            HAS_GPU = False
    except ImportError:
        HAS_GPU = False
    except Exception:
        HAS_GPU = False

_init_gpu_engine()


class LithologyClusteringAnalyzer:
    """
    Depth-wise Lithology Clustering using MRGC
    
    This class performs clustering at each depth point to identify lithology types
    based on well log responses (GR, RHOB, NPHI, DT, etc.)
    """
    
    def __init__(self, las_directory):
        """
        Initialize Lithology Clustering Analyzer
        
        Parameters:
        -----------
        las_directory : str
            Path to directory containing LAS files
        """
        self.las_directory = las_directory
        self.wells_data = {}
        self.cluster_results = {}
        self.features_scaled = None
        self.scaler = StandardScaler()
        self.graphs = {}
        self.clustered_data = None
        self._use_gpu = False  # Default to CPU for stability
        
        # Print GPU status
        if HAS_GPU:
            print(f"MRGC: GPU available but disabled by default. Enable via GUI if needed.")
        else:
            print(f"MRGC: Running in CPU-only mode (CuPy not available)")
        
    @property
    def use_gpu(self):
        return self._use_gpu
        
    @use_gpu.setter
    def use_gpu(self, value):
        # Force False if hardware/library not available
        self._use_gpu = value if HAS_GPU else False
        
    def load_well(self, well_name, curves_of_interest=None):
        """
        Load a single well LAS file
        
        Parameters:
        -----------
        well_name : str
            Name of the well (e.g., 'SBK-001')
        curves_of_interest : list, optional
            List of curve names to extract. If None, uses common curves.
        
        Returns:
        --------
        DataFrame : Well data with depth and log curves
        """
        if curves_of_interest is None:
            curves_of_interest = ['GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'SP', 'ILD', 'LLD', 'LLS']
        
        
        # Find LAS file
        las_files = [f for f in os.listdir(self.las_directory) if f.endswith('.las')]
        matching_file = None
        
        for las_file in las_files:
            # map S- to SBK- for file lookup
            search_pattern = well_name.upper().replace("S-", "SBK-")
            if search_pattern in las_file.upper():
                matching_file = las_file
                break
        
        if matching_file is None:
            raise FileNotFoundError(f"No LAS file found for well {well_name}")
        
        # Read LAS file
        las_path = os.path.join(self.las_directory, matching_file)
        las = lasio.read(las_path)
        df = las.df().reset_index()
        
        # Get depth column
        depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
        
        # Extract available curves
        available_curves = {'Depth': df[depth_col].values}
        
        for curve in curves_of_interest:
            # More robust search (matching engine.py logic)
            matching_col = next((c for c in df.columns if curve.upper() in c.upper()), None)
            if matching_col:
                available_curves[curve] = df[matching_col].values
        
        well_df = pd.DataFrame(available_curves)
        
        # Unit Conversion Logic: Always standardize to METERS
        depth_unit = las.curves[depth_col].unit.upper() if depth_col in las.curves else ""
        
        if 'FT' in depth_unit or 'F' in depth_unit:
            # Confirmed Feet -> Convert to Meters
            well_df['Depth'] = well_df['Depth'] * 0.3048
        elif 'M' not in depth_unit:
            # Ambiguous: if depths are large (e.g. > 5000), likely Feet
            if well_df['Depth'].max() > 5000:
                well_df['Depth'] = well_df['Depth'] * 0.3048
        
        self.wells_data[well_name] = well_df
        return well_df
    
    def get_las_curves(self, well_name):
        """
        Get available curves in a LAS file without loading full data
        
        Parameters:
        -----------
        well_name : str
            Name of the well
            
        Returns:
        --------
        list : List of available curve names
        """
        las_files = [f for f in os.listdir(self.las_directory) if f.endswith('.las')]
        matching_file = None
        for las_file in las_files:
            # map S- to SBK- for file lookup
            search_pattern = well_name.upper().replace("S-", "SBK-")
            if search_pattern in las_file.upper():
                matching_file = las_file
                break
        
        if matching_file is None:
            return []
            
        las_path = os.path.join(self.las_directory, matching_file)
        # Read header only for speed
        try:
            # Common depth mnemonics to exclude
            depth_mnemonics = ['DEPT', 'DEPTH', 'DEPTH_M', 'DEPTH_FT']
            las = lasio.read(las_path, ignore_data=True)
            return [c.mnemonic for c in las.curves if c.mnemonic.upper() not in depth_mnemonics]
        except:
            # Fallback
            las = lasio.read(las_path)
            depth_mnemonics = ['DEPT', 'DEPTH', 'DEPTH_M', 'DEPTH_FT']
            return [c.mnemonic for c in las.curves if c.mnemonic.upper() not in depth_mnemonics]

    def get_available_wells(self):
        """
        Get list of available wells from LAS directory
        
        Returns:
        --------
        list : List of well names
        """
        las_files = [f for f in os.listdir(self.las_directory) if f.endswith('.las')]
        well_names = []
        
        for las_file in las_files:
            # Extract well name and convert "SBK" to "S"
            well_name = las_file.replace('_wire_raw.las', '').upper()
            well_name = well_name.replace('SBK-', 'S-') if 'SBK-' in well_name else well_name
            well_names.append(well_name)
        
        return sorted(well_names)
    
    def prepare_features(self, well_name, feature_curves=None):
        """
        Prepare features for clustering from a single well
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        feature_curves : list, optional
            List of curves to use as features. If None, uses all available.
        
        Returns:
        --------
        DataFrame : Prepared features with depth index
        """
        if well_name not in self.wells_data:
            raise ValueError(f"Well {well_name} not loaded. Call load_well() first.")
        
        well_df = self.wells_data[well_name].copy()
        
        # Determine feature columns
        if feature_curves is None:
            feature_curves = [col for col in well_df.columns if col != 'Depth']
        else:
            feature_curves = [col for col in feature_curves if col in well_df.columns and col != 'Depth']
        
        # Create feature matrix
        features_df = well_df[['Depth'] + feature_curves].copy()
        
        # Remove rows with all NaN features
        features_df = features_df.dropna(subset=feature_curves, how='all')
        
        # Fill remaining NaNs with median
        for col in feature_curves:
            if features_df[col].isna().any():
                features_df[col].fillna(features_df[col].median(), inplace=True)
        
        return features_df
    
    def cluster_well(self, well_name, n_clusters=5, method='kmeans', k=7, 
                     feature_curves=None, random_state=42):
        """
        Perform lithology clustering on a single well
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        n_clusters : int
            Number of lithology clusters
        method : str
            Clustering method: 'kmeans', 'gmm', 'louvain', or 'spectral'
        k : int
            K value for k-NN graph (used in louvain/spectral methods)
        feature_curves : list, optional
            List of curves to use as features
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        DataFrame : Results with depth and cluster labels
        """
        # Prepare features
        features_df = self.prepare_features(well_name, feature_curves)
        
        # Get feature columns
        feature_cols = [col for col in features_df.columns if col != 'Depth']
        
        # Scale features
        features_array = features_df[feature_cols].values
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=random_state)
            labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'louvain':
            # Build k-NN graph
            G = self._build_knn_graph(features_scaled, k)
            
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G, random_state=random_state)
                labels = np.array([partition[i] for i in range(len(partition))])
            except ImportError:
                print("Warning: python-louvain not available. Using spectral clustering.")
                labels = self._spectral_clustering(features_scaled, n_clusters, G)
                
        elif method == 'spectral':
            G = self._build_knn_graph(features_scaled, k)
            labels = self._spectral_clustering(features_scaled, n_clusters, G)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create results dataframe
        results_df = features_df.copy()
        results_df['Cluster'] = labels
        results_df['Well'] = well_name
        
        # Store results
        self.cluster_results[well_name] = {
            'data': results_df,
            'features_scaled': features_scaled,
            'labels': labels,
            'n_clusters': len(np.unique(labels)),
            'method': method,
            'feature_curves': feature_cols
        }
        
        return results_df
    
    def _build_knn_graph(self, features_scaled, k):
        """Build k-NN graph for graph-based clustering"""
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(features_scaled)), metric='euclidean')
        nbrs.fit(features_scaled)
        distances, indices = nbrs.kneighbors(features_scaled)
        
        # Build graph
        G = nx.Graph()
        n_samples = len(features_scaled)
        
        for i in range(n_samples):
            G.add_node(i)
        
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i, 1:]):  # Skip self
                if j_idx + 1 < len(distances[i]):
                    dist = distances[i, j_idx + 1]
                    weight = 1.0 / (dist + 1e-10)
                    G.add_edge(i, j, weight=weight, distance=dist)
        
        return G
    
    def _spectral_clustering(self, features_scaled, n_clusters, G):
        """Spectral clustering fallback"""
        adj_matrix = nx.to_numpy_array(G)
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                       random_state=42)
        labels = clusterer.fit_predict(adj_matrix)
        return labels
    
    # ============================================================
    # MRGC (Multi-Resolution Graph-Based Clustering) Implementation
    # Accurate implementation based on user specifications
    # ============================================================
    
    def _build_mrgc_knn(self, features_scaled, k=None):
        """
        Build KNN structure for MRGC with optimized scikit-learn search.
        """
        n_samples = len(features_scaled)
        
        # Default k based on data size
        if k is None:
            k = min(int(np.sqrt(n_samples)) * 2, n_samples - 1, 100)
        k = max(10, min(k, n_samples - 1))
        
        # Use scikit-learn for much faster KNN search instead of O(N^2) pdist
        # Force float64 for input to ensure consistent distance precision
        features_scaled_64 = features_scaled.astype(np.float64)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
        nbrs.fit(features_scaled_64)
        dist_matrix, neighbor_indices = nbrs.kneighbors(features_scaled_64)
        
        # We need a dense rank matrix for NI computation
        # For memory efficiency with large N, we only compute ranks for the found neighbors
        # NI(x) = sum over all y where x is in y's neighbors of exp(-rank(x in y) / alpha)
        
        return {
            'dist_matrix': dist_matrix.astype(np.float64), 
            'neighbor_indices': neighbor_indices,
            'k': k,
            'n_samples': n_samples,
            'features_scaled': features_scaled_64
        }
    
    def _compute_ni(self, knn_data, alpha):
        """
        Compute Neighboring Index (NI) - Optimized version
        """
        n_samples = knn_data['n_samples']
        neighbor_indices = knn_data['neighbor_indices']
        k = knn_data['k']
        
        xp = cp if self.use_gpu else np
        
        # Use float64 for maximum precision alignment between CPU/GPU
        dtype = xp.float64
        
        # Prepare exp_ranks
        ranks = xp.arange(k + 1, dtype=dtype)
        exp_ranks = xp.exp(-ranks / alpha)
        
        ni = xp.zeros(n_samples, dtype=dtype)
        
        # Accumulate NI values using vectorized operations
        try:
            for m in range(k + 1):
                val = float(exp_ranks[m])
                idx = neighbor_indices[:, m]
                # Use proper add.at for both NumPy and CuPy
                if self.use_gpu:
                    # CuPy's scatter_add expects indices as CuPy array
                    idx_gpu = xp.asarray(idx) if not isinstance(idx, xp.ndarray) else idx
                    xp.add.at(ni, idx_gpu, val)
                else:
                    np.add.at(ni, idx, val)
            
            ni = ni - 1.0  # Subtract self (rank 0)
            
            # Ensure return is always NumPy array
            if self.use_gpu:
                return cp.asnumpy(ni).astype(np.float64)
            return ni.astype(np.float64)
            
        except Exception as e:
            # GPU operation failed - fall back to CPU
            if self.use_gpu:
                print(f"GPU _compute_ni failed: {e}. Falling back to CPU.")
                self._use_gpu = False  # Disable GPU for remainder
                # Recompute with CPU
                return self._compute_ni(knn_data, alpha)
    
    def _compute_kri(self, ni, knn_data):
        """
        Compute Kernel Representative Index (KRI) - Fully Vectorized
        """
        xp = cp if self.use_gpu else np
        n_samples = knn_data['n_samples']
        k = knn_data['k']
        dtype = xp.float64
        
        try:
            dist_matrix = xp.asarray(knn_data['dist_matrix'], dtype=dtype)
            neighbor_indices = xp.asarray(knn_data['neighbor_indices'], dtype=int)
            ni_xp = xp.asarray(ni, dtype=dtype)
            
            # We need to find the nearest neighbor with higher NI for each point
            # ni_neighbors shape: (n_samples, k)
            ni_neighbors = ni_xp[neighbor_indices[:, 1:]]
            
            # Find points where neighbor_ni > point_ni
            # higher_mask shape: (n_samples, k)
            higher_mask = ni_neighbors > ni_xp[:, xp.newaxis]
            
            # Results arrays
            kri = xp.zeros(n_samples, dtype=dtype)
            higher_density_index = xp.zeros(n_samples, dtype=int)
            dist_to_higher = xp.zeros(n_samples, dtype=dtype)
            rank_to_higher = xp.zeros(n_samples, dtype=dtype)
            
            # Points with at least one higher neighbor
            has_higher = xp.any(higher_mask, axis=1)
            idx_has_higher = xp.where(has_higher)[0]
            
            if len(idx_has_higher) > 0:
                # For each row, find the first (closest) higher-density neighbor
                # argmax gives first True (due to max on boolean being 1)
                first_higher_rank_idx = xp.argmax(higher_mask[idx_has_higher], axis=1)
                
                # Extract corresponding neighbor indices
                y_indices = neighbor_indices[idx_has_higher, first_higher_rank_idx + 1]
                
                M_xy = (first_higher_rank_idx + 1).astype(dtype)
                D_xy = dist_matrix[idx_has_higher, first_higher_rank_idx + 1]
                
                kri[idx_has_higher] = ni_xp[idx_has_higher] * M_xy * D_xy
                higher_density_index[idx_has_higher] = y_indices
                dist_to_higher[idx_has_higher] = D_xy
                rank_to_higher[idx_has_higher] = M_xy
                
            # Global peaks (no higher neighbor within k neighbors)
            idx_peak = xp.where(~has_higher)[0]
            if len(idx_peak) > 0:
                # For very large N, we use a large constant instead of a full search
                target_rank = float(n_samples)
                max_dists = dist_matrix[idx_peak, -1]
                
                kri[idx_peak] = ni_xp[idx_peak] * target_rank * max_dists
                higher_density_index[idx_peak] = idx_peak
                dist_to_higher[idx_peak] = max_dists
                rank_to_higher[idx_peak] = target_rank

            if self.use_gpu:
                return (cp.asnumpy(kri).astype(np.float64), 
                        cp.asnumpy(higher_density_index), 
                        cp.asnumpy(dist_to_higher).astype(np.float64), 
                        cp.asnumpy(rank_to_higher).astype(np.float64))
            return kri, higher_density_index, dist_to_higher, rank_to_higher
            
        except Exception as e:
            # GPU operation failed - fall back to CPU
            if self.use_gpu:
                print(f"GPU _compute_kri failed: {e}. Falling back to CPU.")
                self._use_gpu = False  # Disable GPU for remainder
                # Recompute with CPU
                return self._compute_kri(ni, knn_data)
            else:
                raise  # If CPU also failed, propagate error
    
    def _detect_kernels(self, kri, ni, n_clusters=None):
        """
        Detect cluster kernels based on KRI values.
        
        Parameters:
        -----------
        kri : np.ndarray
            Kernel Representative Index values
        ni : np.ndarray
            Neighboring Index values
        n_clusters : int, optional
            Number of clusters. If None, auto-detect using break points
            
        Returns:
        --------
        np.ndarray : Indices of kernel points
        """
        n_samples = len(kri)
        
        if n_clusters is not None:
            # Use specified number of clusters
            n_kernels = min(n_clusters, n_samples)
            kernel_indices = np.argsort(kri)[-n_kernels:]
        else:
            # Auto-detect using break points in sorted KRI curve
            sorted_kri = np.sort(kri)[::-1]  # Descending
            
            # Calculate differences between consecutive KRI values
            if len(sorted_kri) > 1:
                diffs = np.abs(np.diff(sorted_kri))
                
                # Find significant break point (large drop in KRI)
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                threshold = mean_diff + 1.5 * std_diff
                
                # Find first significant break
                significant_breaks = np.where(diffs > threshold)[0]
                
                if len(significant_breaks) > 0:
                    n_kernels = significant_breaks[0] + 1
                else:
                    # Fallback: use statistical threshold on KRI
                    kri_threshold = np.mean(kri) + 1.5 * np.std(kri)
                    n_kernels = max(2, np.sum(kri > kri_threshold))
            else:
                n_kernels = 1
            
            # Limit number of kernels
            n_kernels = max(2, min(n_kernels, 15, n_samples // 5))
            kernel_indices = np.argsort(kri)[-n_kernels:]
        
        # Sort kernels by NI (descending) - highest NI = primary kernel
        kernel_indices = kernel_indices[np.argsort(ni[kernel_indices])[::-1]]
        
        return kernel_indices
    
    def _watershed_merge(self, ni, kernels, higher_density_index, knn_data):
        """
        Assign points to clusters using watershed-like merging.
        
        Algorithm:
        1. Assign kernel points first
        2. Process remaining points in descending NI order
        3. Each point inherits cluster from its higher-density neighbor
        4. Merge shallow valleys (modes separated by small NI dips)
        
        Parameters:
        -----------
        ni : np.ndarray
            Neighboring Index values
        kernels : np.ndarray
            Indices of kernel points
        higher_density_index : np.ndarray
            Index of higher-density neighbor for each point
        knn_data : dict
            KNN structure
            
        Returns:
        --------
        np.ndarray : Cluster labels for each point
        """
        n_samples = knn_data['n_samples']
        labels = np.full(n_samples, -1, dtype=int)
        
        # Step 1: Assign kernels
        for cluster_id, kernel_idx in enumerate(kernels):
            labels[kernel_idx] = cluster_id
        
        # Step 2: Process points in descending NI order
        sorted_indices = np.argsort(ni)[::-1]
        
        for idx in sorted_indices:
            if labels[idx] != -1:
                continue  # Already assigned (kernel)
            
            # Follow chain to higher density until we reach a labeled point
            current = idx
            chain = [current]
            
            while labels[current] == -1:
                next_point = higher_density_index[current]
                if next_point == current:
                    # Dead end (shouldn't happen if kernels are set correctly)
                    break
                chain.append(next_point)
                current = next_point
            
            # Assign all points in chain to the label found
            if labels[current] != -1:
                for point in chain:
                    labels[point] = labels[current]
        
        # Step 3: Handle any remaining unlabeled points
        unlabeled = np.where(labels == -1)[0]
        if len(unlabeled) > 0 and len(kernels) > 0:
            features_scaled = knn_data['features_scaled']
            # Re-calculate distances to kernels for these few points
            kernel_feats = features_scaled[kernels]
            for idx in unlabeled:
                point_feat = features_scaled[idx]
                dists = np.linalg.norm(kernel_feats - point_feat, axis=1)
                closest_kernel = kernels[np.argmin(dists)]
                labels[idx] = labels[closest_kernel]
        
        return labels

    def cluster_mrgc_petro(self, well_name, alpha=10.0, n_clusters_manual=None, 
                          feature_curves=None, k=None):
        """
        Perform MRGC (Multi-Resolution Graph-Based Clustering) on a single well.
        
        Algorithm steps:
        1. Build KNN structure with mutual neighbor relationships
        2. Compute Neighboring Index (NI) = sum of exp(-rank/alpha)
        3. Compute Kernel Representative Index (KRI) = NI * M * D
        4. Detect cluster kernels (highest KRI points)
        5. Assign points using watershed merging
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        alpha : float
            Smoothing parameter (Resolution). Larger = fewer clusters
        n_clusters_manual : int, optional
            If set, forces the number of clusters
        feature_curves : list, optional
            Features to use for clustering
        k : int, optional
            Number of neighbors for KNN. If None, auto-calculate
            
        Returns:
        --------
        DataFrame : Results with Depth, features, Cluster, NI, KRI
        """
        # Prepare features
        features_df = self.prepare_features(well_name, feature_curves)
        feature_cols = [col for col in features_df.columns if col != 'Depth']
        # FORCE float64 for bit-perfect consistency between CPU and GPU
        features_array = features_df[feature_cols].values.astype(np.float64)
        features_scaled = self.scaler.fit_transform(features_array).astype(np.float64)
        
        try:
            # Step 1: Build KNN structure
            knn_data = self._build_mrgc_knn(features_scaled, k=k)
            
            # Step 2: Compute Neighboring Index (NI)
            ni = self._compute_ni(knn_data, alpha)
            
            # Step 3: Compute Kernel Representative Index (KRI)
            kri, higher_density_index, dist_to_higher, rank_to_higher = self._compute_kri(ni, knn_data)
            
            # Step 4: Detect cluster kernels
            kernels = self._detect_kernels(kri, ni, n_clusters=n_clusters_manual)
            
            # Step 5: Watershed merging
            labels = self._watershed_merge(ni, kernels, higher_density_index, knn_data)
            
        except Exception as e:
            print(f"MRGC Error with GPU={self.use_gpu}: {str(e)}")
            if self.use_gpu:
                print("Falling back to CPU mode...")
                self._use_gpu = False  # Temporarily disable GPU
                try:
                    # Retry with CPU
                    knn_data = self._build_mrgc_knn(features_scaled, k=k)
                    ni = self._compute_ni(knn_data, alpha)
                    kri, higher_density_index, dist_to_higher, rank_to_higher = self._compute_kri(ni, knn_data)
                    kernels = self._detect_kernels(kri, ni, n_clusters=n_clusters_manual)
                    labels = self._watershed_merge(ni, kernels, higher_density_index, knn_data)
                    self._use_gpu = True  # Restore GPU setting
                except Exception as e2:
                    print(f"CPU fallback also failed: {str(e2)}")
                    raise
            else:
                raise
        
        # Create results dataframe
        results_df = features_df.copy()
        # Ensure results use float64 for precision items and int for labels
        results_df['Cluster'] = labels.astype(int)
        results_df['NI'] = ni.astype(np.float64)
        results_df['KRI'] = kri.astype(np.float64)
        
        # Store results with additional metadata
        self.cluster_results[well_name] = {
            'data': results_df,
            'features_scaled': features_scaled,
            'labels': labels,
            'n_clusters': len(np.unique(labels)),
            'method': 'mrgc_petro',
            'feature_curves': feature_cols,
            'kernels': kernels,
            'alpha': alpha
        }
        
        return results_df

    def cluster_mrgc_multi_well(self, well_names, alpha=10.0, n_clusters_manual=None, 
                                feature_curves=None, k=None):
        """
        Perform MRGC on multiple wells combined.
        
        Data from all wells is combined, clustered together, then split back
        to maintain consistent cluster assignments across wells.
        
        Parameters:
        -----------
        well_names : list
            List of well names to cluster together
        alpha : float
            Smoothing parameter (Resolution)
        n_clusters_manual : int, optional
            Number of clusters. If None, auto-detect
        feature_curves : list, optional
            Features to use. Uses intersection of available curves
        k : int, optional
            Number of neighbors for KNN
            
        Returns:
        --------
        dict : {well_name: DataFrame} with clustering results for each well
        """
        # Step 1: Collect and prepare data for each well
        all_features = []
        for name in well_names:
            if name not in self.wells_data:
                self.load_well(name)
            
            f_df = self.prepare_features(name, feature_curves)
            f_df['Well'] = name # Add identifier for splitting back later
            all_features.append(f_df)
            
        if not all_features:
            return {}
            
        # Step 2: Combine and Clean
        combined_df = pd.concat(all_features, axis=0).reset_index(drop=True)
        feature_cols = [col for col in combined_df.columns if col not in ['Depth', 'Well']]
        
        # Remove any remaining NaNs (crucial for MRGC math)
        combined_df = combined_df.dropna(subset=feature_cols).reset_index(drop=True)
        
        # Recalculate Boundaries AFTER dropna so index ranges are accurate
        well_boundaries = []
        last_idx = 0
        for name in well_names:
            count = len(combined_df[combined_df['Well'] == name])
            well_boundaries.append((name, last_idx, last_idx + count))
            last_idx += count
        
        # Step 3: Run MRGC on combined data
        # FORCE float64 for multi-well consistency
        features_array = combined_df[feature_cols].values.astype(np.float64)
        features_scaled = self.scaler.fit_transform(features_array).astype(np.float64)
        
        # Build KNN structure
        knn_data = self._build_mrgc_knn(features_scaled, k=k)
        
        # Compute NI
        ni = self._compute_ni(knn_data, alpha)
        
        # Compute KRI
        kri, higher_density_index, dist_to_higher, rank_to_higher = self._compute_kri(ni, knn_data)
        
        # Detect kernels
        kernels = self._detect_kernels(kri, ni, n_clusters=n_clusters_manual)
        
        # Watershed merging
        labels = self._watershed_merge(ni, kernels, higher_density_index, knn_data)
        
        # Step 4: Split back and store individual results
        multi_results = {}
        for name, start, end in well_boundaries:
            well_res_df = combined_df.iloc[start:end].copy()
            well_res_df['Cluster'] = labels[start:end]
            well_res_df['NI'] = ni[start:end]
            well_res_df['KRI'] = kri[start:end]
            
            self.cluster_results[name] = {
                'data': well_res_df,
                'features_scaled': features_scaled[start:end],
                'labels': labels[start:end],
                'n_clusters': len(np.unique(labels)),
                'method': 'mrgc_petro_multi',
                'feature_curves': feature_cols,
                'kernels': kernels,
                'alpha': alpha
            }
            multi_results[name] = well_res_df
            
        return multi_results
    
    def suggest_alpha(self, well_name, feature_curves=None):
        """
        Suggest a recommended smoothing parameter (alpha) based on data size.
        Formula: suggest = sqrt(N) / 2, capped between 5.0 and 50.0.
        """
        try:
            if well_name not in self.wells_data:
                self.load_well(well_name)
            features_df = self.prepare_features(well_name, feature_curves)
            n_samples = len(features_df)
            if n_samples == 0: return 10.0
            suggested = np.sqrt(n_samples) / 2.0
            return float(np.clip(suggested, 5.0, 50.0))
        except Exception:
            return 10.0

    def multi_resolution_analysis(self, well_name, k_values=None, n_clusters=5, 
                                   method='louvain', feature_curves=None):
        """
        Perform clustering with different k values to analyze stability
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        k_values : list, optional
            List of k values. If None, uses [3, 5, 7, 10, 15]
        n_clusters : int
            Number of clusters
        method : str
            Clustering method
        feature_curves : list, optional
            Feature curves to use
        
        Returns:
        --------
        dict : Results for each k value
        """
        if k_values is None:
            k_values = [3, 5, 7, 10, 15]
        
        results = {}
        
        for k in k_values:
            result_df = self.cluster_well(well_name, n_clusters=n_clusters, 
                                         method=method, k=k, feature_curves=feature_curves)
            
            features_scaled = self.cluster_results[well_name]['features_scaled']
            labels = self.cluster_results[well_name]['labels']
            
            # Compute silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(features_scaled, labels)
            else:
                sil_score = 0.0
            
            results[k] = {
                'labels': labels,
                'n_clusters': len(np.unique(labels)),
                'silhouette_score': sil_score
            }
        
        return results
    
    def compute_silhouette(self, well_name):
        """
        Compute silhouette analysis for clustering results
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        
        Returns:
        --------
        dict : Silhouette scores and metrics
        """
        if well_name not in self.cluster_results:
            raise ValueError(f"No clustering results for {well_name}")
        
        features_scaled = self.cluster_results[well_name]['features_scaled']
        labels = self.cluster_results[well_name]['labels']
        
        # Compute scores
        silhouette_avg = silhouette_score(features_scaled, labels)
        sample_silhouette_values = silhouette_samples(features_scaled, labels)
        
        return {
            'average': silhouette_avg,
            'samples': sample_silhouette_values,
            'labels': labels
        }
    
    def get_cluster_statistics(self, well_name):
        """
        Compute statistics for each cluster
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        
        Returns:
        --------
        DataFrame : Statistics per cluster
        """
        if well_name not in self.cluster_results:
            raise ValueError(f"No clustering results for {well_name}")
        
        data = self.cluster_results[well_name]['data']
        feature_curves = self.cluster_results[well_name]['feature_curves']
        
        # Group by cluster and compute stats
        stats_list = []
        
        for cluster_id in sorted(data['Cluster'].unique()):
            cluster_data = data[data['Cluster'] == cluster_id]
            
            stats = {
                'Cluster': cluster_id,
                'Count': len(cluster_data),
                'Depth_Min': cluster_data['DEPTH'].min(),
                'Depth_Max': cluster_data['DEPTH'].max(),
            }
            
            # Add mean values for each feature
            for curve in feature_curves:
                stats[f'{curve}_mean'] = cluster_data[curve].mean()
                stats[f'{curve}_std'] = cluster_data[curve].std()
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def compute_pca(self, well_name, n_components=2):
        """
        Compute PCA projection of feature space
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        n_components : int
            Number of PCA components
        
        Returns:
        --------
        dict : PCA coordinates and explained variance
        """
        if well_name not in self.cluster_results:
            raise ValueError(f"No clustering results for {well_name}")
        
        features_scaled = self.cluster_results[well_name]['features_scaled']
        labels = self.cluster_results[well_name]['labels']
        
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(features_scaled)
        
        return {
            'coordinates': coords,
            'labels': labels,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def compute_tsne(self, well_name, n_components=2):
        """
        Compute t-SNE projection of feature space
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        n_components : int
            Number of t-SNE components
        
        Returns:
        --------
        dict : t-SNE coordinates and labels
        """
        if well_name not in self.cluster_results:
            raise ValueError(f"No clustering results for {well_name}")
        
        features_scaled = self.cluster_results[well_name]['features_scaled']
        labels = self.cluster_results[well_name]['labels']
        
        # Adjust perplexity based on sample size
        n_samples = len(features_scaled)
        perplexity = min(30, max(5, n_samples // 3))
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(features_scaled)
        
        return {
            'coordinates': coords,
            'labels': labels
        }
    
    def export_results(self, well_name, output_path):
        """
        Export clustering results to CSV
        
        Parameters:
        -----------
        well_name : str
            Name of the well
        output_path : str
            Path to save CSV file
        """
        if well_name not in self.cluster_results:
            raise ValueError(f"No clustering results for {well_name}")
        
        data = self.cluster_results[well_name]['data']
        data.to_csv(output_path, index=False)
        
        return output_path
