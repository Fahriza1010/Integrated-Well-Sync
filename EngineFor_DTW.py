"""
EngineFor_DTW.py
DTW (Dynamic Time Warping) algorithm engine for Well Correlation application
"""
import numpy as np


class DTWEngine:
    """
    Core logic for Dynamic Time Warping with Sakoe-Chiba constraint.
    """
    
    @staticmethod
    def dtw_with_path(x, y, w):
        """
        DTW implementation with Sakoe-Chiba constraint, return cost, pi, pj, and Dmat.
        Matches WellCorrelationWithINPEFAandDTW.ipynb logic.
        
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
        n, m = len(x), len(y)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0

        for i in range(1, n + 1):
            jmin = max(1, i - w)
            jmax = min(m, i + w)
            for j in range(jmin, jmax + 1):
                cost = abs(x[i - 1] - y[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        # Traceback
        pi, pj = [], []
        i, j = n, m
        while i > 0 and j > 0:
            pi.append(i - 1)
            pj.append(j - 1)
            # Prioritize diagonal step (i-1, j-1) as per notebook
            step = min(
                (D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j],     i - 1, j),
                (D[i, j - 1],     i,     j - 1),
                key=lambda t: t[0]
            )
            i, j = step[1], step[2]
        
        return D[n, m], np.array(pi[::-1]), np.array(pj[::-1]), D[1:, 1:]

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
        xi = x[::downsample]
        yi = y[::downsample]
        
        max_wi = len(xi) // 2
        step = max(1, max_wi // 25)
        # Notebook starts from 0, not 2
        windows = np.arange(0, max_wi + 1, step)
        
        def dtw_cost_only(a, b, w):
            na, mb = len(a), len(b)
            Di = np.full((na + 1, mb + 1), np.inf)
            Di[0, 0] = 0.0
            for r in range(1, na + 1):
                cmin = max(1, r - w)
                cmax = min(mb, r + w)
                for c in range(cmin, cmax + 1):
                    cst = abs(a[r - 1] - b[c - 1])
                    Di[r, c] = cst + min(Di[r - 1, c], Di[r, c - 1], Di[r - 1, c - 1])
            return Di[na, mb]

        # Optimize: Pre-calculate costs to avoid repeated work in the loop
        costs = []
        for w in windows:
            # We can't easily vectorize the whole DTW O(NM), 
            # but we can ensure the cost-only version is as tight as possible.
            costs.append(dtw_cost_only(xi, yi, int(w)))
        
        costs = np.array(costs)
        
        # Matches notebook: deriv = np.gradient(costs, windows)
        if len(costs) > 1:
            deriv = np.gradient(costs, windows)
            # Notebook threshold: np.percentile(np.abs(deriv), 25)
            threshold = np.percentile(np.abs(deriv), 25)
            
            plateau = np.where(np.abs(deriv) <= threshold)[0]
            chosen_idx = plateau[0] if len(plateau) else len(windows) // 3
        else:
            chosen_idx = 0
        
        # Consistent with notebook: chosen_w = chosen_w_ds * down
        return int(windows[chosen_idx]) * downsample

    @staticmethod
    def dtw_sectional(grid, log_A, log_B, ref_boundaries, off_boundaries, downsample=4):
        """
        Asymmetric section-based DTW.

        Pairs each ref-well depth section with the corresponding offset-well depth
        section, based on each well's own Top Sand marker depths. Because the two
        wells may have markers at *different* depths, the paired sections can have
        *different lengths* – which is exactly what DTW is built for.

        Section construction example
        ----------------------------
        overlap : [100, 1000]
        ref markers inside overlap : [110, 500, 900]   (sorted)
        off markers inside overlap : [115, 560, 910]   (sorted)

        Resulting section pairs (ref_slice → off_slice):
          0. [100 → 110]  correlated with  [100 → 115]
          1. [110 → 500]  correlated with  [115 → 560]
          2. [500 → 900]  correlated with  [560 → 910]
          3. [900 → 1000] correlated with  [910 → 1000]

        If the marker counts differ, the shorter list determines how many
        intermediate breakpoints are used; any extra markers are ignored.

        Parameters
        ----------
        grid : np.ndarray
            Shared 1-D depth grid (meters) covering the overlap.
        log_A : np.ndarray
            INPEFA for reference well, sampled at `grid` depths.
        log_B : np.ndarray
            INPEFA for offset well, sampled at `grid` depths.
        ref_boundaries : list of float
            Reference well's Top Sand marker depths (meters, within overlap).
        off_boundaries : list of float
            Offset well's Top Sand marker depths (meters, within overlap).
        downsample : int
            Downsampling factor for `get_recommended_window`.

        Returns
        -------
        tuple : (total_cost, pi, pj)
            total_cost : float       – summed DTW cost over all sections
            pi         : np.ndarray  – global ref-grid indices for corr path
            pj         : np.ndarray  – global off-grid indices for corr path
        """
        n = len(grid)
        g_min, g_max = float(grid[0]), float(grid[-1])

        def _snap(depth):
            """Nearest grid index for a given depth."""
            return int(np.argmin(np.abs(grid - depth)))

        # --- Build independent boundary index lists --------------------------
        # Filter each well's markers to those strictly inside the overlap range
        valid_ref = sorted([d for d in ref_boundaries if g_min < d < g_max])
        valid_off = sorted([d for d in off_boundaries if g_min < d < g_max])

        # Snap marker depths to grid indices for each well independently
        ref_bnd_idx = [_snap(d) for d in valid_ref]
        off_bnd_idx = [_snap(d) for d in valid_off]

        # Remove edge indices and deduplicate
        ref_bnd_idx = sorted(set(i for i in ref_bnd_idx if 0 < i < n - 1))
        off_bnd_idx = sorted(set(i for i in off_bnd_idx if 0 < i < n - 1))

        # Match pairwise: if counts differ, use the shorter list's length
        # Extra markers on the longer side are ignored (they'd be in the last section)
        n_breaks = min(len(ref_bnd_idx), len(off_bnd_idx))
        ref_bnd_idx = ref_bnd_idx[:n_breaks]
        off_bnd_idx = off_bnd_idx[:n_breaks]

        # --- Fallback: no valid paired boundaries → single full-range DTW ----
        if n_breaks == 0:
            rec_w = DTWEngine.get_recommended_window(log_A, log_B, downsample=downsample)
            w = max(1, rec_w)
            cost, pi, pj, _ = DTWEngine.dtw_with_path(log_A, log_B, w)
            return cost, pi, pj

        # --- Build section pairs (start_a, end_a, start_b, end_b) -----------
        # ref sections:  0→r0,  r0→r1,  r1→r2, ..., rN→(n-1)
        # off sections:  0→o0,  o0→o1,  o1→o2, ..., oN→(n-1)
        ref_starts = [0]          + ref_bnd_idx
        ref_ends   = ref_bnd_idx  + [n - 1]
        off_starts = [0]          + off_bnd_idx
        off_ends   = off_bnd_idx  + [n - 1]

        sections = list(zip(ref_starts, ref_ends, off_starts, off_ends))

        # --- Run DTW per section --------------------------------------------
        all_pi = []
        all_pj = []
        total_cost = 0.0

        for s_a, e_a, s_b, e_b in sections:
            seg_A = log_A[s_a: e_a + 1]   # ref segment  (length may differ)
            seg_B = log_B[s_b: e_b + 1]   # off segment

            if len(seg_A) < 2 or len(seg_B) < 2:
                continue  # skip degenerate sections

            # Per-section auto-recommended Sakoe-Chiba window
            rec_w = DTWEngine.get_recommended_window(seg_A, seg_B, downsample=downsample)
            w = max(1, rec_w)

            cost_sec, pi_sec, pj_sec, _ = DTWEngine.dtw_with_path(seg_A, seg_B, w)

            if np.isinf(cost_sec):
                # Retry with unconstrained window for this section
                w_full = max(len(seg_A), len(seg_B))
                cost_sec, pi_sec, pj_sec, _ = DTWEngine.dtw_with_path(seg_A, seg_B, w_full)

            # Offset section-local indices to global grid coordinates
            # CRITICAL: ref path offsets by s_a, off path offsets by s_b (different!)
            all_pi.extend((pi_sec + s_a).tolist())
            all_pj.extend((pj_sec + s_b).tolist())
            total_cost += cost_sec if not np.isinf(cost_sec) else 0.0

        if not all_pi:
            # Complete fallback
            rec_w = DTWEngine.get_recommended_window(log_A, log_B, downsample=downsample)
            w = max(1, rec_w)
            total_cost, pi_full, pj_full, _ = DTWEngine.dtw_with_path(log_A, log_B, w)
            return total_cost, pi_full, pj_full

        return total_cost, np.array(all_pi), np.array(all_pj)
