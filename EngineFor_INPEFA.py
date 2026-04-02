"""
EngineFor_INPEFA.py
Integrated INPEFA algorithm engine for Well Correlation application.
Includes core l1 trend filtering logic to eliminate external package dependencies.
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
from cvxopt import matrix, spmatrix, solvers
from spectrum.burg import _arburg2

# Set cvxopt to quiet mode
solvers.options['show_progress'] = False


def _l1tf_core(y, lambdaaa):
    """
    Finds the solution of the l1 trend estimation problem.
    Core logic from PyNPEFA.l1tf
    """
    n = len(y)
    m = n - 2

    # Convert array y to cvxopt.spmatrix
    y_mat = spmatrix(y, range(n), [0] * n, tc='d')
    
    # Create second order difference matrix
    D = spmatrix([1, -2, 1] * m,
                 [j for i in range(m) for j in [i] * 3],
                 [j for i in range(m) for j in [i, i + 1, i + 2]], tc='d')

    # Create P and q
    P = D * D.T
    q = D * y_mat * (-1)
    q = matrix(q)
    
    # Create G and h
    G = spmatrix([1] * m + [-1] * m, range(2 * m), 2 * [*range(m)])
    h = matrix(lambdaaa, (2 * m, 1))
    
    # Solve the QP problem
    res = solvers.qp(P, q, G, h)
    sol = y_mat - D.T * res['x']
    return sol


def _l1tf_max_lambda(y):
    """
    Returns an upperbound of lambda.
    Core logic from PyNPEFA.l1tf_lm
    """
    n = len(y)
    m = n - 2
    
    y_csr = csr_matrix((y, (np.array([*range(n)]), np.array([0] * n))))
    D = diags([1, -2, 1], [0, 1, 2], shape=(m, n))
    DDt = D * D.T
    Dy = D * y_csr

    return norm(spsolve(DDt, Dy), np.inf)


def _PyNPEFA_Local(y, x):
    """
    Local implementation of the PyNPEFA main function.
    Eliminates dependency on the PyNPEFA external folder.
    """
    y_val = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)

    # Set maximum regularization parameter
    lambdamax = _l1tf_max_lambda(y_val)
    z = {}
    z['0'] = matrix(y_val)

    # l1 trend filtering - calculate 9 levels of smoothing
    for i in range(1, 10):
        z[str(i)] = _l1tf_core(z[str(i - 1)], 10**(-10 + i) * lambdamax)

    # Set trend filter for long, mid, short, and shorter term
    fy = {}
    fy['1'] = z['0'] - (z['1'] + z['2'] + z['3'] + z['4'] + z['5'] + z['6'] + z['7'] + z['8']) / 8.0
    fy['2'] = z['0'] - (z['1'] + z['2'] + z['3'] + z['4'] + z['5'] + z['6']) / 6.0
    fy['3'] = z['0'] - (z['1'] + z['2'] + z['3'] + z['4'] + z['5']) / 5.0
    fy['4'] = z['0'] - (z['1'] + z['2'] + z['3'] + z['4']) / 4.0

    # Compute Burg Filter, Prediction Error, and Integrated Prediction Error
    ipfy = {}
    ipfy['OG'] = y_val
    for j in range(1, 5):
        current_fy = np.array(fy[str(j)]).flatten()
        
        # Burg's AR coeff - dynamic order based on length to prevent spectrum error
        order = min(32, len(current_fy) - 1)
        if order < 1:
            ipfy[str(j)] = np.zeros(len(y_val))
            continue
            
        bffy = _arburg2(current_fy, order)[0].real
        
        # PEFA
        pffy = signal.convolve(current_fy, np.reshape(bffy, (len(bffy), 1)).flatten(), mode='same')
        
        # INPEFA
        iipfy = np.cumsum(pffy[::-1])[::-1]
        
        # Normalize to -1 <= INPEFA <= 1
        max_abs = max(abs(iipfy))
        if max_abs > 0:
            ipfy[str(j)] = iipfy / max_abs
        else:
            ipfy[str(j)] = iipfy

    return ipfy


class INPEFAEngine:
    """
    Core logic for INPEFA (Indicators of Net-to-Gross from Electrofacies Analysis).
    """
    
    @staticmethod
    def run_inpefa(y_series, x_series, term="long"):
        """
        Runs the INPEFA algorithm on a data series for a specific term.
        Uses the integrated local core logic.
        """
        # 1. Handle Input
        y = y_series.values if isinstance(y_series, pd.Series) else y_series
        x = x_series.values if isinstance(x_series, pd.Series) else x_series

        # 2. Check for constant data or insufficient points
        if len(y) < 5 or np.all(y == y[0]):
            return np.zeros(len(y))

        term_map = {"long": "1", "mid": "2", "short": "3", "shorter": "4"}
        key = term_map.get(term, "1")
        
        try:
            # Call local integrated logic
            results = _PyNPEFA_Local(pd.Series(y), x)
            inpefa = np.array(results[key]).flatten()
            return inpefa
        except Exception as e:
            print(f"DEBUG: Integrated INPEFA failed: {e}")
            return np.zeros(len(y))
