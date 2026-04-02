
import numpy as np
import pandas as pd
from EngineFor_INPEFA import INPEFAEngine

def test_inpefa_run():
    # Create dummy data
    n = 100
    depth = np.linspace(0, 100, n)
    gr = 50 + 10 * np.sin(0.1 * depth) + np.random.normal(0, 1, n)
    
    print("Running INPEFA 'long' term...")
    try:
        res = INPEFAEngine.run_inpefa(gr, depth, term="long")
        print(f"Success! Result shape: {res.shape}")
        print(f"Max value: {np.max(res)}, Min value: {np.min(res)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inpefa_run()
