import os
import sys
import traceback

def setup_cuda_environment():
    """
    Runtime hook to help frozen applications find bundled NVIDIA DLLs.
    Logging to a file for debugging in windowed apps.
    """
    if not getattr(sys, 'frozen', False):
        return

    # Log to a file in the same directory as the exe
    log_path = os.path.join(os.path.dirname(sys.executable), "cuda_debug_log.txt")
    
    with open(log_path, "w") as f:
        f.write("CUDA Runtime Hook Started\n")
        f.write(f"sys.executable: {sys.executable}\n")
        f.write(f"sys._MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}\n")
        
        base_path = sys._MEIPASS
        nvidia_dir = os.path.join(base_path, 'nvidia')
        f.write(f"NVIDIA Dir expected at: {nvidia_dir}\n")
        
        if os.path.exists(nvidia_dir):
            f.write("NVIDIA Dir found!\n")
            # Recursively look for 'bin' directories containing DLLs
            for root, dirs, files in os.walk(nvidia_dir):
                f.write(f"Searching in: {root}\n")
                if 'bin' in dirs:
                    bin_path = os.path.join(root, 'bin')
                    f.write(f"Found bin path: {bin_path}\n")
                    
                    # Register the DLL directory
                    if hasattr(os, 'add_dll_directory'):
                        try:
                            os.add_dll_directory(bin_path)
                            f.write(f"Successfully added DLL directory: {bin_path}\n")
                        except Exception as e:
                            f.write(f"Failed to add DLL directory {bin_path}: {e}\n")
                    
                    # Also add to PATH
                    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                    
                    # Check for NVRTC
                    if any(f_.startswith('nvrtc64') for f_ in os.listdir(bin_path)):
                        f.write(f"NVRTC DLL found in {bin_path}\n")
                        if 'CUDA_PATH' not in os.environ:
                            os.environ['CUDA_PATH'] = os.path.dirname(bin_path)
                            f.write(f"Set CUDA_PATH to: {os.environ['CUDA_PATH']}\n")
        else:
            f.write("NVIDIA Dir NOT found!\n")
            # List all files in _MEIPASS to see where it went
            f.write("Files in _MEIPASS root:\n")
            try:
                f.write(", ".join(os.listdir(base_path)) + "\n")
            except: pass

if __name__ == '__main__':
    try:
        setup_cuda_environment()
    except Exception as e:
        # Emergency log
        log_path = os.path.join(os.path.dirname(sys.executable), "cuda_error_log.txt")
        with open(log_path, "w") as f:
            f.write(f"CRITICAL ERROR in setup_cuda_environment: {e}\n")
            f.write(traceback.format_exc())
