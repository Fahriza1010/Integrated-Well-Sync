# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import sys
import site

# Robustly find all nvidia site-packages directories to bundle DLLs
nvidia_folders = []
paths_to_check = sys.path + site.getsitepackages()
if hasattr(site, 'getusersitepackages'):
    paths_to_check.append(site.getusersitepackages())

# Add global site-packages and known locations explicitly
explicit_paths = [
    r"C:\path\to\your\python\site-packages\nvidia",
    r"path/to/your/venv/site-packages/nvidia",
    r"path/to/your/another_venv/site-packages/nvidia"
]

for ep in explicit_paths:
    if os.path.exists(ep) and ep not in paths_to_check:
        paths_to_check.append(ep)

seen = set()
for path in paths_to_check:
    if not path or path in seen: continue
    seen.add(path)
    
    # If the path itself is 'nvidia' (from our explicit list)
    if os.path.basename(path) == 'nvidia' and os.path.exists(path):
        if path not in nvidia_folders: nvidia_folders.append(path)
    else:
        # Check for 'nvidia' subfolder
        test_path = os.path.join(path, 'nvidia')
        if os.path.exists(test_path):
            if test_path not in nvidia_folders: nvidia_folders.append(test_path)

datas = []
datas += collect_data_files('spectrum')

# Bundle only the DLLs from found nvidia components to keep the size manageable
for nf in nvidia_folders:
    # Recursively find 'bin' folders
    for root, dirs, files in os.walk(nf):
        if 'bin' in dirs:
            bin_path = os.path.join(root, 'bin')
            # Bundle each bin's content to its relative path under 'nvidia'
            rel_path = os.path.relpath(bin_path, os.path.dirname(nf))
            datas.append((bin_path, rel_path))

# Collect all submodules for robust importing
hidden_imports = collect_submodules('cupy') + collect_submodules('xgboost')
hidden_imports += ['numpy', 'pandas', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils']

# Explicitly add some dependencies that might be missed
binaries = []

a = Analysis(
    ['Integrated Well Sync.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_cuda.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    # Collect all binaries found in analysis and merge
    a.binaries,
    a.datas,
    [],
    name='Integrated Well Sync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
