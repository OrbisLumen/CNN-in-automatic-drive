# Optional CuPy GPU Environment

Use the existing root `.venv` with Python 3.12 and the baseline dependencies.
These additional dependencies target NVIDIA CUDA 12 on Windows with a compatible
driver. They are separate from the CPU/macOS environment. Do not install multiple
CuPy distributions together. See the [CuPy installation guide](https://docs.cupy.dev/en/v13.6.0/install.html).

Run from the repository root:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The requirements include CUDA runtime, NVRTC, cuBLAS, and cuRAND wheels, so a
system-wide Toolkit is not needed for the tested operations. `nvidia-smi` reports
driver capability, not the installed Toolkit version. A CUDA 12.9-compatible
driver is required for this pinned runtime.

## Windows DLL Discovery

CuPy 13 needs help finding the NVIDIA wheel DLLs. Run this snippet with the
virtual environment's Python once after installation. It writes an ignored,
environment-local startup file, enabling scripts and notebook kernels to find
the libraries. Restart existing notebook kernels afterward. No system settings
are changed. Remove `cnn_cuda.pth` from the environment to undo the configuration.

```python
from pathlib import Path
import sysconfig

startup = "; ".join([
    "import os, sys",
    "from pathlib import Path",
    "_cuda_root = Path(sys.prefix) / 'Lib/site-packages/nvidia'",
    "_cuda_bins = list(_cuda_root.glob('*/bin')) if os.name == 'nt' else []",
    "sys._cnn_cuda_dlls = [os.add_dll_directory(str(p)) for p in _cuda_bins]",
    "os.environ['PATH'] = os.pathsep.join(map(str, _cuda_bins)) + os.pathsep + os.environ.get('PATH', '') if _cuda_bins else os.environ.get('PATH', '')",
    "os.environ.setdefault('CUDA_PATH', str(_cuda_root / 'cuda_runtime')) if _cuda_bins else None",
])
(Path(sysconfig.get_path("purelib")) / "cnn_cuda.pth").write_text(startup + "\n")
```

Both DLL registration and the process-local `PATH` are needed: NVRTC loads its
builtins dynamically. The handles remain alive for the interpreter's lifetime.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest projl/proj/tests/integration/test_gpu.py -v
.\.venv\Scripts\python.exe -m pytest projl/proj/tests -q
```

Confirm all eight GPU cases pass with no skips for GPU acceptance. Tests verify
device array types, compare CPU/GPU outputs and gradients, and require a classifier
to reduce its loss and predict the training labels correctly. A missing CuPy
installation or unavailable device skips GPU cases in the ordinary suite;
compilation and library failures on an available device remain errors.

These tests do not benchmark performance or validate full MNIST/CNN training.
