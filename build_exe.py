"""Package the desktop app into a standalone Windows .exe with PyInstaller.

    pip install pyinstaller
    python build_exe.py

The result is dist/Pixelizer.exe. The AI model is looked up in a models/ folder next
to the .exe (the app's "Install AI model" button puts it there).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEP = ";" if sys.platform == "win32" else ":"
# Big optional packages that onnxruntime / Pillow might import if present; the app needs none of them.
EXCLUDE = ["torch", "torchvision", "torchaudio", "xformers", "transformers", "bitsandbytes", "tensorflow",
           "scipy", "pandas", "matplotlib", "numba", "llvmlite", "sympy", "cv2", "IPython", "jedi",
           "librosa", "datasets", "nltk", "moviepy", "imageio", "lxml", "openpyxl", "grpc", "pyarrow"]

subprocess.run([
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", "Pixelizer", "--icon", str(ROOT / "icon.ico"),
    "--add-data", f"{ROOT / 'icon.ico'}{SEP}.",
    *[arg for mod in EXCLUDE for arg in ("--exclude-module", mod)],
    str(ROOT / "pixelizer_gui.pyw"),
], cwd=ROOT, check=True)
print(f"\nBuilt: {ROOT / 'dist' / 'Pixelizer.exe'}")
