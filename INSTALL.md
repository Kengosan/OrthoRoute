# OrthoRoute: Installation Guide

This guide will help you install the OrthoRoute KiCad plugin and its required dependencies.

## Prerequisites

1. **KiCad 7.0 or newer**
   - Download from [kicad.org](https://www.kicad.org/download/)

2. **NVIDIA GPU with CUDA support**
   - NVIDIA GPU with compute capability 5.0+
   - Check your GPU's compute capability [here](https://developer.nvidia.com/cuda-gpus)

3. **CUDA Toolkit 11.8+ or 12.x**
   - Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

4. **Python 3.8 or newer**
   - Make sure pip is installed

## Installation Steps

### Automatic Installation (Recommended)

#### Windows

1. Clone or download this repository
2. Run `install_plugin.bat` by double-clicking it or running from Command Prompt
3. The script will:
   - Install the OrthoRoute package
   - Copy plugin files to the KiCad plugins directory
   - Set up the necessary dependencies

#### Linux/macOS

1. Clone or download this repository
2. Open a terminal in the repository directory
3. Run the installer script:
   ```bash
   chmod +x install_plugin.sh
   ./install_plugin.sh
   ```

### Manual Installation

If the automatic installation doesn't work for you, follow these steps:

1. Install CuPy:
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

2. Install OrthoRoute in development mode:
   ```bash
   pip install -e .
   ```

3. Copy the plugin files to KiCad plugin directory:
   - **Windows**: `%APPDATA%\kicad\7.0\scripting\plugins\OrthoRoute`
   - **Linux**: `~/.local/share/kicad/7.0/scripting/plugins/OrthoRoute`
   - **macOS**: `~/Library/Preferences/kicad/7.0/scripting/plugins/OrthoRoute`

## Verifying Installation

1. Start KiCad and open a PCB file
2. You should see the OrthoRoute icon in the toolbar
3. Click on the icon to open the OrthoRoute configuration dialog

## Troubleshooting

### Plugin Not Appearing in KiCad

1. Check that the plugin files are in the correct directory
2. Restart KiCad
3. Enable Python plugin support in KiCad preferences
4. Check the KiCad message panel for any plugin loading errors

### CUDA/CuPy Issues

1. Make sure your NVIDIA drivers are up to date
2. Verify CUDA Toolkit installation with:
   ```bash
   nvcc --version
   ```
3. Test CuPy installation:
   ```python
   import cupy as cp
   print(cp.cuda.runtime.getDeviceCount())
   ```
   
If you encounter any issues not covered here, please open an issue on GitHub.
