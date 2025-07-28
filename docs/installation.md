# OrthoRoute Installation Guide

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with Compute Capability 7.5 or higher
- Minimum 8GB GPU RAM (16GB recommended for complex boards)
- 16GB System RAM
- 64-bit operating system (Windows, Linux, or macOS)

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit 11.8+ or 12.x
- KiCad 7.0 or higher (for KiCad plugin)

## Step-by-Step Installation

### 1. Install CUDA Toolkit
First, install the NVIDIA CUDA Toolkit:

1. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your operating system and follow installation instructions
3. Verify installation:
```bash
nvidia-smi  # Should show GPU information
nvcc --version  # Should show CUDA compiler version
```

### 2. Install CuPy
Install CuPy package matching your CUDA version:

```bash
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
pip install cupy-cuda11x
```

Verify CuPy installation:
```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should show number of GPUs
```

### 3. Install OrthoRoute Package

#### From PyPI (not yet available):
```bash
pip install orthoroute
```

#### From Source:
```bash
# Clone repository
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Install package
pip install .

# For development installation:
pip install -e .[dev]
```

### 4. Install KiCad Plugin

After installing the OrthoRoute package, install the KiCad plugin:

#### Windows:
```powershell
# Create plugin directory
$PLUGIN_DIR="$env:APPDATA\kicad\7.0\3rdparty\plugins\OrthoRoute"
mkdir -p $PLUGIN_DIR

# Copy plugin files
cp -r kicad_plugin\* $PLUGIN_DIR
```

#### Linux:
```bash
# Create plugin directory
PLUGIN_DIR="~/.local/share/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p $PLUGIN_DIR

# Copy plugin files
cp -r kicad_plugin/* $PLUGIN_DIR
```

#### macOS:
```bash
# Create plugin directory
PLUGIN_DIR="~/Library/Application Support/kicad/7.0/3rdparty/plugins/OrthoRoute"
mkdir -p "$PLUGIN_DIR"

# Copy plugin files
cp -r kicad_plugin/* "$PLUGIN_DIR"
```

### 5. Verify Installation

1. Start KiCad PCB Editor
2. Look for "OrthoRoute GPU Autorouter" in the toolbar
3. Run a test route on a simple board

## Troubleshooting

### Common Issues

1. **Plugin Not Appearing in KiCad**
   - Verify plugin directory location
   - Check KiCad version (7.0+ required)
   - Restart KiCad

2. **GPU Not Found**
   - Check NVIDIA drivers are installed
   - Verify CUDA Toolkit installation
   - Run `nvidia-smi` to check GPU status

3. **Import Errors**
   - Verify OrthoRoute package installation: `pip show orthoroute`
   - Check Python path includes OrthoRoute
   - Verify CuPy installation matches CUDA version

4. **Memory Errors**
   - Reduce board grid resolution
   - Close other GPU applications
   - Try a smaller test board first

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
2. Run with debug logging:
```bash
export ORTHOROUTE_DEBUG=1
```
3. Submit bug reports with:
   - Full error message
   - System specifications
   - CUDA/CuPy versions
   - Simple test case reproducing the issue
