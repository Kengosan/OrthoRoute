# OrthoRoute Installation Guide

## Quick Installation (Recommended)

### Method 1: Plugin and Content Manager
1. Download `orthoroute-kicad-addon.zip` from releases
2. Open KiCad PCB Editor
3. Go to **Tools → Plugin and Content Manager**
4. Click **Install from File**
5. Select the zip file
6. **Restart KiCad completely**

⚠️ **Important**: Always restart KiCad after installation to ensure proper plugin loading.

## Development Installation

### Method 2: Development Setup
```bash
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute
python install_dev.py
```

### Method 3: Build from Source
```bash
python build_addon.py
# Then install the generated zip via Plugin Manager
```

## Troubleshooting Installation

### Plugin Not Appearing
- Verify installation in Plugin and Content Manager
- Check KiCad user directory for plugin files
- Restart KiCad completely
- Check Python console for error messages

### Recent Fixes Applied (July 2025)
- ✅ Fixed plugin crashes and import errors
- ✅ Added missing track creation functionality  
- ✅ Fixed wxPython UI compatibility for KiCad 8.0+
- ✅ Corrected net-pad matching logic (critical bug fix)

### Current Status
The plugin now loads and runs without errors, but net detection may still need refinement. If you see "Nets processed: 0", this indicates the net detection logic needs further debugging for your specific board configuration.

## System Requirements
- KiCad 8.0 or later
- Python 3.8+ (bundled with KiCad)
- Optional: NVIDIA GPU with CUDA support for acceleration

## Verification
After installation, the OrthoRoute icon should appear in the KiCad PCB Editor toolbar. The plugin can be accessed via:
- Toolbar icon (🔀)
- Tools → External Plugins → OrthoRoute GPU Autorouter
