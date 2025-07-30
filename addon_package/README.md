# OrthoRoute KiCad Addon Package

This directory contains the KiCad addon package structure for OrthoRoute GPU Autorouter.

## Recent Development Status (July 2025)

**Major Debugging Session Completed**: Resolved core issue where plugin "doesn't actually route"

### Issues Fixed
- ✅ **Plugin Crashes**: Fixed import errors and KiCad API compatibility
- ✅ **Missing Track Creation**: Added `_create_tracks_from_path()` functionality
- ✅ **UI Compatibility**: Fixed wxPython dialogs for KiCad 8.0+
- ✅ **Net Detection Bug**: Corrected net-pad matching logic (critical fix)

### Current Status
Plugin now loads and runs without errors, but may show "0 nets processed" on some boards. This indicates additional refinement needed in net detection logic.

## Structure

```
addon_package/
├── metadata.json          # Package metadata for KiCad PCM
├── plugins/              # Plugin files
│   ├── __init__.py       # Main plugin with UI and routing logic
│   ├── orthoroute_engine.py  # Standalone GPU routing engine
│   └── icon.png          # 24x24 toolbar icon
└── resources/            # Package resources
    └── icon.png          # 64x64 package icon
```

## Building the Package

To create the installable zip package:

```bash
python build_addon.py
```

This creates `orthoroute-kicad-addon.zip` which can be installed via KiCad's Plugin and Content Manager.

## Installation

1. Open KiCad
2. Go to **Tools → Plugin and Content Manager**
3. **Uninstall any existing version first**
4. Click **Install from File**
5. Select the `orthoroute-kicad-addon.zip` file
6. **Restart KiCad completely**

⚠️ **Important**: Always restart KiCad after installation to ensure proper plugin loading.

## Requirements

- KiCad 8.0+
- NVIDIA GPU with CUDA support (optional)
- CUDA Toolkit 11.8+ or 12.x (optional)
- CuPy package: `pip install cupy-cuda12x` (optional)

## Features

- **GPU-accelerated routing**: Uses NVIDIA CUDA through CuPy for parallel algorithms
- **Real-time visualization**: Optional progress display during routing
- **Multi-layer support**: Works with complex PCB designs
- **Configurable parameters**: Grid pitch, iteration limits, via costs
- **Fallback mode**: Works without GPU for testing (CPU-only routing)
- **Enhanced debugging**: Comprehensive logging and error reporting

## Troubleshooting

### "0 nets processed" Issue
If the plugin runs but processes no nets:
1. Ensure your PCB has unrouted connections (ratsnest lines visible)
2. Run "Update PCB from Schematic" in KiCad
3. Verify components have proper net assignments
4. Check KiCad version compatibility (8.0+ required)

### Plugin Not Appearing
- Verify installation in Plugin and Content Manager
- Restart KiCad completely
- Check KiCad Python console for error messages

## Plugin Structure

The addon is designed as a self-contained package that doesn't require external Python packages to be installed in KiCad's environment. The routing engine is included directly in the plugin.

### Key Components

- **Main Plugin** (`__init__.py`): KiCad ActionPlugin interface with configuration dialog
- **Routing Engine** (`orthoroute_engine.py`): Standalone GPU routing implementation
- **Metadata** (`metadata.json`): Package information for KiCad PCM

This structure follows KiCad's official addon packaging guidelines and ensures reliable installation and loading.
