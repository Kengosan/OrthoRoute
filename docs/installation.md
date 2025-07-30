# OrthoRoute Installation Documentation

## Overview
This document covers the installation and initial setup of the OrthoRoute GPU Autorouter for KiCad.

## Installation Methods

### 1. Plugin and Content Manager (Recommended)
- Easiest method for end users
- Download pre-built package
- Install through KiCad's built-in manager
- Automatic dependency handling

### 2. Development Installation
- For developers and contributors
- Direct access to source code
- Enables modification and testing
- Uses `install_dev.py` script

### 3. Build from Source
- Complete control over build process
- Latest unreleased features
- Custom configurations possible

## Recent Development Progress (July 2025)

### Major Issues Resolved
1. **Plugin Crashes**: Fixed import errors and KiCad API compatibility
2. **Missing Functionality**: Added actual track creation to PCB
3. **UI Compatibility**: Fixed wxPython dialog issues for KiCad 8.0+
4. **Net Detection**: Corrected critical net-pad matching logic

### Debugging Journey
The plugin underwent extensive debugging to resolve the core issue: "plugin runs but doesn't actually route."

**Timeline of Fixes**:
- **Phase 1**: Eliminated crashes and import errors
- **Phase 2**: Implemented missing track creation functionality
- **Phase 3**: Fixed UI compatibility issues
- **Phase 4**: Identified and corrected net-pad matching bug
- **Phase 5**: Package rebuilt with all fixes (current status)

### Current Status
- ✅ Plugin loads without errors
- ✅ UI appears and functions correctly
- ✅ Track creation logic implemented
- 🔄 **Next**: Debug remaining net detection issues

The plugin now executes the routing process without crashes, but may still show "0 nets processed" depending on board configuration. This indicates further refinement needed in the net detection logic.

## Post-Installation Testing

### Verification Steps
1. Install plugin via preferred method
2. Restart KiCad completely
3. Open a PCB with unrouted nets
4. Access OrthoRoute via toolbar or menu
5. Run routing process

### Expected Behavior
- Plugin dialog should open without errors
- Configuration options should be accessible
- Routing process should complete without crashes
- Results dialog should show processing statistics

### Troubleshooting
If you encounter "0 nets processed":
1. Ensure PCB has components with unrouted connections
2. Run "Update PCB from Schematic" in KiCad
3. Verify ratsnest lines are visible
4. Check KiCad version compatibility (8.0+ required)

## Technical Requirements

### Minimum Requirements
- KiCad 8.0 or later
- Python 3.8+ (bundled with KiCad)
- Windows 10/11, Ubuntu 20.04+, or macOS 12+

### Optional Acceleration
- NVIDIA GPU with CUDA support
- 4GB+ GPU memory for large boards
- CuPy library for GPU acceleration

### Development Requirements
- Git for source code access
- Python development environment
- pytest for running test suite
