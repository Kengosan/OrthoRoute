<table width="100%">
  <tr>
    <td align="center" width="300">
      <img src="graphics/icon200.png" alt="OrthoRoute Logo" width="200" />
    </td>
    <td align="left">
      <h2>OrthoRoute - GPU Accelerated Autorouting for KiCad</h2>
      <p><em>You shouldn't trust the autorouter, but at least this one is faster</em></p>
    </td>
  </tr>
</table>

GPU-accelerated PCB autorouting plugin with advanced thermal relief visualization and professional-grade capabilities.

## Features

- **Thermal Relief Visualization**: Full visualization and processing of thermal relief patterns from KiCad's complex polygon data
- **Exact Pad Shapes**: Uses KiCad's native `get_pad_shapes_as_polygons()` API for precise pad geometry
- **KiCad IPC Integration**: Seamless communication with KiCad via modern IPC APIs
- **Real-time Visualization**: Interactive 2D board view with zoom, pan, and layer controls
- **Professional Interface**: Clean PyQt6-based interface with KiCad color themes
- **Multi-layer Support**: Handle complex multi-layer PCB designs with front/back copper visualization
- **GPU-Accelerated Routing**: Future support for ultra-fast routing algorithms
- **Manhattan Routing**: Where OrthoRoute gets its name
- **It's A Plugin**: Just install it using the KiCad Plugin Manager

## Screenshots

### Main Interface
<div align="center">
  <img src="graphics/screenshots/Screencap1-cseduino4.png" alt="OrthoRoute Main Interface" width="720">
  <br>
  <em>The OrthoRoute plugin interface showing an unrouted <a href="https://jpralves.net/pages/cseduino-v4.html">CSEduino v4</a> board with airwires and board information panel.</em>
</div>

## Project Structure

```
OrthoRoute/
├── src/                           # Core thermal relief plugin
│   ├── orthoroute_plugin.py      # Main plugin entry point
│   ├── thermal_relief_loader.py  # KiCad data extraction with thermal relief support
│   ├── orthoroute_window.py      # PyQt6 GUI with thermal relief visualization
│   ├── kicad_interface.py        # KiCad IPC API communication
│   └── unused/                   # Legacy routing algorithms (for future development)
├── debug/                         # Development and debugging files
├── docs/                         # Documentation
├── graphics/                     # Icons and screenshots
└── tests/                        # Test files
```

## Installation

### Option 1: Direct Download (Recommended)
1. Download the latest release from the [Releases](https://github.com/bbenchoff/OrthoRoute/releases) page
2. Extract the plugin ZIP file
3. Install via KiCad Plugin Manager

### Option 2: Build from Source
```bash
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute
python build.py --package production
```

## Requirements

### Required Dependencies
- **Python 3.8+**
- **KiCad 9.0+** with IPC API support
- **kipy** - KiCad IPC client library
- **PyQt6** - GUI framework

### Optional Dependencies
- **NumPy** - Array operations for future routing algorithms
- **CuPy** - GPU acceleration for future development

## Usage

1. Open your PCB project in KiCad
2. Run the thermal relief plugin from the command line:
   ```bash
   cd src
   python orthoroute_plugin.py
   ```
3. The plugin will automatically connect to KiCad and analyze your board
4. Use the visualization interface to:
   - View thermal relief patterns embedded in copper pours
   - Examine exact pad shapes from KiCad
   - Toggle layer visibility (F.Cu/B.Cu)
   - Zoom and pan around the PCB

## Building

The project includes build scripts for creating plugin packages:

```bash
# Build KiCad plugin package
python build_ipc_plugin.py

# General build system (work in progress)
python build.py
```

### Current Status

- **Thermal Relief Plugin**: ✅ Fully functional with KiCad IPC integration
- **GPU Routing**: 🚧 Future development (algorithms preserved in `src/unused/`)
- **Plugin Packaging**: ✅ Available via build scripts

## Current Capabilities

OrthoRoute currently excels at thermal relief visualization:

- **Thermal Relief Analysis**: Processes complex 5,505+ point polygon outlines from KiCad
- **Exact Geometry**: Renders precise pad shapes using KiCad's native polygon API
- **Real-time Visualization**: Interactive PCB viewer with zoom, pan, and layer controls
- **KiCad Integration**: Direct IPC connection for live board data

## Future Development

- **GPU Routing**: CUDA-accelerated pathfinding algorithms (preserved in `src/unused/`)
- **Advanced Algorithms**: Push-and-shove, differential pairs, design rule checking
- **Memory Efficient**: Optimized for large PCB designs

## Stability

- **Process Isolation**: Runs in separate process to prevent KiCad crashes
- **Error Recovery**: Robust error handling and recovery mechanisms
- **IPC Communication**: Safe inter-process communication with KiCad
- **Memory Management**: Efficient memory usage for large designs

## Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md) 
- [API Reference](docs/api_reference.md)
- [Contributing](docs/contributing.md)

## Contributing

If something's not working or you just don't like it, first please complain. Complaining about free stuff will actually force me to fix it. Please see [Contributing Guidelines](docs/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋Support

- **Issues**: [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)
- **Documentation**: [Project Wiki](https://github.com/bbenchoff/OrthoRoute/wiki)

## Roadmap

- [ ] Advanced routing algorithms (push-and-shove, differential pairs)
- [ ] PCB stackup awareness and layer-specific routing
- [ ] Design rule checking integration
- [ ] Batch routing for multiple PCBs
- [ ] Machine learning route optimization
- [ ] Integration with external routing services

---

**OrthoRoute** - Professional PCB autorouting for KiCad with modern GPU acceleration and official API integration.