<table width="100%">
  <tr>
    <td align="center" width="300">
      <img src="graphics/icon200.png" alt="OrthoRoute Logo" width="200" />
    </td>
    <td align="left">
      <h2>Professional PCB Autorouting Using Official KiCad 9.0+ IPC APIs</h2>
      <p><em>"Modern GPU-accelerated routing with official API integration."</em></p>
    </td>
  </tr>
</table>

GPU-accelerated PCB autorouting plugin with real-time visualization and professional-grade capabilities.

## 🚀 Features

- **GPU-Accelerated Routing**: Harness the power of modern graphics cards for ultra-fast routing
- **Real-time Visualization**: Live routing progress with interactive 2D board view
- **KiCad IPC Integration**: Seamless communication with KiCad via modern IPC APIs
- **Professional Interface**: Clean Qt6-based interface with routing controls and progress monitoring
- **Multi-layer Support**: Handle complex multi-layer PCB designs
- **Smart Algorithms**: Advanced pathfinding with obstacle avoidance and via optimization
- **Manhattan Routing**: Where OrthoRoute gets its name. 

## 📦 Installation

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

## 🛠️ Requirements

### Required Dependencies
- **Python 3.8+**
- **KiCad 9.0+** with IPC API support
- **kipy** - KiCad IPC client library
- **PyQt6** - GUI framework
- **NumPy** - Array operations

### Optional Dependencies
- **CuPy** - GPU acceleration (highly recommended)
- **NVIDIA GPU** with CUDA support for best performance

## 🎮 Usage

1. Open your PCB project in KiCad
2. Run the plugin from the command line:
   ```bash
   cd src
   python orthoroute.py
   ```
3. The plugin will automatically connect to KiCad and analyze your board
4. Use the routing controls to:
   - Select nets to route
   - Adjust routing parameters
   - Monitor progress in real-time
   - Apply routes back to KiCad

## 🏗️ Project Structure

```
OrthoRoute/
├── src/                       # Core application code
│   ├── orthoroute.py         # Main plugin entry point ⭐
│   ├── orthoroute_window.py   # Qt6 main interface
│   ├── orthoroute_main.py     # Core routing logic
│   ├── gpu_routing_engine.py  # GPU acceleration
│   └── kicad_interface.py     # KiCad IPC integration
├── graphics/                  # Icons and resources
├── docs/                      # Documentation
├── tests/                     # Test suite
├── build.py                   # Unified build system
└── requirements.txt           # Python dependencies
```

## 🔧 Building

The project includes a unified build system that creates multiple package formats:

```bash
# Build all packages
python build.py

# Build specific package
python build.py --package production
python build.py --package lite
python build.py --package development

# Clean build directory
python build.py --clean
```

### Package Types

- **Production** (`orthoroute-production`): Full-featured package with GPU acceleration and documentation
- **Lite** (`orthoroute-lite`): Minimal package for basic routing functionality
- **Development** (`orthoroute-dev`): Development build with debugging tools and tests

## 🚀 Performance

OrthoRoute leverages GPU acceleration for exceptional performance:

- **CPU Routing**: Traditional algorithms for compatibility
- **GPU Routing**: CUDA-accelerated pathfinding with 10-100x speedup
- **Real-time Updates**: Live visualization of routing progress
- **Memory Efficient**: Optimized for large PCB designs

## 🛡️ Stability

- **Process Isolation**: Runs in separate process to prevent KiCad crashes
- **Error Recovery**: Robust error handling and recovery mechanisms
- **IPC Communication**: Safe inter-process communication with KiCad
- **Memory Management**: Efficient memory usage for large designs

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md) 
- [API Reference](docs/api_reference.md)
- [Contributing](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see [Contributing Guidelines](docs/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)
- **Documentation**: [Project Wiki](https://github.com/bbenchoff/OrthoRoute/wiki)

## 🎯 Roadmap

- [ ] Advanced routing algorithms (push-and-shove, differential pairs)
- [ ] PCB stackup awareness and layer-specific routing
- [ ] Design rule checking integration
- [ ] Batch routing for multiple PCBs
- [ ] Machine learning route optimization
- [ ] Integration with external routing services

---

**OrthoRoute** - Professional PCB autorouting for KiCad with modern GPU acceleration and official API integration.