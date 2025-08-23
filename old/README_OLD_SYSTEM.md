# OrthoRoute - Legacy System (Archived)

This folder contains the original monolithic OrthoRoute implementation that was replaced by the new hexagonal architecture.

## Original Files Preserved Here:

- `src/` - Original source code with monolithic architecture
- `orthoroute_main.py` - Original main entry point

## Original Entry Points:

- `python src/orthoroute_plugin.py` - Original plugin launcher
- `python orthoroute_main.py gui` - Original GUI mode

## Migration Notes:

This legacy system has been replaced by the new modular architecture located in the project root:

- `orthoroute/` - New hexagonal architecture
- `main.py` - New unified entry point

**New Usage:**
```bash
python main.py                    # KiCad plugin with GUI (default)
python main.py plugin --no-gui    # Plugin without GUI  
python main.py cli board.kicad_pcb # CLI mode
```

The new system provides:
- ✅ Same GUI functionality
- ✅ Better separation of concerns
- ✅ Modular architecture
- ✅ CQRS pattern
- ✅ Event-driven design
- ✅ Multiple KiCad adapters (IPC/SWIG/File)

## Why Archived:

The legacy system is preserved here for reference but should not be used for new development. All functionality has been migrated to the new architecture with improved maintainability and extensibility.