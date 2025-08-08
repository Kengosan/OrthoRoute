#!/usr/bin/env python3
"""
OrthoRoute Installation Script
Installs OrthoRoute plugin for development and testing
"""

import os
import sys
import shutil
import platform
from pathlib import Path

def get_kicad_plugin_dir():
    """Get KiCad plugin directory for current platform"""
    system = platform.system()
    
    if system == "Windows":
        # Windows: C:\Users\<username>\Documents\KiCad\<version>\plugins\
        base_dir = Path.home() / "Documents" / "KiCad"
    elif system == "Darwin":  # macOS
        # macOS: ~/Documents/KiCad/<version>/plugins/
        base_dir = Path.home() / "Documents" / "KiCad"
    else:  # Linux
        # Linux: ~/.local/share/KiCad/<version>/plugins/
        base_dir = Path.home() / ".local" / "share" / "KiCad"
    
    # Look for KiCad version directories
    if not base_dir.exists():
        return None
    
    # Check for version 9.0+ first (latest)
    for version in ["9.0", "8.0", "7.0"]:
        plugin_dir = base_dir / version / "plugins"
        if plugin_dir.exists() or version == "9.0":  # Create 9.0 if it doesn't exist
            return plugin_dir
    
    return None

def install_plugin(dev_mode=True):
    """Install the plugin"""
    print("🔧 Installing OrthoRoute plugin...")
    
    # Get plugin directory
    plugin_dir = get_kicad_plugin_dir()
    if not plugin_dir:
        print("❌ Could not find KiCad plugin directory")
        print("Please create the directory manually:")
        print("  Windows: C:\\Users\\<username>\\Documents\\KiCad\\9.0\\plugins\\")
        print("  macOS: ~/Documents/KiCad/9.0/plugins/")
        print("  Linux: ~/.local/share/KiCad/9.0/plugins/")
        return False
    
    # Create plugin directory if it doesn't exist
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Create orthoroute plugin subdirectory
    orthoroute_dir = plugin_dir / "orthoroute"
    if orthoroute_dir.exists():
        print(f"  🗑️  Removing existing installation: {orthoroute_dir}")
        shutil.rmtree(orthoroute_dir)
    
    orthoroute_dir.mkdir()
    
    # Copy source files
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ Source directory not found. Run from project root.")
        return False
    
    for file in src_dir.glob("*.py"):
        shutil.copy2(file, orthoroute_dir)
        print(f"  📄 Copied {file.name}")
    
    # Copy icon if available
    icon_file = src_dir / "icon24.png"
    if icon_file.exists():
        shutil.copy2(icon_file, orthoroute_dir / "icon.png")
        print(f"  🎨 Copied icon")
    
    print(f"✅ Plugin installed to: {orthoroute_dir}")
    
    if dev_mode:
        print("\n🔧 Development mode installation completed")
        print("Changes to source files will require reinstallation")
    
    return True

def uninstall_plugin():
    """Uninstall the plugin"""
    print("🗑️  Uninstalling OrthoRoute plugin...")
    
    plugin_dir = get_kicad_plugin_dir()
    if not plugin_dir:
        print("❌ Could not find KiCad plugin directory")
        return False
    
    orthoroute_dir = plugin_dir / "orthoroute"
    if orthoroute_dir.exists():
        shutil.rmtree(orthoroute_dir)
        print(f"✅ Plugin removed from: {orthoroute_dir}")
    else:
        print("⚠️  Plugin not found, nothing to uninstall")
    
    return True

def check_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "Python": sys.version_info >= (3, 8),
        "KiCad Plugin Dir": get_kicad_plugin_dir() is not None
    }
    
    # Check optional dependencies
    try:
        import numpy
        requirements["NumPy"] = True
    except ImportError:
        requirements["NumPy"] = False
    
    try:
        import cupy
        requirements["CuPy (GPU)"] = True
    except ImportError:
        requirements["CuPy (GPU)"] = False
    
    try:
        import wx
        requirements["wxPython"] = True
    except ImportError:
        requirements["wxPython"] = False
    
    # Print results
    all_good = True
    for req, status in requirements.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {req}")
        if req in ["Python", "KiCad Plugin Dir"] and not status:
            all_good = False
    
    if not all_good:
        print("\n❌ Critical requirements not met")
        return False
    
    print("\n✅ Requirements check passed")
    return True

def main():
    """Main installation function"""
    print("🚀 OrthoRoute Installation Script")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "install"
    
    if command == "check":
        success = check_requirements()
    elif command == "uninstall":
        success = uninstall_plugin()
    elif command == "install":
        if not check_requirements():
            return 1
        success = install_plugin(dev_mode=True)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python install.py [install|uninstall|check]")
        return 1
    
    if success:
        if command == "install":
            print("\n📋 Next steps:")
            print("1. Restart KiCad completely")
            print("2. Open PCB Editor")
            print("3. Look for 'OrthoRoute GPU Autorouter' in:")
            print("   - Tools → External Plugins")
            print("   - Toolbar (if icon appears)")
        print("\n✅ Operation completed successfully!")
        return 0
    else:
        print("\n❌ Operation failed!")
        return 1

if __name__ == "__main__":
    exit(main())
