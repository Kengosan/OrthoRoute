#!/usr/bin/env python3
"""
OrthoRoute Build Script - Clean and Simple
Builds KiCad 9.0+ IPC plugin package
"""

import os
import json
import zipfile
import hashlib
from pathlib import Path

def main():
    """Build the OrthoRoute plugin package"""
    
    # Paths
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    build_dir = root_dir / "build"
    assets_dir = root_dir / "assets"
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # Package info
    package_name = "orthoroute-gpu-1.0.0.zip"
    package_path = build_dir / package_name
    
    # Remove existing package
    if package_path.exists():
        package_path.unlink()
        print(f"Removed existing {package_name}")
    
    print("🚀 Building OrthoRoute GPU Plugin Package")
    print("=" * 50)
    
    # Create ZIP package
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add metadata.json
        zipf.write(src_dir / "metadata.json", "metadata.json")
        print("  ✅ metadata.json")
        
        # Add plugin files
        zipf.write(src_dir / "plugin.json", "plugins/plugin.json")
        print("  ✅ plugins/plugin.json")
        
        zipf.write(src_dir / "gpu_autorouter.py", "plugins/gpu_autorouter.py") 
        print("  ✅ plugins/gpu_autorouter.py")
        
        zipf.write(src_dir / "gpu_router.py", "plugins/gpu_router.py")
        print("  ✅ plugins/gpu_router.py")
        
        zipf.write(src_dir / "requirements.txt", "plugins/requirements.txt")
        print("  ✅ plugins/requirements.txt")
        
        zipf.write(src_dir / "icon24.png", "plugins/icon24.png")
        print("  ✅ plugins/icon24.png")
        
        # Add resources
        if (assets_dir / "icon.png").exists():
            zipf.write(assets_dir / "icon.png", "resources/icon.png")
            print("  ✅ resources/icon.png")
    
    # Get file size and hash
    file_size = package_path.stat().st_size
    with open(package_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    print("\n🎉 Package built successfully!")
    print(f"📦 File: {package_path}")
    print(f"📏 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"🔒 SHA256: {file_hash}")
    
    print("\n📋 Installation Instructions:")
    print("1. Open KiCad PCB Editor")
    print("2. Go to Tools → Plugin and Content Manager")
    print("3. Click 'Install from File'")
    print(f"4. Select: {package_name}")
    print("5. Enable API server: Preferences → Plugins → 'Enable external plugin API server'")
    print("6. Restart KiCad")
    print("7. Look for 'Run GPU Autorouter' button in PCB Editor toolbar!")

if __name__ == "__main__":
    main()
