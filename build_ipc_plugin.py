#!/usr/bin/env python3
"""
OrthoRoute Plugin Package Builder - Creates complete KiCad plugin
"""
import os
import shutil
import zipfile
import tempfile
from pathlib import Path


def create_plugin_package():
    """Create a complete OrthoRoute plugin package"""

    print("Building OrthoRoute IPC Plugin Package...")

    current_dir = Path(__file__).parent
    build_dir = current_dir / "build"
    build_dir.mkdir(exist_ok=True)

    package_name = "orthoroute-ipc-plugin.zip"
    package_path = build_dir / package_name

    if package_path.exists():
        package_path.unlink()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create plugins directory structure
        plugins_dir = temp_path / "plugins"
        plugins_dir.mkdir()

        # Copy the KiCad plugin wrapper (ActionPlugin) as fallback
        plugin_wrapper = current_dir / "orthoroute_plugin.py"
        if plugin_wrapper.exists():
            shutil.copy2(plugin_wrapper, plugins_dir / "__init__.py")
            print("✓ Copied KiCad plugin wrapper")
        else:
            print("❌ KiCad plugin wrapper not found")
            return False

        # Copy the IPC client
        ipc_client = current_dir / "orthoroute_working_plugin.py"
        if ipc_client.exists():
            shutil.copy2(ipc_client, plugins_dir / "orthoroute_working_plugin.py")
            print("✓ Copied IPC client")
        else:
            print("❌ IPC client not found")
            return False

        # Include Qt UI and GPU engine from src/
        src_dir = current_dir / "src"
        qt_ui = src_dir / "orthoroute_window.py"
        gpu_engine = src_dir / "gpu_routing_engine.py"
        if qt_ui.exists():
            shutil.copy2(qt_ui, plugins_dir / "orthoroute_window.py")
            print("✓ Included Qt UI: orthoroute_window.py")
        if gpu_engine.exists():
            shutil.copy2(gpu_engine, plugins_dir / "gpu_routing_engine.py")
            print("✓ Included GPU engine: gpu_routing_engine.py")

        # Copy icon if it exists
        icon_files = [
            current_dir / "assets" / "BigIcon.png",
            current_dir / "icon.png",
            current_dir / "icon24.png",
        ]

        for icon_file in icon_files:
            if icon_file.exists():
                shutil.copy2(icon_file, plugins_dir / "icon.png")
                print(f"✓ Copied icon: {icon_file.name}")
                break

        # Create plugin.json for IPC Python runtime (ensures KiCad passes IPC credentials)
        plugin_json = {
            "identifier": "com.orthoroute.ipc",
            "name": "OrthoRoute IPC",
            "version": "1.1.0",
            "description": "GPU-accelerated autorouter with Qt visualization and KiCad IPC integration",
            "runtime": "python",
            "actions": [
                {
                    "id": "orthoroute.run",
                    "label": "OrthoRoute",
                    "icon": "icon.png",
                    "entry_point": "orthoroute_working_plugin.py"
                }
            ]
        }
        import json
        with open(plugins_dir / "plugin.json", 'w', encoding='utf-8') as f:
            json.dump(plugin_json, f, indent=2)
        print("✓ Created plugin.json (IPC Python runtime)")

        # Create PCM metadata.json (package descriptor)
        metadata_content = '''{
    "versions": [
        {
            "version": "1.1.0",
            "status": "stable",
            "kicad_version": "9.0"
        }
    ],
    "name": "OrthoRoute IPC",
    "description": "GPU-accelerated autorouter with Qt visualization and KiCad IPC API integration",
    "description_full": "OrthoRoute plugin that connects to KiCad via the official IPC API and launches a Qt UI with optional CuPy GPU acceleration.",
    "identifier": "com.orthoroute.ipc",
    "type": "plugin",
    "author": {"name": "OrthoRoute Team", "contact": {"email": "support@orthoroute.com"}},
    "maintainer": {"name": "OrthoRoute Team", "contact": {"email": "support@orthoroute.com"}},
    "license": "MIT",
    "resources": {"homepage": "https://github.com/orthoroute/orthoroute"},
    "tags": ["autorouter", "routing", "gpu", "ipc", "qt"],
    "keep_on_update": []
}'''

        metadata_path = temp_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        print("✓ Created metadata.json")

        # Create the ZIP package
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            zf.write(metadata_path, "metadata.json")

            # Add all plugin files
            for file_path in plugins_dir.rglob('*'):
                if file_path.is_file():
                    arc_name = "plugins" / file_path.relative_to(plugins_dir)
                    zf.write(file_path, str(arc_name))
                    print(f"Added: {arc_name}")

    print(f"\n✅ OrthoRoute IPC Plugin created: {package_path}")
    print(f"Size: {package_path.stat().st_size:,} bytes")
    print("\nInstall via KiCad PCM → Install from File. KiCad will manage a venv and pass IPC credentials (socket/token).")
    print("If ActionPlugin also appears, use the 'OrthoRoute' button that comes from the IPC system.")

    return True


if __name__ == "__main__":
    create_plugin_package()
