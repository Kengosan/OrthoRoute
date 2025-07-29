# Quick fix script for OrthoRoute plugin
import os
import shutil

# Source files
project_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_dir, 'kicad_plugin')
plugin_path = os.path.join(source_dir, 'orthoroute_plugin.py')
init_path = os.path.join(source_dir, '__init__.py')

# Destination directory
dest_dir = os.path.join(os.path.expanduser("~"), "Documents", "KiCad", "9.0", "scripting", "plugins", "OrthoRoute")
os.makedirs(dest_dir, exist_ok=True)

# Read source files
with open(plugin_path, 'r', encoding='utf-8') as f:
    plugin_content = f.read()

with open(init_path, 'r', encoding='utf-8') as f:
    init_content = f.read()

# Write to destination files
with open(os.path.join(dest_dir, 'orthoroute_plugin.py'), 'w', encoding='utf-8') as f:
    f.write(plugin_content)
    print(f"Written {len(plugin_content)} bytes to orthoroute_plugin.py")

with open(os.path.join(dest_dir, '__init__.py'), 'w', encoding='utf-8') as f:
    f.write(init_content)
    print(f"Written {len(init_content)} bytes to __init__.py")

print("Files copied successfully!")
