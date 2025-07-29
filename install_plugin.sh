#!/bin/bash

echo "OrthoRoute KiCad Plugin Installer"
echo "==============================="
echo

# Determine OS type and set KiCad plugin directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    KICAD_PLUGIN_DIR="$HOME/Library/Preferences/kicad/7.0/scripting/plugins/OrthoRoute"
else
    # Linux
    KICAD_PLUGIN_DIR="$HOME/.local/share/kicad/7.0/scripting/plugins/OrthoRoute"
fi

echo "Installing to: $KICAD_PLUGIN_DIR"
echo

# Create plugin directory if it doesn't exist
if [ ! -d "$KICAD_PLUGIN_DIR" ]; then
    echo "Creating plugin directory..."
    mkdir -p "$KICAD_PLUGIN_DIR"
else
    echo "Plugin directory already exists, will update files..."
fi

# Copy plugin files
echo "Copying plugin files..."
cp -f kicad_plugin/*.py "$KICAD_PLUGIN_DIR/"
cp -f kicad_plugin/icon.png "$KICAD_PLUGIN_DIR/"

# Install the orthoroute package in development mode
echo
echo "Installing orthoroute package..."
pip install -e .

echo
echo "Installation complete!"
echo
echo "Restart KiCad for the plugin to appear in the PCB Editor."
echo "The plugin will be available in the Tools menu and toolbar."
echo

read -p "Press Enter to exit..."
