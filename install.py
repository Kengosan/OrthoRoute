#!/usr/bin/env python3
"""
OrthoRoute Installation Script

This script automatically installs the OrthoRoute KiCad plugin to the user's
KiCad scripting plugins directory and ensures all dependencies are met.
"""

import os
import sys
import shutil
import platform
import subprocess
import importlib.util
import configparser
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}== {text} =={Colors.ENDC}")

def print_success(text):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.YELLOW}! {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def get_kicad_plugin_dir():
    """Determine the KiCad plugin directory based on the OS."""
    system = platform.system()
    
    if system == "Windows":
        # Check for KiCad 7.0 first (most common)
        candidates = [
            os.path.join(os.path.expanduser("~"), "Documents", "KiCad", "9.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Documents", "KiCad", "8.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Documents", "KiCad", "7.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Documents", "KiCad", "6.0", "scripting", "plugins"),
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            os.path.join(os.path.expanduser("~"), "Library", "Preferences", "kicad", "9.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Library", "Preferences", "kicad", "8.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Library", "Preferences", "kicad", "7.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), "Library", "Preferences", "kicad", "6.0", "scripting", "plugins"),
        ]
    else:  # Linux and others
        candidates = [
            os.path.join(os.path.expanduser("~"), ".local", "share", "kicad", "9.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "kicad", "8.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "kicad", "7.0", "scripting", "plugins"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "kicad", "6.0", "scripting", "plugins"),
        ]
    
    # Return the first directory that exists
    for path in candidates:
        if os.path.exists(os.path.dirname(path)):
            os.makedirs(path, exist_ok=True)
            return path
    
    # If none exist, use the first candidate and create it
    os.makedirs(candidates[0], exist_ok=True)
    return candidates[0]

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = ["numpy", "cupy", "wxpython"]
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies."""
    print_header("Installing Dependencies")
    
    for package in missing_packages:
        print(f"Installing {package}...")
        
        if package == "cupy":
            # CuPy requires special installation based on CUDA version
            try:
                # Check if CUDA is available and get version
                try:
                    result = subprocess.run(["nvcc", "--version"], 
                                          capture_output=True, 
                                          text=True, 
                                          check=False)
                    if "release" in result.stdout:
                        # Extract CUDA version
                        import re
                        match = re.search(r"release (\d+\.\d+)", result.stdout)
                        if match:
                            cuda_version = match.group(1)
                            cuda_major = int(cuda_version.split('.')[0])
                            
                            if cuda_major >= 12:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])
                            elif cuda_major == 11:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda11x"])
                            elif cuda_major == 10:
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda10x"])
                            else:
                                print_warning(f"CUDA version {cuda_version} detected but not supported. Installing default CuPy.")
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
                        else:
                            print_warning("Could not determine CUDA version. Installing default CuPy.")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
                    else:
                        print_warning("CUDA not detected. Installing default CuPy.")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
                except FileNotFoundError:
                    print_warning("CUDA not detected. Installing default CuPy.")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
            except Exception as e:
                print_error(f"Failed to install CuPy: {str(e)}")
                print_warning("Please install CuPy manually: https://docs.cupy.dev/en/stable/install.html")
                continue
        else:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except Exception as e:
                print_error(f"Failed to install {package}: {str(e)}")
                continue
        
        print_success(f"{package} installed successfully")

def install_plugin(plugin_dir):
    """Install the OrthoRoute plugin to the KiCad plugin directory."""
    print_header("Installing OrthoRoute Plugin")
    
    # Get the current script directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the OrthoRoute plugin directory
    orthoroute_plugin_dir = os.path.join(plugin_dir, "OrthoRoute")
    os.makedirs(orthoroute_plugin_dir, exist_ok=True)
    print_success(f"Created plugin directory: {orthoroute_plugin_dir}")
    
    # Clean up any existing __pycache__ directory
    pycache_dir = os.path.join(orthoroute_plugin_dir, "__pycache__")
    if os.path.exists(pycache_dir):
        try:
            shutil.rmtree(pycache_dir)
            print_success("Removed old __pycache__ directory")
        except Exception as e:
            print_warning(f"Could not remove __pycache__: {e}")
    
    # Define plugin file contents
    init_content = '''"""
OrthoRoute KiCad Plugin
========================

GPU-accelerated PCB autorouter integration for KiCad
"""

# Import the plugin class first
from .orthoroute_plugin import OrthoRoute

# Then register the plugin
OrthoRoute().register()
'''
    
    # Write updated init file
    init_path = os.path.join(orthoroute_plugin_dir, "__init__.py")
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(init_content)
    print_success(f"Created __init__.py with {os.path.getsize(init_path)} bytes")
    
    # Copy main plugin file
    plugin_src = os.path.join(current_dir, "kicad_plugin", "orthoroute_plugin.py")
    plugin_dest = os.path.join(orthoroute_plugin_dir, "orthoroute_plugin.py")
    
    if os.path.exists(plugin_src):
        with open(plugin_src, 'r', encoding='utf-8') as src_file:
            content = src_file.read()
        
        with open(plugin_dest, 'w', encoding='utf-8') as dest_file:
            dest_file.write(content)
        print_success(f"Copied orthoroute_plugin.py with {os.path.getsize(plugin_dest)} bytes")
    else:
        print_error(f"Source file not found: {plugin_src}")
    
    # Create resources directory and copy icon
    resources_dir = os.path.join(orthoroute_plugin_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    
    icon_src = os.path.join(current_dir, "Assets", "icon24.png")
    icon_dest = os.path.join(resources_dir, "icon.png")
    
    if os.path.exists(icon_src):
        shutil.copy2(icon_src, icon_dest)
        print_success(f"Copied icon.png with {os.path.getsize(icon_dest)} bytes")
    else:
        print_warning("Icon not found at {icon_src}, plugin will not have an icon")
    
    return orthoroute_plugin_dir

def verify_installation(plugin_dir):
    """Verify the plugin installation."""
    print_header("Verifying Installation")
    
    # Check if the plugin directory exists
    orthoroute_plugin_dir = plugin_dir
    if not os.path.exists(orthoroute_plugin_dir):
        print_error(f"Plugin directory does not exist: {orthoroute_plugin_dir}")
        return False
        
    # Create a sitecustomize.py file to add the parent directory to Python path
    # This helps KiCad find the orthoroute package
    site_path = os.path.join(orthoroute_plugin_dir, "sitecustomize.py")
    with open(site_path, 'w', encoding='utf-8') as f:
        f.write("""
import os
import sys
import site

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
""")
    print_success("Created sitecustomize.py to help with package imports")
    
    # Check for required files
    required_files = ["__init__.py", "orthoroute_plugin.py"]
    all_files_exist = True
    
    for file in required_files:
        file_path = os.path.join(orthoroute_plugin_dir, file)
        if not os.path.exists(file_path):
            print_error(f"Required file missing: {file}")
            all_files_exist = False
    
    if all_files_exist:
        print_success("All required files found")
    
    return all_files_exist

def main():
    """Main installation function."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}OrthoRoute KiCad Plugin Installer{Colors.ENDC}")
    print("=" * 40)
    
    # Check and install dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print_warning("Missing dependencies: " + ", ".join(missing_packages))
        install_dependencies(missing_packages)
    else:
        print_success("All dependencies are installed")
    
    # Get KiCad plugin directory
    try:
        plugin_dir = get_kicad_plugin_dir()
        print_success(f"KiCad plugin directory: {plugin_dir}")
    except Exception as e:
        print_error(f"Failed to determine KiCad plugin directory: {str(e)}")
        return False
    
    # Install plugin
    try:
        plugin_dir = install_plugin(plugin_dir)
    except Exception as e:
        print_error(f"Failed to install plugin: {str(e)}")
        return False
    
    # Verify installation
    if verify_installation(plugin_dir):
        print(f"\n{Colors.GREEN}{Colors.BOLD}OrthoRoute plugin installed successfully!{Colors.ENDC}")
        print("\nTo use the plugin:")
        print(f"1. Restart KiCad if it's currently running")
        print(f"2. Open a PCB file in the PCB Editor")
        print(f"3. Look for 'OrthoRoute GPU Autorouter' in the Tools menu or PCB toolbar")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}OrthoRoute plugin installation failed!{Colors.ENDC}")
        print("Please check the errors above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
