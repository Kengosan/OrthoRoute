@echo off
echo OrthoRoute KiCad Plugin Verification
echo ==================================
echo.

:: Check KiCad plugin directory
set KICAD_PLUGIN_DIR=%APPDATA%\kicad\7.0\scripting\plugins\OrthoRoute
echo Checking KiCad plugin directory...
if exist "%KICAD_PLUGIN_DIR%" (
    echo FOUND: Plugin directory exists at %KICAD_PLUGIN_DIR%
    echo.
    
    :: Check for required files
    echo Checking for required plugin files...
    set MISSING=0
    
    if not exist "%KICAD_PLUGIN_DIR%\__init__.py" set MISSING=1
    if not exist "%KICAD_PLUGIN_DIR%\orthoroute_kicad.py" set MISSING=1
    if not exist "%KICAD_PLUGIN_DIR%\icon.png" set MISSING=1
    
    if %MISSING%==0 (
        echo GOOD: All required plugin files are present.
    ) else (
        echo WARNING: Some required plugin files may be missing.
    )
) else (
    echo ERROR: Plugin directory not found at %KICAD_PLUGIN_DIR%
    echo Plugin may not be installed correctly.
)

echo.

:: Check orthoroute package
echo Checking orthoroute package installation...
python -c "import orthoroute; print('FOUND: orthoroute package version', orthoroute.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: orthoroute package not found or not properly installed.
)

echo.

:: Check GPU and CuPy
echo Checking CuPy and GPU availability...
python -c "import cupy as cp; print(f'FOUND: CuPy installed with {cp.cuda.runtime.getDeviceCount()} CUDA devices available')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CuPy not installed or CUDA not properly configured.
)

echo.
echo Verification complete!
echo.
echo If all checks passed, the plugin should be ready to use in KiCad.
echo Please restart KiCad if it's currently running.
echo.
pause
