@echo off
echo OrthoRoute KiCad Plugin Installer
echo ===============================
echo.

:: Determine KiCad plugin directory
set KICAD_PLUGIN_DIR=%APPDATA%\kicad\7.0\scripting\plugins\OrthoRoute

echo Installing to: %KICAD_PLUGIN_DIR%
echo.

:: Create plugin directory if it doesn't exist
if not exist "%KICAD_PLUGIN_DIR%" (
    echo Creating plugin directory...
    mkdir "%KICAD_PLUGIN_DIR%"
) else (
    echo Plugin directory already exists, will update files...
)

:: Copy plugin files
echo Copying plugin files...
xcopy /y /s /i kicad_plugin\*.py "%KICAD_PLUGIN_DIR%\"
xcopy /y kicad_plugin\icon.png "%KICAD_PLUGIN_DIR%\"

:: Install the orthoroute package in development mode
echo.
echo Installing orthoroute package...
pip install -e .

echo.
echo Installation complete!
echo.
echo Restart KiCad for the plugin to appear in the PCB Editor.
echo The plugin will be available in the Tools menu and toolbar.
echo.
pause
