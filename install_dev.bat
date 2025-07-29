@echo off
echo Installing OrthoRoute package in development mode...
pip install -e .

echo.
echo Verifying installation...
python -c "from orthoroute.gpu_engine import OrthoRouteEngine; engine = OrthoRouteEngine(); print('route method exists:', hasattr(engine, 'route'))"

echo.
echo Done!
