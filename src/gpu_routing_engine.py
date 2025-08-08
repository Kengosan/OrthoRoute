#!/usr/bin/env python3
"""
GPU Routing Engine - Core routing algorithms using CUDA/CuPy
"""

import logging
import time
import json
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RoutingResult:
    """Result of routing operation"""
    success: bool
    tracks_created: int = 0
    vias_created: int = 0
    success_rate: float = 0.0
    routing_time: float = 0.0
    error: Optional[str] = None
    failed_nets: List[str] = None

class GPURoutingEngine:
    """GPU-accelerated routing engine with process isolation"""
    
    def __init__(self, board_data: Dict, kicad_interface):
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.routing_process = None
        self.temp_dir = None
        self.settings = {
            'grid_pitch': 0.1,  # mm
            'via_cost': 50,
            'max_iterations': 10,
            'use_gpu': True,
            'batch_size': 5
        }
        
    def update_settings(self, settings: Dict):
        """Update routing settings"""
        self.settings.update(settings)
        logger.info(f"Updated routing settings: {settings}")
    
    def route_all_nets(self, progress_callback=None) -> RoutingResult:
        """Route all unrouted nets using GPU acceleration"""
        logger.info("🚀 Starting GPU routing of all nets...")
        start_time = time.time()
        
        try:
            # Setup temporary directory for communication
            self.temp_dir = tempfile.mkdtemp(prefix="orthoroute_")
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Prepare routing request
            request_data = self._prepare_routing_request()
            request_file = os.path.join(self.temp_dir, "routing_request.json")
            
            with open(request_file, 'w') as f:
                json.dump(request_data, f, indent=2)
            
            # Launch GPU routing server
            result = self._launch_gpu_server(progress_callback)
            
            # Apply results to KiCad
            if result.success:
                self._apply_routing_results()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ GPU routing failed: {e}")
            return RoutingResult(
                success=False,
                error=str(e),
                routing_time=time.time() - start_time
            )
        finally:
            self._cleanup()
    
    def _prepare_routing_request(self) -> Dict:
        """Prepare routing request data"""
        # Extract unrouted nets
        unrouted_nets = [net for net in self.board_data.get('nets', []) 
                        if not net.get('routed', False)]
        
        # Get board bounds
        bounds = self.board_data.get('bounds', (0, 0, 100, 80))
        
        request = {
            'board_info': {
                'width_mm': bounds[2] - bounds[0],
                'height_mm': bounds[3] - bounds[1],
                'layers': self.board_data.get('layers', 2),
                'bounds': bounds
            },
            'nets': unrouted_nets,
            'components': self.board_data.get('components', []),
            'existing_tracks': self.board_data.get('tracks', []),
            'settings': self.settings,
            'temp_dir': self.temp_dir
        }
        
        logger.info(f"Prepared routing request: {len(unrouted_nets)} nets to route")
        return request
    
    def _launch_gpu_server(self, progress_callback=None) -> RoutingResult:
        """Launch GPU routing server in separate process"""
        try:
            # Create GPU routing server script
            server_script = self._create_gpu_server_script()
            
            # Launch server process
            logger.info("Launching GPU routing server...")
            self.routing_process = subprocess.Popen(
                [sys.executable, server_script],
                cwd=self.temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress
            return self._monitor_routing_progress(progress_callback)
            
        except Exception as e:
            logger.error(f"Failed to launch GPU server: {e}")
            return RoutingResult(success=False, error=str(e))
    
    def _create_gpu_server_script(self) -> str:
        """Create GPU routing server script"""
        server_script = os.path.join(self.temp_dir, "gpu_routing_server.py")
        
        script_content = '''#!/usr/bin/env python3
"""
GPU Routing Server - Isolated GPU routing process
"""

import sys
import json
import time
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main GPU routing server"""
    try:
        # Load routing request
        with open('routing_request.json', 'r') as f:
            request = json.load(f)
        
        logger.info("🔥 GPU Routing Server started")
        logger.info(f"Board: {request['board_info']['width_mm']:.1f}x{request['board_info']['height_mm']:.1f}mm")
        logger.info(f"Nets to route: {len(request['nets'])}")
        
        # Check GPU availability
        gpu_available = check_gpu()
        
        # Run routing
        result = run_routing(request, gpu_available)
        
        # Save results
        with open('routing_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✅ Routing completed: {result['success_rate']:.1f}% success")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        traceback.print_exc()
        
        # Save error result
        error_result = {
            'success': False,
            'error': str(e),
            'tracks_created': 0,
            'success_rate': 0.0
        }
        
        try:
            with open('routing_result.json', 'w') as f:
                json.dump(error_result, f)
        except:
            pass
        
        return 1

def check_gpu():
    """Check GPU availability"""
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode("utf-8")
        logger.info(f"✅ GPU available: {gpu_name}")
        return True
    except Exception as e:
        logger.info(f"❌ GPU not available: {e}")
        return False

def run_routing(request, gpu_available):
    """Run the actual routing"""
    nets = request['nets']
    settings = request['settings']
    
    # Initialize progress tracking
    with open('routing_status.json', 'w') as f:
        json.dump({'progress': 0, 'current_net': 'Starting...', 'status': 'running'}, f)
    
    routed_tracks = []
    routed_vias = []
    successful_nets = 0
    
    # Route each net
    for i, net in enumerate(nets):
        # Update progress
        progress = int((i / len(nets)) * 100)
        status = {
            'progress': progress,
            'current_net': net.get('name', f'Net_{i}'),
            'status': 'routing'
        }
        
        with open('routing_status.json', 'w') as f:
            json.dump(status, f)
        
        logger.info(f"🔄 Routing net {i+1}/{len(nets)}: {net.get('name', 'Unknown')}")
        
        # Simulate routing (replace with actual GPU routing)
        if gpu_available:
            success = route_net_gpu(net, settings)
        else:
            success = route_net_cpu(net, settings)
        
        if success:
            successful_nets += 1
            # Create mock track data
            pins = net.get('pins', [])
            if len(pins) >= 2:
                track = {
                    'net_name': net.get('name', f'Net_{i}'),
                    'start': {'x': pins[0]['x'], 'y': pins[0]['y']},
                    'end': {'x': pins[1]['x'], 'y': pins[1]['y']},
                    'layer': 'F.Cu',
                    'width': 0.2  # mm
                }
                routed_tracks.append(track)
        
        # Small delay to simulate routing time
        time.sleep(0.1)
    
    # Final status
    with open('routing_status.json', 'w') as f:
        json.dump({'progress': 100, 'current_net': 'Complete', 'status': 'finished'}, f)
    
    success_rate = (successful_nets / len(nets)) * 100 if nets else 0
    
    return {
        'success': True,
        'tracks_created': len(routed_tracks),
        'vias_created': len(routed_vias),
        'success_rate': success_rate,
        'routed_nets': successful_nets,
        'total_nets': len(nets),
        'tracks': routed_tracks,
        'vias': routed_vias
    }

def route_net_gpu(net, settings):
    """Route a single net using GPU"""
    try:
        import cupy as cp
        # Mock GPU routing algorithm
        grid_pitch = settings.get('grid_pitch', 0.1)
        max_iter = settings.get('max_iterations', 10)
        
        # Simulate GPU pathfinding
        time.sleep(0.05)  # Simulate GPU computation
        
        # 85% success rate for demo
        import random
        return random.random() < 0.85
        
    except Exception as e:
        logger.error(f"GPU routing failed: {e}")
        return False

def route_net_cpu(net, settings):
    """Route a single net using CPU fallback"""
    try:
        # Mock CPU routing algorithm
        time.sleep(0.1)  # Simulate CPU computation
        
        # 70% success rate for CPU mode
        import random
        return random.random() < 0.70
        
    except Exception as e:
        logger.error(f"CPU routing failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(main())
'''
        
        with open(server_script, 'w') as f:
            f.write(script_content)
        
        return server_script
    
    def _monitor_routing_progress(self, progress_callback=None) -> RoutingResult:
        """Monitor routing progress and return result"""
        status_file = os.path.join(self.temp_dir, "routing_status.json")
        result_file = os.path.join(self.temp_dir, "routing_result.json")
        
        try:
            # Monitor progress
            while self.routing_process and self.routing_process.poll() is None:
                if os.path.exists(status_file):
                    try:
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                        
                        if progress_callback:
                            progress_callback(
                                status.get('progress', 0),
                                status.get('current_net', 'Unknown'),
                                status.get('status', 'running')
                            )
                    except:
                        pass
                
                time.sleep(0.5)  # Check every 500ms
            
            # Get final result
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                return RoutingResult(
                    success=result_data.get('success', False),
                    tracks_created=result_data.get('tracks_created', 0),
                    vias_created=result_data.get('vias_created', 0),
                    success_rate=result_data.get('success_rate', 0.0),
                    routing_time=time.time() - time.time(),  # Will be updated
                    error=result_data.get('error')
                )
            else:
                return RoutingResult(success=False, error="No result file generated")
                
        except Exception as e:
            logger.error(f"Error monitoring progress: {e}")
            return RoutingResult(success=False, error=str(e))
    
    def _apply_routing_results(self):
        """Apply routing results to KiCad"""
        result_file = os.path.join(self.temp_dir, "routing_result.json")
        
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            tracks = result.get('tracks', [])
            vias = result.get('vias', [])
            
            logger.info(f"Applying {len(tracks)} tracks and {len(vias)} vias to KiCad")
            
            # Apply tracks
            for track in tracks:
                self.kicad_interface.create_track(
                    track['start']['x'],
                    track['start']['y'],
                    track['end']['x'],
                    track['end']['y'],
                    track.get('layer', 'F.Cu'),
                    track.get('width', 0.2),
                    track.get('net_name', '')
                )
            
            # Apply vias
            for via in vias:
                self.kicad_interface.create_via(
                    via['x'], via['y'],
                    via.get('size', 0.4),
                    via.get('drill', 0.2),
                    via.get('from_layer', 'F.Cu'),
                    via.get('to_layer', 'B.Cu'),
                    via.get('net_name', '')
                )
            
            # Refresh KiCad
            self.kicad_interface.refresh_board()
            
        except Exception as e:
            logger.error(f"Error applying results: {e}")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
    
    def stop_routing(self):
        """Stop the routing process"""
        if self.routing_process:
            try:
                self.routing_process.terminate()
                self.routing_process.wait(timeout=5)
                logger.info("Routing process stopped")
            except:
                self.routing_process.kill()
                logger.info("Routing process killed")
            finally:
                self.routing_process = None
