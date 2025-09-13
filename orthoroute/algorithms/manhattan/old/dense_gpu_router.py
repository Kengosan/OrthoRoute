"""
Dense GPU Manhattan Router - Replaces sparse RRG system
Uses GPU arrays directly, no Python object overhead
"""

import logging
import time
from typing import Dict, List, Tuple, Optional
from .dense_gpu_grid import DenseGPURouter, DenseGridConfig

logger = logging.getLogger(__name__)

class DenseGPUManhattanRouter:
    """GPU-native Manhattan router using dense grids"""
    
    def __init__(self, config: Optional[DenseGridConfig] = None):
        self.config = config or DenseGridConfig()
        self.gpu_router = None
        self.is_initialized = False
        
        logger.info("Dense GPU Manhattan router created")
    
    def initialize(self, board_data) -> bool:
        """Initialize router with board data"""
        try:
            logger.info("Initializing Dense GPU Manhattan router...")
            start_time = time.time()
            
            # Extract board information
            board_bounds = self._extract_board_bounds(board_data)
            pads = self._extract_pads(board_data) 
            nets = self._extract_nets(board_data)
            
            logger.info(f"Board: {len(pads)} pads, {len(nets)} nets")
            
            # Create GPU router
            self.gpu_router = DenseGPURouter(self.config)
            
            # Build dense grid on GPU
            success = self.gpu_router.build_grid(board_bounds, pads, nets)
            
            if success:
                self.is_initialized = True
                init_time = time.time() - start_time
                
                memory_info = self.gpu_router.get_memory_usage()
                logger.info(f"Dense GPU router initialized in {init_time:.2f}s")
                logger.info(f"GPU memory: {memory_info.get('grid_gb', 0):.1f}GB used")
                return True
            else:
                logger.error("Failed to build dense GPU grid")
                return False
                
        except Exception as e:
            logger.error(f"Router initialization failed: {e}")
            return False
    
    def route_nets(self, nets: List[Dict], progress_callback=None) -> Dict:
        """Route all nets using GPU"""
        if not self.is_initialized:
            logger.error("Router not initialized")
            return {'success': False, 'routes': []}
            
        logger.info(f"Starting dense GPU routing for {len(nets)} nets...")
        start_time = time.time()
        
        routes = []
        successful_routes = 0
        
        for i, net in enumerate(nets):
            try:
                # Get source and sink pads
                if 'pads' not in net or len(net['pads']) < 2:
                    logger.debug(f"Skipping net {net.get('name', 'unknown')} - insufficient pads")
                    continue
                
                pads = net['pads']
                net_name = net.get('name', f'net_{i}')
                
                # For now, route point-to-point (source to first sink)
                source_pad = pads[0]
                sink_pad = pads[1]
                
                # Route using GPU
                path = self.gpu_router.route_net(
                    source_pad=source_pad,
                    sink_pad=sink_pad,
                    net_id=net_name
                )
                
                if path:
                    routes.append({
                        'net': net_name,
                        'path': path,
                        'success': True,
                        'length': len(path)
                    })
                    successful_routes += 1
                else:
                    routes.append({
                        'net': net_name,
                        'path': [],
                        'success': False,
                        'error': 'No path found'
                    })
                
                # Progress reporting
                if progress_callback and (i + 1) % 10 == 0:
                    progress = (i + 1) / len(nets) * 100
                    progress_callback(f"Routed {i + 1}/{len(nets)} nets ({successful_routes} successful) - {progress:.1f}%")
                    
            except Exception as e:
                logger.error(f"Error routing net {net.get('name', 'unknown')}: {e}")
                routes.append({
                    'net': net.get('name', 'unknown'),
                    'path': [],
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        success_rate = (successful_routes / len(nets)) * 100 if nets else 0
        
        logger.info(f"Dense GPU routing completed: {successful_routes}/{len(nets)} nets ({success_rate:.1f}%) in {total_time:.2f}s")
        
        return {
            'success': True,
            'routes': routes,
            'stats': {
                'total_nets': len(nets),
                'successful_routes': successful_routes,
                'failed_routes': len(nets) - successful_routes,
                'success_rate': success_rate,
                'routing_time': total_time,
                'memory_usage': self.gpu_router.get_memory_usage()
            }
        }
    
    def _extract_board_bounds(self, board_data) -> Tuple[float, float, float, float]:
        """Extract board bounds from board data"""
        try:
            if hasattr(board_data, 'bounds'):
                bounds = board_data.bounds
                return (bounds['min_x'], bounds['min_y'], bounds['max_x'], bounds['max_y'])
            elif hasattr(board_data, 'pads'):
                # Calculate bounds from pads
                pads = board_data.pads
                if pads:
                    xs = [pad.x for pad in pads]
                    ys = [pad.y for pad in pads]
                    margin = 5.0  # 5mm margin
                    return (
                        min(xs) - margin,
                        min(ys) - margin, 
                        max(xs) + margin,
                        max(ys) + margin
                    )
            
            # Default bounds
            logger.warning("Could not extract board bounds, using defaults")
            return (0.0, 0.0, 300.0, 200.0)
            
        except Exception as e:
            logger.error(f"Error extracting board bounds: {e}")
            return (0.0, 0.0, 300.0, 200.0)
    
    def _extract_pads(self, board_data) -> List[Dict]:
        """Extract pad data for GPU routing"""
        pads = []
        
        try:
            if hasattr(board_data, 'pads'):
                for pad in board_data.pads:
                    pads.append({
                        'x': float(pad.x),
                        'y': float(pad.y),
                        'net': getattr(pad, 'net', ''),
                        'name': getattr(pad, 'name', ''),
                        'layer': getattr(pad, 'layer', 0)
                    })
            elif hasattr(board_data, 'nets'):
                # Extract pads from nets
                for net in board_data.nets:
                    if hasattr(net, 'pads'):
                        for pad in net.pads:
                            pads.append({
                                'x': float(pad.x),
                                'y': float(pad.y), 
                                'net': net.name,
                                'name': getattr(pad, 'name', ''),
                                'layer': getattr(pad, 'layer', 0)
                            })
                            
            logger.info(f"Extracted {len(pads)} pads for GPU routing")
            return pads
            
        except Exception as e:
            logger.error(f"Error extracting pads: {e}")
            return []
    
    def _extract_nets(self, board_data) -> List[Dict]:
        """Extract net data for routing"""
        nets = []
        
        try:
            if hasattr(board_data, 'nets'):
                for net in board_data.nets:
                    if hasattr(net, 'pads') and len(net.pads) >= 2:
                        pad_names = []
                        for pad in net.pads:
                            # Create unique pad identifier
                            pad_name = getattr(pad, 'net', net.name)
                            if not pad_name:
                                pad_name = f"{net.name}_pad_{len(pad_names)}"
                            pad_names.append(pad_name)
                        
                        nets.append({
                            'name': net.name,
                            'pads': pad_names[:2],  # For now, just route first two pads
                            'priority': getattr(net, 'priority', 1)
                        })
                        
            elif hasattr(board_data, 'airwires'):
                # Extract from airwires
                for i, airwire in enumerate(board_data.airwires):
                    if hasattr(airwire, 'net') and hasattr(airwire, 'source') and hasattr(airwire, 'sink'):
                        nets.append({
                            'name': airwire.net,
                            'pads': [airwire.source, airwire.sink],
                            'priority': 1
                        })
                        
            logger.info(f"Extracted {len(nets)} nets for routing")
            return nets
            
        except Exception as e:
            logger.error(f"Error extracting nets: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get router statistics"""
        if not self.is_initialized or not self.gpu_router:
            return {'error': 'Router not initialized'}
        
        return {
            'initialized': self.is_initialized,
            'config': {
                'pitch': self.config.pitch,
                'max_memory_gb': self.config.max_memory_gb,
                'layers': self.config.layers
            },
            'grid': {
                'cols': self.gpu_router.cols,
                'rows': self.gpu_router.rows, 
                'layers': self.gpu_router.layers,
                'total_cells': self.gpu_router.cols * self.gpu_router.rows * self.gpu_router.layers
            },
            'memory': self.gpu_router.get_memory_usage()
        }
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.gpu_router:
            self.gpu_router.cleanup()
            self.gpu_router = None
        self.is_initialized = False
        logger.info("Dense GPU router cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()