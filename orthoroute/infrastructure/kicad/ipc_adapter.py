"""KiCad 9 IPC API adapter."""
import logging
from typing import Dict, List, Optional, Any

from ...domain.models.board import Board, Component, Net, Pad, Layer, Coordinate
from ...domain.models.constraints import DRCConstraints, NetClass

logger = logging.getLogger(__name__)


class KiCadIPCAdapter:
    """Adapter for KiCad 9 IPC API integration."""
    
    def __init__(self):
        """Initialize IPC adapter."""
        self.connection = None
        self.board_api = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to KiCad via IPC API."""
        try:
            # Try multiple methods to connect to KiCad
            
            # Method 1: Try KiCad 8+ HTTP API (if available)
            if self._try_http_api():
                return True
            
            # Method 2: Try socket-based IPC (KiCad 9+)
            if self._try_socket_ipc():
                return True
                
            # Method 3: Try environment-based connection
            if self._try_env_connection():
                return True
                
            logger.warning("No KiCad IPC connection methods succeeded")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to KiCad IPC API: {e}")
            return False
    
    def _try_http_api(self) -> bool:
        """Try to connect via KiCad HTTP API."""
        try:
            import requests
            
            # Try default KiCad API endpoint
            response = requests.get('http://localhost:5555/api/status', timeout=2)
            if response.status_code == 200:
                self.connection = {'type': 'http', 'base_url': 'http://localhost:5555'}
                self.is_connected = True
                logger.info("Connected to KiCad via HTTP API")
                return True
                
        except Exception as e:
            logger.debug(f"HTTP API connection failed: {e}")
        
        return False
    
    def _try_socket_ipc(self) -> bool:
        """Try to connect via socket IPC."""
        try:
            import socket
            
            # Try to connect to KiCad IPC socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 6000))
            sock.close()
            
            if result == 0:
                self.connection = {'type': 'socket', 'host': 'localhost', 'port': 6000}
                self.is_connected = True
                logger.info("Connected to KiCad via Socket IPC")
                return True
                
        except Exception as e:
            logger.debug(f"Socket IPC connection failed: {e}")
        
        return False
    
    def _try_env_connection(self) -> bool:
        """Try to detect KiCad via environment variables or running processes."""
        try:
            import psutil
            import os
            
            # Check if KiCad process is running
            kicad_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'kicad' in proc.info['name'].lower():
                        kicad_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if kicad_processes:
                logger.info(f"Detected KiCad processes: {len(kicad_processes)}")
                self.connection = {'type': 'process', 'processes': kicad_processes}
                self.is_connected = True
                return True
                
        except Exception as e:
            logger.debug(f"Environment connection failed: {e}")
        
        return False
    
    def disconnect(self):
        """Disconnect from KiCad."""
        if self.connection:
            try:
                self.connection.disconnect()
                logger.info("Disconnected from KiCad IPC API")
            except Exception as e:
                logger.error(f"Error disconnecting from KiCad: {e}")
        
        self.connection = None
        self.board_api = None
        self.is_connected = False
    
    def is_available(self) -> bool:
        """Check if KiCad IPC API is available."""
        try:
            import kicad_ipc_api
            return True
        except ImportError:
            return False
    
    def load_board(self) -> Optional[Board]:
        """Load board from KiCad."""
        if not self.is_connected:
            logger.error("Not connected to KiCad")
            return None
        
        try:
            # Handle different connection types
            if self.connection['type'] == 'http':
                return self._load_board_via_http()
            elif self.connection['type'] == 'socket':
                return self._load_board_via_socket()
            elif self.connection['type'] == 'process':
                return self._load_board_via_process()
            else:
                logger.error(f"Unknown connection type: {self.connection['type']}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading board from KiCad: {e}")
            return None
    
    def _load_board_via_process(self) -> Optional[Board]:
        """Load board by detecting KiCad process and finding current board file."""
        try:
            import psutil
            import os
            import glob
            from pathlib import Path
            
            # Find KiCad processes with command line arguments
            kicad_processes = self.connection.get('processes', [])
            
            # Try to find the current project/board file from KiCad processes
            board_files = []
            
            for proc_info in kicad_processes:
                try:
                    pid = proc_info['pid']
                    proc = psutil.Process(pid)
                    
                    # Check command line for board file
                    cmdline = proc.cmdline()
                    for arg in cmdline:
                        if arg.endswith('.kicad_pcb'):
                            board_files.append(arg)
                            break
                    
                    # Check working directory for recent board files
                    try:
                        cwd = proc.cwd()
                        recent_boards = glob.glob(os.path.join(cwd, "*.kicad_pcb"))
                        if recent_boards:
                            # Sort by modification time, get most recent
                            recent_boards.sort(key=os.path.getmtime, reverse=True)
                            board_files.extend(recent_boards)
                    except (psutil.AccessDenied, OSError):
                        pass
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Use the most recently modified board file
            if board_files:
                # Remove duplicates and sort by modification time
                unique_files = list(set(board_files))
                unique_files.sort(key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0, reverse=True)
                
                board_file = unique_files[0]
                logger.info(f"Loading board file from running KiCad: {board_file}")
                
                # Use file parser to load the board
                from .file_parser import KiCadFileParser
                parser = KiCadFileParser()
                return parser.load_board(board_file)
            else:
                logger.warning("Could not find board file from running KiCad process")
                return None
                
        except Exception as e:
            logger.error(f"Error loading board via process detection: {e}")
            return None
    
    def _load_board_via_http(self) -> Optional[Board]:
        """Load board via HTTP API (placeholder for future KiCad HTTP API)."""
        try:
            import requests
            
            base_url = self.connection['base_url']
            response = requests.get(f"{base_url}/api/board", timeout=10)
            
            if response.status_code == 200:
                board_data = response.json()
                # Convert HTTP API response to Board object
                return self._create_board_from_api_data(board_data)
            else:
                logger.error(f"HTTP API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"HTTP API board loading failed: {e}")
            return None
    
    def _load_board_via_socket(self) -> Optional[Board]:
        """Load board via socket IPC (placeholder for future KiCad socket API)."""
        try:
            import socket
            import json
            
            # Connect to KiCad socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.connection['host'], self.connection['port']))
            
            # Send board info request
            request = {"command": "get_board_info"}
            sock.send(json.dumps(request).encode() + b'\n')
            
            # Receive response
            response = sock.recv(4096).decode()
            sock.close()
            
            board_data = json.loads(response)
            return self._create_board_from_api_data(board_data)
            
        except Exception as e:
            logger.error(f"Socket IPC board loading failed: {e}")
            return None
    
    def get_drc_constraints(self) -> Optional[DRCConstraints]:
        """Extract DRC constraints from KiCad."""
        if not self.is_connected:
            return None
        
        try:
            drc_data = self.board_api.get_design_rules()
            
            constraints = DRCConstraints(
                min_track_width=drc_data.get('min_track_width', 0.1),
                min_track_spacing=drc_data.get('min_track_spacing', 0.1),
                min_via_diameter=drc_data.get('min_via_diameter', 0.2),
                min_via_drill=drc_data.get('min_via_drill', 0.1),
                default_track_width=drc_data.get('default_track_width', 0.2),  # KiCad default
                default_clearance=drc_data.get('default_clearance', 0.2),
                default_via_diameter=drc_data.get('default_via_diameter', 0.8),  # KiCad default
                default_via_drill=drc_data.get('default_via_drill', 0.4)  # Typical for 0.8mm via
            )
            
            # Add netclasses
            for nc_name, nc_data in drc_data.get('netclasses', {}).items():
                netclass = NetClass(
                    name=nc_name,
                    track_width=nc_data.get('track_width', constraints.default_track_width),
                    via_diameter=nc_data.get('via_diameter', constraints.default_via_diameter),
                    via_drill=nc_data.get('via_drill', constraints.default_via_drill),
                    clearance=nc_data.get('clearance', constraints.default_clearance)
                )
                constraints.add_netclass(netclass)
            
            logger.info(f"Extracted DRC constraints with {len(constraints.netclasses)} netclasses")
            return constraints
            
        except Exception as e:
            logger.error(f"Error extracting DRC constraints: {e}")
            return None
    
    def create_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                    layer: str, width: float, net_name: str) -> bool:
        """Create a track in KiCad."""
        if not self.is_connected:
            return False
        
        try:
            track_data = {
                'start': {'x': start_x, 'y': start_y},
                'end': {'x': end_x, 'y': end_y},
                'layer': layer,
                'width': width,
                'net': net_name
            }
            
            self.board_api.create_track(track_data)
            return True
            
        except Exception as e:
            logger.error(f"Error creating track: {e}")
            return False
    
    def create_via(self, x: float, y: float, diameter: float, drill: float,
                  from_layer: str, to_layer: str, net_name: str) -> bool:
        """Create a via in KiCad."""
        if not self.is_connected:
            return False
        
        try:
            via_data = {
                'position': {'x': x, 'y': y},
                'diameter': diameter,
                'drill': drill,
                'layers': [from_layer, to_layer],
                'net': net_name
            }
            
            self.board_api.create_via(via_data)
            return True
            
        except Exception as e:
            logger.error(f"Error creating via: {e}")
            return False
    
    def _create_component_from_data(self, comp_data: Dict) -> Component:
        """Create component from KiCad component data."""
        component = Component(
            id=comp_data.get('id', ''),
            reference=comp_data.get('reference', ''),
            value=comp_data.get('value', ''),
            footprint=comp_data.get('footprint', ''),
            position=Coordinate(
                comp_data.get('x', 0),
                comp_data.get('y', 0)
            ),
            angle=comp_data.get('angle', 0),
            layer=comp_data.get('layer', 'F.Cu')
        )
        
        # Add pads
        for pad_data in comp_data.get('pads', []):
            pad = Pad(
                id=pad_data.get('id', ''),
                component_id=component.id,
                net_id=pad_data.get('net_id'),
                position=Coordinate(
                    pad_data.get('x', 0),
                    pad_data.get('y', 0)
                ),
                size=(pad_data.get('width', 1.0), pad_data.get('height', 1.0)),
                drill_size=pad_data.get('drill_size'),
                layer=pad_data.get('layer', 'F.Cu'),
                shape=pad_data.get('shape', 'circle'),
                angle=pad_data.get('angle', 0)
            )
            component.pads.append(pad)
        
        return component
    
    def _create_net_from_data(self, net_data: Dict, board: Board) -> Optional[Net]:
        """Create net from KiCad net data."""
        net_id = net_data.get('id', '')
        net_name = net_data.get('name', '')
        
        if not net_id or not net_name:
            return None
        
        # Find pads for this net
        net_pads = []
        for component in board.components:
            for pad in component.pads:
                if pad.net_id == net_id:
                    net_pads.append(pad)
        
        if not net_pads:
            return None
        
        net = Net(
            id=net_id,
            name=net_name,
            netclass=net_data.get('netclass', 'Default'),
            pads=net_pads
        )
        
        return net