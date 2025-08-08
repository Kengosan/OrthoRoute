#!/usr/bin/env python3
"""
KiCad IPC Interface - Handles communication with KiCad via IPC API (kipy)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import sys
import time

logger = logging.getLogger(__name__)

@dataclass
class BoardData:
    """Container for board data extracted from KiCad"""
    filename: str
    width: float  # mm
    height: float  # mm
    layers: int
    nets: List[Dict]
    components: List[Dict]
    tracks: List[Dict]
    vias: List[Dict]
    pads: List[Dict]
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def _ipc_retry(func, desc: str, max_retries: int = 3, sleep_s: float = 0.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            last_err = e
            logger.warning(f"IPC '{desc}' failed (attempt {attempt}/{max_retries}): {msg}")
            if "Timed out" in msg or "AS_BUSY" in msg or "busy" in msg.lower():
                time.sleep(sleep_s)
                continue
            break
    if last_err:
        raise last_err


class KiCadInterface:
    """Interface to KiCad via IPC API (kicad-python -> kipy)"""

    def __init__(self):
        self.client = None
        self.board = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to KiCad via IPC API"""
        try:
            # Ensure kipy is importable from user site (common when KiCad launches process)
            try:
                from kipy import KiCad  # type: ignore
            except ImportError:
                import site
                user_site = site.getusersitepackages()
                if user_site and user_site not in sys.path:
                    sys.path.insert(0, user_site)
                from kipy import KiCad  # retry

            # Gather credentials if provided by KiCad runtime
            api_socket = os.environ.get('KICAD_API_SOCKET')
            api_token = os.environ.get('KICAD_API_TOKEN')
            timeout_ms = 25000
            if api_socket or api_token:
                self.client = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=timeout_ms)
            else:
                self.client = KiCad(timeout_ms=timeout_ms)

            # Get board to confirm connection
            self.board = _ipc_retry(self.client.get_board, "get_board", max_retries=3, sleep_s=0.5)
            self.connected = True
            logger.info("✅ Connected to KiCad IPC API and retrieved board")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to KiCad: {e}")
            self.connected = False
            return False

    def get_board_data(self) -> Dict:
        """Extract comprehensive board data from KiCad"""
        if not self.connected or not self.board:
            logger.error("Not connected to KiCad")
            return self._get_fallback_board_data()

        board = self.board

        # File/name
        try:
            filename = getattr(board, 'name', None) or getattr(board, 'filename', 'Untitled')
        except Exception:
            filename = 'Untitled'

        # Components
        components = []
        try:
            fps = _ipc_retry(board.get_footprints, "get_footprints", max_retries=3, sleep_s=0.7)
            for i, fp in enumerate(fps):
                try:
                    pos = getattr(fp, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
                    ref = None
                    try:
                        ref = getattr(getattr(getattr(fp, 'reference_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    val = None
                    try:
                        val = getattr(getattr(getattr(fp, 'value_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    rot = getattr(getattr(fp, 'orientation', None), 'degrees', 0.0)
                    layer = getattr(fp, 'layer', 'F.Cu')
                    components.append({
                        'reference': ref or f'U{i}',
                        'value': val or '',
                        'x': x,
                        'y': y,
                        'rotation': float(rot) if isinstance(rot, (int, float)) else 0.0,
                        'layer': layer
                    })
                except Exception as e:
                    logger.warning(f"Footprint parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting footprints: {e}")

        # Pads (used to derive net pins)
        pads = []
        try:
            all_pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            for i, p in enumerate(all_pads):
                try:
                    pos = getattr(p, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
                    net = getattr(getattr(p, 'net', None), 'name', None)
                    num = getattr(p, 'number', None)
                    pads.append({'net': net, 'number': num, 'x': x, 'y': y})
                except Exception as e:
                    logger.warning(f"Pad parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting pads: {e}")

        # Tracks
        tracks = []
        try:
            trs = _ipc_retry(board.get_tracks, "get_tracks", max_retries=3, sleep_s=0.7)
            for i, tr in enumerate(trs):
                try:
                    start = getattr(tr, 'start', None)
                    end = getattr(tr, 'end', None)
                    s = (float(getattr(start, 'x', 0.0)), float(getattr(start, 'y', 0.0))) if start else (0.0, 0.0)
                    e = (float(getattr(end, 'x', 0.0)), float(getattr(end, 'y', 0.0))) if end else (0.0, 0.0)
                    tracks.append({'start': {'x': s[0], 'y': s[1]}, 'end': {'x': e[0], 'y': e[1]}})
                except Exception as e:
                    logger.warning(f"Track parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting tracks: {e}")

        # Vias
        vias = []
        try:
            vs = _ipc_retry(board.get_vias, "get_vias", max_retries=3, sleep_s=0.7)
            for i, v in enumerate(vs):
                try:
                    pos = getattr(v, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
                    vias.append({'x': x, 'y': y})
                except Exception as e:
                    logger.warning(f"Via parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting vias: {e}")

        # Nets (with pins derived from pads)
        nets = []
        try:
            board_nets = _ipc_retry(board.get_nets, "get_nets", max_retries=3, sleep_s=0.7)
            # Group pads by net
            pins_by_net: Dict[str, List[Dict]] = {}
            for pad in pads:
                n = pad.get('net')
                if not n:
                    continue
                pins_by_net.setdefault(n, []).append({'x': pad['x'], 'y': pad['y'], 'layer': 0, 'pad_name': pad.get('number')})
            for i, net in enumerate(board_nets):
                try:
                    name = getattr(net, 'name', f'Net_{i}')
                    nets.append({
                        'id': i,
                        'name': name,
                        'pins': pins_by_net.get(name, []),
                        'routed': False,
                        'priority': 1
                    })
                except Exception as e:
                    logger.warning(f"Net parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting nets: {e}")

        # Compute bounds from geometry
        min_x = min((p['x'] for p in pads), default=0.0)
        min_y = min((p['y'] for p in pads), default=0.0)
        max_x = max((p['x'] for p in pads), default=100.0)
        max_y = max((p['y'] for p in pads), default=80.0)
        # Include component centers
        if components:
            min_x = min(min_x, min(c['x'] for c in components))
            min_y = min(min_y, min(c['y'] for c in components))
            max_x = max(max_x, max(c['x'] for c in components))
            max_y = max(max_y, max(c['y'] for c in components))
        # Add small margin
        margin = 5.0
        bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)

        board_data = {
            'filename': filename,
            'width': bounds[2] - bounds[0],
            'height': bounds[3] - bounds[1],
            'layers': self._get_layer_count(),
            'nets': nets,
            'components': components,
            'tracks': tracks,
            'vias': vias,
            'pads': pads,
            'bounds': bounds,
            'unrouted_count': len([n for n in nets if not n.get('routed', False)]),
            'routed_count': len([n for n in nets if n.get('routed', False)])
        }

        logger.info(f"Extracted board data: {len(nets)} nets, {len(components)} components, {len(tracks)} tracks")
        return board_data

    def _get_layer_count(self) -> int:
        try:
            return getattr(self.board, 'layer_count', 2)
        except Exception:
            return 2

    # The following are stubs; real creation via IPC will be added later
    def create_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                     layer: str, width: float, net_name: str) -> bool:
        logger.info(f"[stub] Create track {net_name} {start_x:.2f},{start_y:.2f}->{end_x:.2f},{end_y:.2f} {layer} w={width}")
        return True

    def create_via(self, x: float, y: float, size: float, drill: float,
                   from_layer: str, to_layer: str, net_name: str) -> bool:
        logger.info(f"[stub] Create via {net_name} at {x:.2f},{y:.2f} size={size} drill={drill}")
        return True

    def refresh_board(self):
        try:
            # Placeholder for a future refresh call via IPC
            pass
        except Exception as e:
            logger.error(f"Error refreshing board: {e}")

    def _get_fallback_board_data(self) -> Dict:
        """Fallback mock data if IPC connection fails"""
        return {
            'filename': 'Mock_Board.kicad_pcb',
            'width': 100.0,
            'height': 80.0,
            'layers': 4,
            'nets': [],
            'components': [],
            'tracks': [],
            'vias': [],
            'pads': [],
            'bounds': (0.0, 0.0, 100.0, 80.0),
            'unrouted_count': 0,
            'routed_count': 0
        }
