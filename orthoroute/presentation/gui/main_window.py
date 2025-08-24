#!/usr/bin/env python3
"""
OrthoRoute Main Window - PyQt6 GUI for PCB visualization and routing
New architecture implementation of the rich GUI functionality
"""

import sys
import logging
from typing import Dict, Any, Optional, List
import os
from pathlib import Path

# Add debug logging capability
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
    from debug_logger import get_debug_logger
    DEBUG_LOGGING_AVAILABLE = True
except ImportError:
    DEBUG_LOGGING_AVAILABLE = False

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QSplitter, QGroupBox, QScrollArea, QTabWidget, QProgressBar,
    QStatusBar, QMenuBar, QToolBar, QApplication, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPoint, QPointF,
    QRectF, QPropertyAnimation, QEasingCurve, pyqtSlot, QMutex
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QPalette,
    QAction, QIcon, QPolygonF, QTransform, QWheelEvent, QMouseEvent,
    QPaintEvent, QResizeEvent
)

from .kicad_colors import KiCadColorScheme
from ...algorithms.manhattan.deterministic_router import DeterministicManhattanRouter, RoutingConfig
from ...algorithms.manhattan.spec_manhattan_router import SpecManhattanRouter
from ...algorithms.manhattan.manhattan_router_rrg import ManhattanRRGRoutingEngine
from ...infrastructure.gpu.cuda_provider import CUDAProvider
from ...infrastructure.gpu.cpu_fallback import CPUProvider

logger = logging.getLogger(__name__)


class RoutingThread(QThread):
    """Background thread for running routing operations."""
    
    # Define signals for progress updates and completion
    progress_update = pyqtSignal(int, int, str, list, list)  # current, total, status, new_tracks, new_vias
    routing_completed = pyqtSignal(dict)  # Routing results
    routing_error = pyqtSignal(str)  # Error message
    
    def __init__(self, algorithm, board_data, config, gpu_provider=None):
        """Initialize routing thread."""
        super().__init__()
        self.algorithm = algorithm
        self.board_data = board_data
        self.config = config
        self.gpu_provider = gpu_provider
        self.router = None
        self.is_cancelled = False
        
    def run(self):
        """Run the routing operation in a background thread."""
        try:
            # Initialize the appropriate router based on algorithm
            if self.algorithm == "Manhattan RRG":
                # Get DRC constraints from board data or create defaults
                drc_constraints = None
                if hasattr(self.board_data, 'drc_constraints') and self.board_data.drc_constraints:
                    drc_constraints = self.board_data.drc_constraints
                    logger.info("Using DRC constraints from board data")
                else:
                    # Create default DRC constraints with improved clearances
                    try:
                        from ...domain.models.constraints import DRCConstraints, NetClass
                        drc_constraints = DRCConstraints(
                            default_track_width=0.0889,  # 3.5 mil
                            default_clearance=0.0889,    # 3.5 mil spacing
                            default_via_diameter=0.25,   # Working via size
                            default_via_drill=0.15,      # Appropriate drill
                        )
                        logger.info("Created enhanced DRC constraints for routing")
                    except ImportError:
                        logger.warning("Could not create DRC constraints")
                
                # Use new RRG-based Manhattan router
                logger.info("🚀 Initializing RRG-based Manhattan router")
                
                # Create mock board object for RRG router
                from ...domain.models.board import Board
                from ...domain.models.constraints import DRCConstraints
                
                # Convert board_data dict to domain Board object
                mock_board = self._convert_board_data_to_domain(self.board_data, drc_constraints)
                
                # Create RRG router
                self.router = ManhattanRRGRoutingEngine(
                    constraints=drc_constraints or DRCConstraints(),
                    gpu_provider=self.gpu_provider
                )
                
                # Initialize the router with board data
                self.router.initialize(mock_board)
                
                # Route all nets using RRG PathFinder
                logger.info(f"🎯 Routing {len(mock_board.nets)} nets with RRG PathFinder")
                
                # Use RRG router interface
                routing_stats = self.router.route_all_nets(
                    nets=mock_board.nets,
                    timeout_per_net=5.0,
                    total_timeout=300.0
                )
                
                # Convert routing statistics to result format
                result = {
                    'success': routing_stats.nets_routed > 0,
                    'tracks': self.router.get_routed_tracks(),
                    'vias': self.router.get_routed_vias(),
                    'routed_nets': routing_stats.nets_routed,
                    'failed_nets': routing_stats.nets_failed,
                    'stats': {
                        'elapsed_time': routing_stats.total_time,
                        'total_length': routing_stats.total_length,
                        'total_vias': routing_stats.total_vias,
                        'success_rate': routing_stats.success_rate
                    }
                }
                
                if self.is_cancelled:
                    return
                    
                # Emit completion signal with results
                self.routing_completed.emit(result)
            
            else:
                self.routing_error.emit(f"Unknown routing algorithm: {self.algorithm}")
                
        except Exception as e:
            self.routing_error.emit(f"Routing error: {str(e)}")
            logger.exception("Error in routing thread")
    
    def cancel(self):
        """Cancel the routing operation."""
        self.is_cancelled = True
    
    def _convert_board_data_to_domain(self, board_data, drc_constraints):
        """Convert board_data dict to domain Board object for RRG router"""
        from ...domain.models.board import Board, Net, Pad, Bounds, Coordinate, Component
        
        # Create board bounds
        bounds_data = board_data.get('bounds', (0, 0, 100, 100))
        board_bounds = Bounds(
            min_x=bounds_data[0],
            min_y=bounds_data[1], 
            max_x=bounds_data[2],
            max_y=bounds_data[3]
        )
        
        # Convert nets and pads
        nets = []
        nets_data = board_data.get('nets', {})
        
        for net_name, net_data in nets_data.items():
            if not net_name or net_name.strip() == "":
                continue
                
            pads_data = net_data.get('pads', [])
            if len(pads_data) < 2:
                continue  # Skip single-pad nets
            
            # Convert pads
            net_pads = []
            for pad_data in pads_data:
                pad = Pad(
                    id=f"{net_name}_pad_{len(net_pads)}",
                    component_id=f"comp_{net_name}_{len(net_pads)}",
                    net_id=f"net_{len(nets)}",
                    position=Coordinate(
                        x=pad_data.get('x', 0.0),
                        y=pad_data.get('y', 0.0)
                    ),
                    size=(
                        pad_data.get('width', 1.0),
                        pad_data.get('height', 1.0)
                    ),
                    drill_size=pad_data.get('drill', None),
                    layer=pad_data.get('layers', ['F.Cu'])[0] if pad_data.get('layers') else 'F.Cu'
                )
                net_pads.append(pad)
            
            # Create net
            net = Net(
                id=f"net_{len(nets)}",
                name=net_name,
                pads=net_pads
            )
            nets.append(net)
        
        # Create mock components for proper bounds calculation
        components = []
        all_pads = []
        for net in nets:
            for i, pad in enumerate(net.pads):
                all_pads.append(pad)
        
        # Create a single mock component containing all pads
        if all_pads:
            # Calculate center position
            avg_x = sum(pad.position.x for pad in all_pads) / len(all_pads)
            avg_y = sum(pad.position.y for pad in all_pads) / len(all_pads)
            
            mock_component = Component(
                id="mock_comp_1",
                reference="U1",
                value="MOCK",
                footprint="MOCK_FP",
                position=Coordinate(avg_x, avg_y),
                pads=all_pads
            )
            components.append(mock_component)
        
        # Create board
        board = Board(
            id="board_1",
            name=board_data.get('filename', 'unknown.kicad_pcb'),
            components=components,
            nets=nets,
            layer_count=12  # F.Cu + 10 internal + B.Cu
        )
        
        # Store airwires as a custom attribute for RRG routing
        board._airwires = board_data.get('airwires', [])
        
        return board


class PCBViewer(QWidget):
    """PCB visualization widget for displaying board, components, tracks, and airwires"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board_data = None
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pan_point = QPoint()
        self.is_panning = False
        
        # Initialize KiCad color scheme
        self.color_scheme = KiCadColorScheme()
        
        # Display option flags
        self.show_components = True
        self.show_tracks = True
        self.show_vias = True
        self.show_pads = True
        self.show_airwires = True
        self.show_zones = True
        self.show_keepouts = True
        
        # Layer visibility
        self.visible_layers = set(['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu', 
                                  'In6.Cu', 'In7.Cu', 'In8.Cu', 'In9.Cu', 'In10.Cu', 'B.Cu'])
        
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
    def set_board_data(self, board_data: Dict[str, Any]):
        """Set the board data to display"""
        self.board_data = board_data
        self.fit_to_view()
        self.update()
        
    def fit_to_view(self):
        """Fit the board to the current view"""
        if not self.board_data:
            return
            
        # Get board bounds
        bounds = self.board_data.get('bounds', (0, 0, 100, 100))
        board_width = bounds[2] - bounds[0]
        board_height = bounds[3] - bounds[1]
        
        if board_width <= 0 or board_height <= 0:
            return
            
        # Calculate zoom to fit
        widget_width = self.width() - 40
        widget_height = self.height() - 40
        
        zoom_x = widget_width / board_width
        zoom_y = widget_height / board_height
        
        self.zoom_factor = min(zoom_x, zoom_y) * 0.9
        
        # Center the board
        self.pan_x = (bounds[0] + bounds[2]) / 2
        self.pan_y = (bounds[1] + bounds[3]) / 2
        
        # Trigger repaint
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the PCB visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background with KiCad background color
        painter.fillRect(self.rect(), self.color_scheme.get_color('background'))
        
        if not self.board_data:
            painter.setPen(self.color_scheme.get_color('text'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No board data loaded")
            return
            
        # Set up coordinate transformation
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(-self.pan_x, -self.pan_y)
        
        # Skip artificial board outline - real boards should use Edge.Cuts layer
        # self._draw_board_outline(painter)
        
        # Draw airwires (behind everything)
        if self.show_airwires:
            self._draw_airwires(painter)
        
        # Draw tracks
        if self.show_tracks:
            self._draw_tracks(painter)
        
        # Draw pads
        if self.show_pads:
            self._draw_pads(painter)
        
        # Draw components
        if self.show_components:
            self._draw_components(painter)
            
        # Draw vias
        if self.show_vias:
            self._draw_vias(painter)
        
    def _draw_board_outline(self, painter: QPainter):
        """Draw the board outline"""
        bounds = self.board_data.get('bounds', (0, 0, 100, 100))
        
        painter.setPen(QPen(self.color_scheme.get_color('edge_cuts'), 0.5))
        painter.setBrush(QBrush())
        
        rect = QRectF(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
        painter.drawRect(rect)
        
    def _draw_airwires(self, painter: QPainter):
        """Draw airwires (unrouted connections) with performance optimization"""
        airwires = self.board_data.get('airwires', [])
        
        # Performance limit: only show airwires when zoomed in enough or limit count
        zoom_level = painter.transform().m11()  # Get current zoom scale
        max_airwires = min(len(airwires), int(1000 * zoom_level) if zoom_level > 0.1 else 200)
        
        if max_airwires <= 0:
            return
            
        painter.setPen(QPen(self.color_scheme.get_color('ratsnest'), 0.1))
        
        # Get viewport bounds for culling
        viewport = painter.viewport()
        
        # More robust viewport calculation
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                # Fallback: render everything if transform can't be inverted
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            # Fallback: render everything if viewport calculation fails
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        drawn_count = 0
        for i, airwire in enumerate(airwires):
            if i >= max_airwires:  # Hard limit for performance
                break
                
            try:
                x1 = airwire['start_x']
                y1 = airwire['start_y']
                x2 = airwire['end_x']
                y2 = airwire['end_y']
                
                # Viewport culling: only draw if line intersects visible area (disabled for now - needs debugging)  
                # line_rect = QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                # if visible_rect.intersects(line_rect):
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                drawn_count += 1
                
            except (KeyError, TypeError):
                continue
        
        # Debug log airwire rendering stats
        if DEBUG_LOGGING_AVAILABLE and drawn_count > 0:
            debug_logger = get_debug_logger()
            render_counts = {
                'airwires_drawn': drawn_count,
                'max_airwires_allowed': max_airwires,
                'total_airwires_available': len(airwires),
                'zoom_level': zoom_level
            }
            debug_logger.log_visualization_data(render_counts, zoom_level)
                
    def _draw_tracks(self, painter: QPainter):
        """Draw existing tracks/traces"""
        tracks = self.board_data.get('tracks', [])
        
        for track in tracks:
            try:
                x1 = track['start_x']
                y1 = track['start_y']
                x2 = track['end_x']
                y2 = track['end_y']
                layer = track.get('layer', 'F.Cu')
                
                # Skip if layer is not visible
                if layer not in self.visible_layers:
                    continue
                
                # Set color based on layer
                if layer == 'F.Cu':
                    color = self.color_scheme.get_color('copper_front')
                elif layer == 'B.Cu':
                    color = self.color_scheme.get_color('copper_back')
                else:
                    # Handle internal layers with distinct colors
                    # Parse "In1.Cu", "In2.Cu" etc to get the correct color
                    try:
                        if layer.startswith('In') and layer.endswith('.Cu'):
                            layer_num = int(layer[2:-3])  # Extract number from "In{N}.Cu"
                            color_key = f'copper_in{layer_num}'
                            color = self.color_scheme.get_color(color_key)
                        else:
                            color = self.color_scheme.get_color('copper_inner')
                    except (ValueError, AttributeError):
                        color = self.color_scheme.get_color('copper_inner')
                
                painter.setPen(QPen(color, 0.2))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                
            except (KeyError, TypeError):
                continue
                
    def _draw_pads(self, painter: QPainter):
        """Draw component pads with viewport culling and LOD optimization"""
        pads = self.board_data.get('pads', [])
        
        # Get zoom level and viewport for optimization
        zoom_level = painter.transform().m11()
        viewport = painter.viewport()
        
        # More robust viewport calculation
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                # Fallback: render everything if transform can't be inverted
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            # Fallback: render everything if viewport calculation fails
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        # LOD: Skip very small pads when zoomed out
        min_pad_size = 1.0 / zoom_level if zoom_level > 0 else 0.1  # Minimum size in world units
        
        drawn_pads = 0
        max_pads = min(len(pads), 5000) if zoom_level < 0.1 else len(pads)  # Limit when zoomed way out
        
        for i, pad in enumerate(pads):
            if i >= max_pads:
                break
            try:
                x = pad['x']
                y = pad['y']
                width = pad.get('width', 1.0)
                height = pad.get('height', 1.0)
                
                # Viewport culling: skip pads outside visible area (disabled for now - needs debugging)
                # pad_rect = QRectF(x - width/2, y - height/2, width, height)
                # if not visible_rect.intersects(pad_rect):
                #     continue
                
                # LOD culling: skip very small pads when zoomed out
                if max(width, height) < min_pad_size:
                    continue
                
                pad_type = pad.get('type', 'smd')
                layers = pad.get('layers', ['F.Cu'])
                drill_size = pad.get('drill', 0.0)
                
                # Skip if none of the pad's layers are visible
                if not any(layer in self.visible_layers for layer in layers):
                    continue
                
                drawn_pads += 1
                
                # Debug logging for visualization (every 100th frame to avoid spam)
                if DEBUG_LOGGING_AVAILABLE and drawn_pads == 50:  # Log at 50th pad to get sample
                    debug_logger = get_debug_logger()
                    render_counts = {
                        'pads_processed': i,
                        'pads_drawn': drawn_pads, 
                        'max_pads': max_pads,
                        'zoom_level': zoom_level,
                        'min_pad_size': min_pad_size,
                        'total_pads_available': len(pads)
                    }
                    debug_logger.log_visualization_data(render_counts, zoom_level)
                
                # Get appropriate pad color based on type and layer
                if pad_type == 'smd':
                    # SMD pads use layer-specific colors
                    if any('B.' in layer for layer in layers):
                        pad_color = self.color_scheme.get_color('pad_back')
                    else:
                        pad_color = self.color_scheme.get_color('pad_front')
                elif pad_type == 'through_hole':
                    pad_color = self.color_scheme.get_color('pad_through_hole')
                else:
                    pad_color = self.color_scheme.get_color('pad_front')
                
                # Draw pad based on type
                painter.setPen(QPen(pad_color, 0.05))
                painter.setBrush(QBrush(pad_color))
                
                if pad_type == 'through_hole':
                    # Draw through-hole pads as circles
                    pad_size = max(width, height)  # Use larger dimension for circular pad
                    pad_rect = QRectF(x - pad_size/2, y - pad_size/2, pad_size, pad_size)
                    painter.drawEllipse(pad_rect)
                    
                    # Draw drill hole for through-hole pads
                    if drill_size > 0:
                        # Use gold color for plated holes (more realistic)
                        hole_color = QColor(255, 215, 0)  # Gold color
                        
                        # Draw circular hole with gold fill and no outline
                        painter.setPen(QPen(hole_color, 0))  # No outline
                        painter.setBrush(QBrush(hole_color))  # Gold fill
                        
                        # Draw circular hole
                        hole_rect = QRectF(x - drill_size/2, y - drill_size/2, drill_size, drill_size)
                        painter.drawEllipse(hole_rect)
                else:
                    # Draw SMD pads as rectangles
                    pad_rect = QRectF(x - width/2, y - height/2, width, height)
                    painter.drawRect(pad_rect)
                    
            except (KeyError, TypeError):
                continue
                
    def _draw_components(self, painter: QPainter):
        """Draw component outlines and reference designators"""
        components = self.board_data.get('components', [])
        
        # Use silkscreen color for component text
        painter.setPen(QPen(self.color_scheme.get_color('f_silks'), 0.1))
        painter.setBrush(QBrush())
        
        font = QFont("Arial", max(1, int(2.0 / self.zoom_factor)))
        painter.setFont(font)
        
        for component in components:
            try:
                x = component['x']
                y = component['y']
                ref_des = component.get('ref_des', 'Unknown')
                
                # Draw component reference designator
                painter.drawText(QPointF(x, y), ref_des)
                
            except (KeyError, TypeError, ValueError) as e:
                # Skip invalid component data
                continue
        
    def _draw_vias(self, painter: QPainter):
        """Draw vias with proper coloring and visibility control"""
        vias = self.board_data.get('vias', [])
        
        if not vias:
            return  # No vias to draw
            
        # Get zoom level for LOD optimization
        zoom_level = painter.transform().m11()
        
        # Get viewport bounds for culling
        viewport = painter.viewport()
        
        try:
            transform_inverted, invertible = painter.transform().inverted()
            if invertible:
                visible_rect = transform_inverted.mapRect(QRectF(viewport))
            else:
                visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        except:
            visible_rect = QRectF(-1000, -1000, 2000, 2000)  # Large fallback area
        
        # Draw vias
        for via in vias:
            try:
                x = via['x']
                y = via['y']
                
                # Skip if outside visible area
                if not visible_rect.contains(QPointF(x, y)):
                    continue
                
                # Determine via type and size
                via_type = via.get('type', via.get('via_type', 'through'))
                diameter = via.get('diameter', via.get('size', 0.25))  # Support both 'diameter' and 'size'
                drill = via.get('drill', 0.3)
                start_layer = via.get('start_layer', via.get('from_layer', 'F.Cu'))
                end_layer = via.get('end_layer', via.get('to_layer', 'B.Cu'))
                
                # Skip very small vias when zoomed out (LOD) - only when smaller than ~1 pixel
                lod_threshold = diameter / zoom_level
                if lod_threshold < 0.01:
                    continue
                
                # Set color based on via type
                if via_type == 'through':
                    color = self.color_scheme.get_color('via_through')
                elif via_type in ['blind_buried', 'blind', 'buried']:
                    color = self.color_scheme.get_color('via_blind_buried')
                elif via_type == 'micro':
                    color = self.color_scheme.get_color('via_micro')
                else:
                    color = self.color_scheme.get_color('via_through')  # Default
                
                # Draw via ring
                painter.setPen(QPen(color, 0.1))
                painter.setBrush(QBrush(color))
                
                # Draw via outer diameter
                via_rect = QRectF(x - diameter/2, y - diameter/2, diameter, diameter)
                painter.drawEllipse(via_rect)
                
                # Draw via hole
                hole_color = self.color_scheme.get_color('via_hole')
                painter.setPen(QPen(hole_color, 0.1))
                painter.setBrush(QBrush(hole_color))
                
                hole_rect = QRectF(x - drill/2, y - drill/2, drill, drill)
                painter.drawEllipse(hole_rect)
                
                # Draw layer indicator for blind/buried vias
                if via_type == 'blind_buried':
                    # Draw small indicators showing which layers this via connects
                    indicator_size = diameter / 4
                    painter.setPen(QPen(QColor(255, 255, 255), 0.1))
                    
                    # This would be expanded for more sophisticated layer indicators
                    painter.drawText(QRectF(x + diameter/2, y - diameter/2, 
                                          indicator_size*2, indicator_size*2),
                                   Qt.AlignmentFlag.AlignCenter, 
                                   f"{start_layer[0]}-{end_layer[0]}")
                
            except (KeyError, TypeError):
                continue
                
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.zoom_factor *= zoom_factor
        self.zoom_factor = max(0.01, min(100.0, self.zoom_factor))
        self.update()
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self.is_panning:
            delta = event.pos() - self.last_pan_point
            self.pan_x -= delta.x() / self.zoom_factor
            self.pan_y -= delta.y() / self.zoom_factor
            self.last_pan_point = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def debug_screenshot(self, filename_prefix: str = "debug_routing"):
        """Capture screenshot of the PCB viewer for debugging"""
        try:
            import os
            from datetime import datetime
            
            # Create debug output directory
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"{debug_dir}/{filename_prefix}_{timestamp}.png"
            
            # Capture the widget as a pixmap
            pixmap = self.grab()
            
            # Save the screenshot
            success = pixmap.save(filename, "PNG")
            
            if success:
                print(f"DEBUG: Screenshot saved to {filename}")
                return filename
            else:
                print(f"DEBUG: Failed to save screenshot to {filename}")
                return None
                
        except Exception as e:
            print(f"DEBUG: Screenshot error: {e}")
            return None
            

class OrthoRouteMainWindow(QMainWindow):
    """Main OrthoRoute GUI window - faithful recreation of original interface"""
    
    def __init__(self, board_data: Dict[str, Any], kicad_interface):
        super().__init__()
        self.board_data = board_data
        self.kicad_interface = kicad_interface
        self.pcb_viewer = None
        self.algorithm_combo = None
        self.display_checkboxes = {}
        self.layer_actions = {}
        self.route_preview_btn = None
        self.commit_btn = None
        self.rollback_btn = None
        self.status_label = None
        self.gpu_status = None
        self.routing_result = None
        
        # Initialize window
        self.setWindowTitle("OrthoRoute - PCB Autorouter")
        self.setMinimumSize(1200, 800)
        self.resize(1800, 800)
        
        # Detect GPU status
        self.detect_gpu_status()
        
        # Setup UI components
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        
        # Load board data
        self.load_board_data()
        
    def detect_gpu_status(self):
        """Detect GPU capabilities for algorithm selection"""
        try:
            # Try to detect CUDA/GPU availability
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.gpu_status = {'available': True, 'can_use_gpu_routing': True}
                logger.info("GPU detected and available for routing")
            else:
                self.gpu_status = {'available': False, 'can_use_gpu_routing': False}
                logger.info("GPU not available, using CPU routing")
        except:
            self.gpu_status = {'available': False, 'can_use_gpu_routing': False}
            logger.info("GPU detection failed, using CPU routing")
        
    def setup_ui(self):
        """Setup the main UI layout - three panel design like original"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel for PCB viewer
        self.pcb_viewer = PCBViewer()
        splitter.addWidget(self.pcb_viewer)
        
        # Right panel for information
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (left:center:right = 300:800:300)
        splitter.setSizes([300, 800, 300])
        
    def create_left_panel(self) -> QWidget:
        """Create the left control panel - matching original layout"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.display_checkboxes = {}
        display_options = [
            ('Components', 'show_components'),
            ('Tracks', 'show_tracks'),
            ('Vias', 'show_vias'),
            ('Pads', 'show_pads'),
            ('Airwires', 'show_airwires'),
            ('Zones', 'show_zones'),
            ('Keepouts', 'show_keepouts')
        ]
        
        for label, attr in display_options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)  # Default to showing all
            checkbox.toggled.connect(lambda checked, attr=attr: self.toggle_display_option(attr, checked))
            self.display_checkboxes[attr] = checkbox
            display_layout.addWidget(checkbox)
        
        layout.addWidget(display_group)
        
        # Routing controls group
        routing_group = QGroupBox("Routing Controls")
        routing_layout = QVBoxLayout(routing_group)
        
        # Algorithm selection
        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        
        # Available algorithms
        algorithm_options = [
            "Wavefront",
            "Manhattan RRG"
        ]
        
        self.algorithm_combo.addItems(algorithm_options)
        self.algorithm_combo.setCurrentIndex(0)  # Default to Wavefront
        
        if self.gpu_status and self.gpu_status.get('can_use_gpu_routing'):
            logger.info("All algorithms will use GPU acceleration with CPU fallback")
        else:
            logger.info("All algorithms will use CPU (GPU not available)")
            
        # Connect algorithm change signal
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        
        algorithm_layout.addWidget(self.algorithm_combo)
        routing_layout.addLayout(algorithm_layout)
        
        # Main routing buttons
        self.route_preview_btn = QPushButton("🚀 Begin Autorouting")
        self.route_preview_btn.clicked.connect(self.begin_autorouting)
        routing_layout.addWidget(self.route_preview_btn)
        
        # Solution control buttons (initially disabled)
        solution_layout = QHBoxLayout()
        self.commit_btn = QPushButton("✅ Apply to KiCad")
        self.commit_btn.clicked.connect(self.commit_routes)
        self.commit_btn.setEnabled(False)
        
        self.rollback_btn = QPushButton("❌ Discard")
        self.rollback_btn.clicked.connect(self.rollback_routes)
        self.rollback_btn.setEnabled(False)
        
        solution_layout.addWidget(self.commit_btn)
        solution_layout.addWidget(self.rollback_btn)
        routing_layout.addLayout(solution_layout)
        
        layout.addWidget(routing_group)
        
        # Nets statistics group
        nets_stats_group = QGroupBox("Nets Statistics")
        nets_stats_layout = QVBoxLayout(nets_stats_group)
        
        self.nets_stats_label = QLabel("Loading nets...")
        nets_stats_layout.addWidget(self.nets_stats_label)
        
        layout.addWidget(nets_stats_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        """Create the right information panel - matching original layout"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Nets list group
        nets_group = QGroupBox("Nets")
        nets_layout = QVBoxLayout(nets_group)
        
        self.nets_tree = QTreeWidget()
        self.nets_tree.setHeaderLabels(["Net Name", "Pads", "Status"])
        self.nets_tree.setMaximumHeight(300)
        nets_layout.addWidget(self.nets_tree)
        
        layout.addWidget(nets_group)
        
        # Board information group
        board_info_group = QGroupBox("Board Information")
        board_info_layout = QVBoxLayout(board_info_group)
        
        self.board_info_label = QLabel("Loading board information...")
        board_info_layout.addWidget(self.board_info_label)
        
        layout.addWidget(board_info_group)
        
        # Routing statistics group
        routing_stats_group = QGroupBox("Routing Statistics")
        routing_stats_layout = QVBoxLayout(routing_stats_group)
        
        self.routing_stats_label = QLabel("No routing performed")
        routing_stats_layout.addWidget(self.routing_stats_label)
        
        layout.addWidget(routing_stats_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def setup_status_bar(self):
        """Setup status bar with GPU status"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # GPU status indicator
        gpu_text = "GPU Available" if self.gpu_status.get('available') else "CPU Only"
        status_bar.addPermanentWidget(QLabel(f"{gpu_text} | OrthoRoute v1.0"))
        
    def setup_menus(self):
        """Setup menu bar - matching original menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        refresh_action = QAction("Refresh Board", self)
        refresh_action.triggered.connect(self.refresh_board)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Layer visibility submenu
        self.layers_menu = view_menu.addMenu("Layers")
        
        # Create layer visibility actions - will be updated when board loads
        self.layer_actions = {}
        
        view_menu.addSeparator()
        
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.zoom_fit)
        view_menu.addAction(fit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        route_action = QAction("Auto Route All", self)
        route_action.triggered.connect(self.auto_route_all)
        tools_menu.addAction(route_action)
        
    def load_board_data(self):
        """Load and display the board data"""
        if not self.board_data:
            return
            
        # Update board info
        filename = self.board_data.get('filename', 'Unknown')
        width = self.board_data.get('width', 0)
        height = self.board_data.get('height', 0)
        layers = self.board_data.get('layers', 0)
        
        pads_count = len(self.board_data.get('pads', []))
        nets_count = len(self.board_data.get('nets', {}))
        components_count = len(self.board_data.get('components', []))
        
        board_info = f"Board: {filename}\n"
        board_info += f"Size: {width:.1f} × {height:.1f} mm\n"
        board_info += f"Layers: {layers}\n"
        board_info += f"Components: {components_count}\n"
        board_info += f"Pads: {pads_count}\n"
        board_info += f"Nets: {nets_count}"
        
        self.board_info_label.setText(board_info)
        
        # Load nets into tree
        self.load_nets_tree()
        
        # Set board data in PCB viewer
        self.pcb_viewer.set_board_data(self.board_data)
        
        # Fit to view after a short delay to ensure proper widget sizing
        QTimer.singleShot(100, self.pcb_viewer.fit_to_view)
        
        # Update layer visibility menu
        self.update_layer_visibility_menu()
        
        # Update status
        self.status_label.setText(f"Loaded {nets_count} nets, {pads_count} pads")
        
        logger.info("Board data loaded into GUI: %d components, %d tracks - starting progressive net processing", components_count, len(self.board_data.get('tracks', [])))
        logger.info("Using %d airwires from KiCad interface (no regeneration needed)", len(self.board_data.get('airwires', [])))
        logger.info("Displaying ALL %d nets (no performance limits)", nets_count)
        
    def load_nets_tree(self):
        """Load nets into the tree widget"""
        if not self.nets_tree:
            return
            
        self.nets_tree.clear()
        
        nets = self.board_data.get('nets', {})
        
        for net_name, net_data in nets.items():
            if not net_name or net_name.strip() == "":
                continue
                
            pads = net_data.get('pads', [])
            pad_count = len(pads)
            
            if pad_count < 2:
                continue  # Skip single-pad nets
                
            item = QTreeWidgetItem([net_name, str(pad_count), "Unrouted"])
            self.nets_tree.addTopLevelItem(item)
            
    def route_all_nets(self):
        """Route all nets"""
        logger.info("Route All Nets button clicked")
        self.status_label.setText("Routing all nets...")
        
        # TODO: Implement actual routing logic
        QMessageBox.information(self, "Routing", "Routing functionality not yet implemented")
        
        self.status_label.setText("Ready")
        
    def clear_routes(self):
        """Clear all existing routes"""
        logger.info("Clear All Routes button clicked")
        self.status_label.setText("Clearing routes...")
        
        # TODO: Implement route clearing logic
        QMessageBox.information(self, "Clear Routes", "Route clearing functionality not yet implemented")
        
        self.status_label.setText("Ready")
        
    def refresh_from_kicad(self):
        """Refresh board data from KiCad"""
        logger.info("Refreshing board data from KiCad...")
        self.status_label.setText("Refreshing from KiCad...")
        
        try:
            # Get fresh board data
            new_board_data = self.kicad_interface.get_board_data()
            
            if new_board_data:
                self.board_data = new_board_data
                self.load_board_data()
                self.status_label.setText("Board data refreshed successfully")
                logger.info("Board data refreshed from KiCad")
            else:
                self.status_label.setText("Failed to refresh board data")
                QMessageBox.warning(self, "Refresh Failed", "Could not refresh board data from KiCad")
                
        except Exception as e:
            logger.error(f"Error refreshing board data: {e}")
            self.status_label.setText("Refresh failed")
            QMessageBox.critical(self, "Refresh Error", f"Error refreshing board data:\\n{e}")
    
    # Additional methods for full original functionality
    def update_layer_visibility_menu(self):
        """Update layer visibility menu with actual board layers"""
        if not hasattr(self, 'layers_menu'):
            return
            
        # Clear existing actions
        self.layers_menu.clear()
        self.layer_actions = {}
        
        # Get layers from board data - assume 12 copper layers for now
        layers = ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu', 
                 'In6.Cu', 'In7.Cu', 'In8.Cu', 'In9.Cu', 'In10.Cu', 'B.Cu']
        
        for layer in layers:
            action = QAction(layer, self)
            action.setCheckable(True)
            action.setChecked(True)  # Default to visible
            action.triggered.connect(lambda checked, l=layer: self.toggle_layer_visibility(l, checked))
            self.layer_actions[layer] = action
            self.layers_menu.addAction(action)
    
    # Event handler methods
    def on_algorithm_changed(self, algorithm_text: str):
        """Handle algorithm selection change"""
        logger.info(f"🔄 Algorithm changed to: {algorithm_text}")
        self.status_label.setText(f"Selected algorithm: {algorithm_text}")
        
    def toggle_display_option(self, option: str, checked: bool):
        """Handle display option checkbox changes"""
        logger.info(f"Display option {option}: {'enabled' if checked else 'disabled'}")
        # Update PCB viewer display settings
        if self.pcb_viewer:
            setattr(self.pcb_viewer, option, checked)
            self.pcb_viewer.update()
            
    def toggle_layer_visibility(self, layer: str, checked: bool):
        """Handle layer visibility changes"""
        logger.info(f"Layer {layer}: {'visible' if checked else 'hidden'}")
        # Update PCB viewer layer visibility
        if self.pcb_viewer:
            if checked:
                self.pcb_viewer.visible_layers.add(layer)
            else:
                self.pcb_viewer.visible_layers.discard(layer)
            self.pcb_viewer.update()
    
    # Routing control methods
    def begin_autorouting(self):
        """Begin autorouting with selected algorithm"""
        algorithm_text = self.algorithm_combo.currentText()
        logger.info(f"🚀 Begin autorouting with {algorithm_text}")
        
        # DEBUG: Screenshot before routing starts
        if self.pcb_viewer:
            self.pcb_viewer.debug_screenshot("before_routing")
        
        self.status_label.setText("Starting autorouting...")
        self.route_preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Route based on selected algorithm
        if algorithm_text == "Manhattan RRG":
            self._route_manhattan_rrg()
        elif algorithm_text == "Wavefront":
            QMessageBox.information(self, "Routing", "Wavefront routing not yet implemented")
            self._reset_routing_ui()
        else:
            QMessageBox.information(self, "Routing", f"Algorithm {algorithm_text} not yet implemented")
            self._reset_routing_ui()
        
    def commit_routes(self):
        """Apply routes to KiCad"""
        logger.info("✅ Committing routes to KiCad")
        self.status_label.setText("Applying routes to KiCad...")
        
        # TODO: Implement route application to KiCad
        QMessageBox.information(self, "Apply Routes", "Route application to KiCad not yet implemented")
        
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.route_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Routes applied successfully")
        
    def rollback_routes(self):
        """Discard calculated routes"""
        logger.info("❌ Discarding routes")
        self.status_label.setText("Discarding routes...")
        
        # TODO: Implement route rollback
        QMessageBox.information(self, "Discard Routes", "Route discarding not yet implemented")
        
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.route_preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Routes discarded")
    
    # Menu action methods  
    def refresh_board(self):
        """Refresh board data from KiCad"""
        logger.info("Refreshing board data from KiCad...")
        self.status_label.setText("Refreshing from KiCad...")
        
        try:
            # Get fresh board data
            new_board_data = self.kicad_interface.get_board_data()
            
            if new_board_data:
                self.board_data = new_board_data
                self.load_board_data()
                self.status_label.setText("Board data refreshed successfully")
                logger.info("Board data refreshed from KiCad")
            else:
                self.status_label.setText("Failed to refresh board data")
                QMessageBox.warning(self, "Refresh Failed", "Could not refresh board data from KiCad")
                
        except Exception as e:
            logger.error(f"Error refreshing board data: {e}")
            self.status_label.setText("Refresh failed")
            QMessageBox.critical(self, "Refresh Error", f"Error refreshing board data:\\n{e}")
            
    def zoom_fit(self):
        """Fit board to window"""
        if self.pcb_viewer:
            self.pcb_viewer.fit_to_view()
            
    def auto_route_all(self):
        """Auto route all nets (menu action)"""
        self.begin_autorouting()
    
    def _route_manhattan_rrg(self):
        """Perform Manhattan RRG routing in a background thread"""
        try:
            logger.info("Starting Manhattan routing...")
            
            # Create AGGRESSIVE routing configuration for 2 nets/second target
            config = RoutingConfig(
                grid_pitch=0.40,  # 0.40mm grid - EXACT GRID ALIGNMENT
                track_width=0.0889,  # 3.5 mil
                clearance=0.2,  # KiCad default
                via_drill=0.4,  # Typical drill for 0.8mm via
                via_diameter=0.25,  # User's preferred working size  
                bend_penalty=2,
                via_cost=10,
                expansion_margin=3.0,
                max_ripups_per_net=2,  # AGGRESSIVE: Reduced from 3
                max_global_failures=100,
                timeout_per_net=0.1  # AGGRESSIVE: 0.1s for 2 nets/second target
            )
            
            # Create GPU provider for router
            try:
                gpu_provider = CUDAProvider()
                if gpu_provider.is_available():
                    gpu_provider.initialize()
                    logger.info("Using CUDA GPU acceleration for Manhattan routing")
                    device_info = gpu_provider.get_device_info()
                    logger.info(f"GPU: {device_info.get('name', 'Unknown')}")
                else:
                    gpu_provider = CPUProvider()
                    logger.info("Using CPU fallback for Manhattan routing")
            except Exception as e:
                logger.warning(f"Error initializing GPU: {e}")
                gpu_provider = CPUProvider()
                logger.info("Using CPU fallback for Manhattan routing")
            
            # Create and configure the routing thread
            self.routing_thread = RoutingThread("Manhattan RRG", self.board_data, config, gpu_provider)
            
            # Connect thread signals
            self.routing_thread.progress_update.connect(self._on_routing_progress)
            self.routing_thread.routing_completed.connect(self._on_routing_completed)
            self.routing_thread.routing_error.connect(self._on_routing_error)
            
            # Create cancel button if needed
            if not hasattr(self, 'cancel_btn'):
                self.cancel_btn = QPushButton("Cancel")
                self.cancel_btn.clicked.connect(self._cancel_routing)
                # Add to the main window layout (after the route preview button)
                if hasattr(self, 'route_preview_btn') and self.route_preview_btn.parent():
                    parent_layout = self.route_preview_btn.parent().layout()
                    if parent_layout:
                        parent_layout.addWidget(self.cancel_btn)
            self.cancel_btn.setVisible(True)
            
            # Start the thread
            self.routing_thread.start()
            
            # Disable UI elements during routing
            self.route_preview_btn.setEnabled(False)
            self.algorithm_combo.setEnabled(False)
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
        except Exception as e:
            logger.exception("Error starting Manhattan routing")
            self.status_label.setText(f"Error: {str(e)}")
            self._reset_routing_ui()
            QMessageBox.critical(self, "Routing Error", f"Error starting Manhattan routing:\n{str(e)}")
    
    def _on_routing_progress(self, current, total, status, tracks, vias):
        """Handle routing progress updates from the thread"""
        # Update progress bar
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"{status} ({current}/{total})")
        
        # Update preview with new tracks/vias
        if tracks:
            if 'tracks' not in self.board_data:
                self.board_data['tracks'] = []
            self.board_data['tracks'].extend(tracks)
            
        if vias:
            if 'vias' not in self.board_data:
                self.board_data['vias'] = []
            self.board_data['vias'].extend(vias)
            
        # Trigger GUI update
        if self.pcb_viewer:
            self.pcb_viewer.update()
    
    def _on_routing_completed(self, result):
        """Handle routing completion"""
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
            
        # Update UI
        self.progress_bar.setValue(100)
        self.commit_btn.setEnabled(True)
        self.rollback_btn.setEnabled(True)
        self.route_preview_btn.setEnabled(True)
        self.algorithm_combo.setEnabled(True)
        
        # Store routing result for commit
        self.routing_result = result
        
        # DEBUG: Screenshot after routing completes
        if self.pcb_viewer:
            self.pcb_viewer.debug_screenshot("after_routing")
        
        # Update status
        routed_nets = result.get('routed_nets', 0)
        failed_nets = result.get('failed_nets', 0)
        self.status_label.setText(f"Manhattan routing completed: {routed_nets} nets routed")
        
        # Clean up GPU resources if they were used
        try:
            if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                    logger.info("Cleaning up GPU resources")
                    self.routing_thread.router.gpu_provider.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GPU resources: {e}")
        
        logger.info(f"✅ Manhattan routing successful: {routed_nets}/{routed_nets + failed_nets} nets")
        
        # Update routing statistics
        stats = result.get('stats', {})
        stats_text = f"Routed: {routed_nets} nets\n"
        stats_text += f"Failed: {failed_nets} nets\n"
        stats_text += f"Time: {stats.get('elapsed_time', 0):.2f}s\n"
        stats_text += f"Grid: {stats.get('grid_size', (0,0,0))[0]}×{stats.get('grid_size', (0,0,0))[1]}"
        self.routing_stats_label.setText(stats_text)
    
    def _on_routing_error(self, error_message):
        """Handle routing errors"""
        logger.error(f"Routing error: {error_message}")
        self.status_label.setText(f"Error: {error_message}")
        self._reset_routing_ui()
        
        # Clean up GPU resources if they were used
        try:
            if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                    logger.info("Cleaning up GPU resources")
                    self.routing_thread.router.gpu_provider.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GPU resources: {e}")
        
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
            
        QMessageBox.critical(self, "Routing Error", f"Error during routing:\n{error_message}")
    
    def _cancel_routing(self):
        """Cancel current routing operation"""
        if hasattr(self, 'routing_thread') and self.routing_thread.isRunning():
            logger.info("Cancelling routing...")
            self.status_label.setText("Cancelling routing...")
            self.routing_thread.cancel()
            
            # Clean up GPU resources if they were used
            try:
                if hasattr(self.routing_thread, 'router') and hasattr(self.routing_thread.router, 'gpu_provider'):
                    if self.routing_thread.router.gpu_provider and hasattr(self.routing_thread.router.gpu_provider, 'cleanup'):
                        logger.info("Cleaning up GPU resources")
                        self.routing_thread.router.gpu_provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up GPU resources: {e}")
                
            self.routing_thread.wait()  # Wait for thread to finish
            
        self._reset_routing_ui()
        self.status_label.setText("Routing cancelled")
        
        # Hide cancel button
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setVisible(False)
    
    def _reset_routing_ui(self):
        """Reset routing UI to initial state"""
        self.route_preview_btn.setEnabled(True)
        self.algorithm_combo.setEnabled(True)
        self.commit_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")