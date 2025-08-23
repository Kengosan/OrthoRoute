"""KiCad color theme integration for OrthoRoute GUI."""

import json
import os
from typing import Dict, Any, Tuple
from PyQt6.QtGui import QColor


class KiCadColorScheme:
    """Manages KiCad color scheme from theme JSON file."""
    
    def __init__(self, theme_file_path: str = None):
        """Initialize color scheme from KiCad theme file."""
        if theme_file_path is None:
            # Default to graphics/kicad_theme.json relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            theme_file_path = os.path.join(project_root, 'graphics', 'kicad_theme.json')
        
        self.theme_data = self._load_theme_file(theme_file_path)
        self.board_colors = self.theme_data.get('board', {})
        self._init_qcolors()
        
    def _load_theme_file(self, file_path: str) -> Dict[str, Any]:
        """Load theme data from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load KiCad theme file {file_path}: {e}")
            # Return default minimal color scheme
            return {
                'board': {
                    'background': 'rgb(0, 16, 35)',
                    'pad_front': 'rgb(200, 52, 52)',
                    'pad_back': 'rgb(77, 127, 196)',
                    'pad_through_hole': 'rgb(227, 183, 46)',
                    'pad_plated_hole': 'rgb(194, 194, 0)',
                    'copper': {'f': 'rgb(200, 52, 52)', 'b': 'rgb(77, 127, 196)'},
                    'ratsnest': 'rgba(245, 255, 213, 0.702)',
                    'edge_cuts': 'rgb(208, 210, 205)'
                }
            }
    
    def _init_qcolors(self):
        """Initialize QColor objects from theme data."""
        self.colors = {}
        
        # Background
        self.colors['background'] = self._parse_color(self.board_colors.get('background', 'rgb(0, 16, 35)'))
        
        # Pads
        self.colors['pad_front'] = self._parse_color(self.board_colors.get('pad_front', 'rgb(200, 52, 52)'))
        self.colors['pad_back'] = self._parse_color(self.board_colors.get('pad_back', 'rgb(77, 127, 196)'))
        self.colors['pad_through_hole'] = self._parse_color(self.board_colors.get('pad_through_hole', 'rgb(227, 183, 46)'))
        self.colors['pad_plated_hole'] = self._parse_color(self.board_colors.get('pad_plated_hole', 'rgb(194, 194, 0)'))
        
        # Holes
        self.colors['via_hole'] = self._parse_color(self.board_colors.get('via_hole', 'rgb(227, 183, 46)'))
        self.colors['plated_hole'] = self._parse_color(self.board_colors.get('plated_hole', 'rgb(26, 196, 210)'))
        
        # Copper layers
        copper_colors = self.board_colors.get('copper', {})
        self.colors['copper_front'] = self._parse_color(copper_colors.get('f', 'rgb(200, 52, 52)'))
        self.colors['copper_back'] = self._parse_color(copper_colors.get('b', 'rgb(77, 127, 196)'))
        
        # Internal copper layers with distinct colors
        # Use a predefined color palette to ensure each layer has a unique color
        internal_layer_colors = [
            'rgb(194, 0, 194)',    # Magenta (In1)
            'rgb(0, 132, 132)',    # Teal (In2)
            'rgb(180, 90, 0)',     # Brown (In3)
            'rgb(100, 180, 0)',    # Green-Yellow (In4)
            'rgb(0, 0, 132)',      # Navy (In5)
            'rgb(132, 0, 0)',      # Dark Red (In6)
            'rgb(132, 132, 0)',    # Olive (In7)
            'rgb(0, 0, 255)',      # Blue (In8)
            'rgb(255, 0, 0)',      # Red (In9)
            'rgb(0, 132, 0)',      # Green (In10)
            'rgb(132, 0, 132)',    # Purple (In11)
            'rgb(255, 132, 0)',    # Orange (In12)
            'rgb(0, 132, 255)',    # Light Blue (In13)
            'rgb(132, 132, 132)',  # Gray (In14)
            'rgb(255, 0, 132)',    # Pink (In15)
        ]
        
        # Apply colors for internal layers
        for i in range(1, 31):  # KiCad supports up to 30 internal layers
            layer_key = f'in{i}'
            default_color = internal_layer_colors[(i-1) % len(internal_layer_colors)]
            if layer_key in copper_colors:
                self.colors[f'copper_in{i}'] = self._parse_color(copper_colors[layer_key])
            else:
                self.colors[f'copper_in{i}'] = self._parse_color(default_color)
        
        # Create a generic inner layer color as fallback
        self.colors['copper_inner'] = self._parse_color('rgb(0, 132, 132)')
        self.colors['ratsnest'] = self._parse_color(self.board_colors.get('ratsnest', 'rgba(245, 255, 213, 0.702)'))
        
        # Board edge
        self.colors['edge_cuts'] = self._parse_color(self.board_colors.get('edge_cuts', 'rgb(208, 210, 205)'))
        
        # Vias
        self.colors['via_through'] = self._parse_color(self.board_colors.get('via_through', 'rgb(236, 236, 236)'))
        self.colors['via_blind_buried'] = self._parse_color(self.board_colors.get('via_blind_buried', 'rgb(187, 151, 38)'))
        self.colors['via_micro'] = self._parse_color(self.board_colors.get('via_micro', 'rgb(0, 132, 132)'))
        
        # Silkscreen
        self.colors['f_silks'] = self._parse_color(self.board_colors.get('f_silks', 'rgb(242, 237, 161)'))
        self.colors['b_silks'] = self._parse_color(self.board_colors.get('b_silks', 'rgb(232, 178, 167)'))
        
        # Text
        self.colors['text'] = QColor(255, 255, 255)  # White for contrast
        
    def _parse_color(self, color_str: str) -> QColor:
        """Parse color string (rgb/rgba) into QColor object."""
        if not color_str:
            return QColor(255, 255, 255)
            
        color_str = color_str.strip()
        
        try:
            if color_str.startswith('rgba('):
                # Parse rgba(r, g, b, alpha)
                values = color_str[5:-1].split(',')
                if len(values) == 4:
                    r, g, b, a = [float(v.strip()) for v in values]
                    return QColor(int(r), int(g), int(b), int(a * 255))
            elif color_str.startswith('rgb('):
                # Parse rgb(r, g, b)
                values = color_str[4:-1].split(',')
                if len(values) == 3:
                    r, g, b = [int(v.strip()) for v in values]
                    return QColor(r, g, b)
        except (ValueError, IndexError):
            pass
            
        # Fallback to white
        return QColor(255, 255, 255)
    
    def get_color(self, color_name: str) -> QColor:
        """Get QColor object for named color."""
        return self.colors.get(color_name, QColor(255, 255, 255))
    
    def get_layer_color(self, layer_name: str) -> QColor:
        """Get color for specific PCB layer."""
        layer_mappings = {
            'F.Cu': 'copper_front',
            'B.Cu': 'copper_back',
            'F.SilkS': 'f_silks',
            'B.SilkS': 'b_silks',
            'Edge.Cuts': 'edge_cuts'
        }
        
        # Handle internal copper layers
        if layer_name.startswith('In') and layer_name.endswith('.Cu'):
            try:
                layer_num = int(layer_name[2:-3])
                color_key = f'copper_in{layer_num}'
                if color_key in self.colors:
                    return self.colors[color_key]
            except ValueError:
                pass
        
        color_key = layer_mappings.get(layer_name, 'copper_front')
        return self.get_color(color_key)
    
    def get_pad_color(self, pad_type: str, layer: str = None) -> QColor:
        """Get appropriate pad color based on type and layer."""
        if pad_type == 'smd':
            if layer and 'B.' in layer:
                return self.get_color('pad_back')
            else:
                return self.get_color('pad_front')
        elif pad_type == 'through_hole':
            return self.get_color('pad_through_hole')
        else:
            return self.get_color('pad_front')
    
    def get_hole_color(self, hole_type: str) -> QColor:
        """Get appropriate hole color based on type."""
        if hole_type == 'plated':
            return self.get_color('plated_hole')
        elif hole_type == 'via':
            return self.get_color('via_hole')
        else:
            return self.get_color('pad_plated_hole')