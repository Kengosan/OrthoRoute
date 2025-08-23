"""Query handlers for board data retrieval."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .routing_queries import Query, QueryHandler
from ...domain.models.board import Board, Component, Net, Layer


@dataclass
class GetBoardInfoQuery(Query):
    """Query to get board information."""
    board_id: Optional[str] = None  # None means current board


@dataclass
class GetLayersQuery(Query):
    """Query to get layer information."""
    board_id: Optional[str] = None
    routing_layers_only: bool = False


@dataclass
class GetComponentsQuery(Query):
    """Query to get component information."""
    board_id: Optional[str] = None
    component_ids: Optional[List[str]] = None
    include_pads: bool = True


@dataclass
class GetNetsQuery(Query):
    """Query to get net information."""
    board_id: Optional[str] = None
    net_ids: Optional[List[str]] = None
    routable_only: bool = False


class GetBoardInfoQueryHandler(QueryHandler):
    """Handler for GetBoardInfoQuery."""
    
    def __init__(self, board_repository):
        self.board_repository = board_repository
    
    def handle(self, query: GetBoardInfoQuery) -> Dict[str, Any]:
        """Handle board info query."""
        if query.board_id:
            board = self.board_repository.get_board(query.board_id)
        else:
            board = self.board_repository.get_current_board()
        
        if not board:
            return {}
        
        bounds = board.get_bounds()
        
        return {
            'id': board.id,
            'name': board.name,
            'thickness': board.thickness,
            'layer_count': board.layer_count,
            'component_count': len(board.components),
            'net_count': len(board.nets),
            'routable_net_count': len(board.get_routable_nets()),
            'bounds': {
                'min_x': bounds.min_x,
                'min_y': bounds.min_y,
                'max_x': bounds.max_x,
                'max_y': bounds.max_y,
                'width': bounds.width,
                'height': bounds.height
            },
            'layers': [layer.name for layer in board.layers]
        }


class GetLayersQueryHandler(QueryHandler):
    """Handler for GetLayersQuery."""
    
    def __init__(self, board_repository):
        self.board_repository = board_repository
    
    def handle(self, query: GetLayersQuery) -> List[Dict[str, Any]]:
        """Handle layers query."""
        if query.board_id:
            board = self.board_repository.get_board(query.board_id)
        else:
            board = self.board_repository.get_current_board()
        
        if not board:
            return []
        
        layers = board.layers
        if query.routing_layers_only:
            layers = board.get_routing_layers()
        
        return [{
            'name': layer.name,
            'type': layer.type,
            'stackup_position': layer.stackup_position,
            'thickness': layer.thickness,
            'material': layer.material,
            'is_routing_layer': layer.is_routing_layer
        } for layer in layers]


class GetComponentsQueryHandler(QueryHandler):
    """Handler for GetComponentsQuery."""
    
    def __init__(self, board_repository):
        self.board_repository = board_repository
    
    def handle(self, query: GetComponentsQuery) -> List[Dict[str, Any]]:
        """Handle components query."""
        if query.board_id:
            board = self.board_repository.get_board(query.board_id)
        else:
            board = self.board_repository.get_current_board()
        
        if not board:
            return []
        
        components = board.components
        if query.component_ids:
            components = [c for c in components if c.id in query.component_ids]
        
        result = []
        for component in components:
            bounds = component.get_bounds()
            comp_data = {
                'id': component.id,
                'reference': component.reference,
                'value': component.value,
                'footprint': component.footprint,
                'position': {
                    'x': component.position.x,
                    'y': component.position.y
                },
                'angle': component.angle,
                'layer': component.layer,
                'bounds': {
                    'min_x': bounds.min_x,
                    'min_y': bounds.min_y,
                    'max_x': bounds.max_x,
                    'max_y': bounds.max_y
                },
                'pad_count': len(component.pads)
            }
            
            if query.include_pads:
                comp_data['pads'] = [{
                    'id': pad.id,
                    'net_id': pad.net_id,
                    'position': {
                        'x': pad.position.x,
                        'y': pad.position.y
                    },
                    'size': pad.size,
                    'drill_size': pad.drill_size,
                    'layer': pad.layer,
                    'shape': pad.shape,
                    'angle': pad.angle
                } for pad in component.pads]
            
            result.append(comp_data)
        
        return result


class GetNetsQueryHandler(QueryHandler):
    """Handler for GetNetsQuery."""
    
    def __init__(self, board_repository):
        self.board_repository = board_repository
    
    def handle(self, query: GetNetsQuery) -> List[Dict[str, Any]]:
        """Handle nets query."""
        if query.board_id:
            board = self.board_repository.get_board(query.board_id)
        else:
            board = self.board_repository.get_current_board()
        
        if not board:
            return []
        
        nets = board.nets
        if query.routable_only:
            nets = board.get_routable_nets()
        
        if query.net_ids:
            nets = [n for n in nets if n.id in query.net_ids]
        
        result = []
        for net in nets:
            bounds = net.get_bounds()
            result.append({
                'id': net.id,
                'name': net.name,
                'netclass': net.netclass,
                'pad_count': len(net.pads),
                'is_routable': net.is_routable,
                'bounds': {
                    'min_x': bounds.min_x,
                    'min_y': bounds.min_y,
                    'max_x': bounds.max_x,
                    'max_y': bounds.max_y
                },
                'min_distance': net.calculate_min_distance(),
                'pads': [{
                    'id': pad.id,
                    'component_id': pad.component_id,
                    'position': {
                        'x': pad.position.x,
                        'y': pad.position.y
                    }
                } for pad in net.pads]
            })
        
        return result