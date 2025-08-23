"""In-memory board repository implementation."""
import logging
from typing import Optional, List, Dict, Any

from ...application.interfaces.board_repository import BoardRepository
from ...domain.models.board import Board

logger = logging.getLogger(__name__)


class MemoryBoardRepository(BoardRepository):
    """In-memory implementation of board repository."""
    
    def __init__(self):
        """Initialize memory board repository."""
        self._boards: Dict[str, Board] = {}
        self._current_board_id: Optional[str] = None
    
    def get_board(self, board_id: str) -> Optional[Board]:
        """Get board by ID."""
        return self._boards.get(board_id)
    
    def get_current_board(self) -> Optional[Board]:
        """Get the currently active board."""
        if self._current_board_id:
            return self._boards.get(self._current_board_id)
        return None
    
    def set_current_board(self, board_id: str) -> None:
        """Set the currently active board."""
        if board_id in self._boards:
            self._current_board_id = board_id
            logger.info(f"Set current board to {board_id}")
        else:
            raise ValueError(f"Board {board_id} not found")
    
    def save_board(self, board: Board) -> None:
        """Save board data."""
        self._boards[board.id] = board
        
        # Set as current board if no current board
        if not self._current_board_id:
            self._current_board_id = board.id
        
        logger.debug(f"Saved board {board.id}: {board.name}")
    
    def delete_board(self, board_id: str) -> bool:
        """Delete board by ID."""
        if board_id in self._boards:
            del self._boards[board_id]
            
            # Clear current board if it was deleted
            if self._current_board_id == board_id:
                self._current_board_id = None
            
            logger.info(f"Deleted board {board_id}")
            return True
        
        return False
    
    def list_boards(self) -> List[Dict[str, Any]]:
        """List all available boards with metadata."""
        boards_list = []
        
        for board_id, board in self._boards.items():
            bounds = board.get_bounds()
            boards_list.append({
                'id': board_id,
                'name': board.name,
                'component_count': len(board.components),
                'net_count': len(board.nets),
                'layer_count': len(board.layers),
                'routable_net_count': len(board.get_routable_nets()),
                'bounds': {
                    'width': bounds.width,
                    'height': bounds.height
                },
                'is_current': board_id == self._current_board_id
            })
        
        return boards_list
    
    def board_exists(self, board_id: str) -> bool:
        """Check if board exists."""
        return board_id in self._boards
    
    def clear_all_boards(self) -> int:
        """Clear all boards. Returns count cleared."""
        count = len(self._boards)
        self._boards.clear()
        self._current_board_id = None
        logger.info(f"Cleared {count} boards")
        return count