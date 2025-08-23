"""Abstract board repository interface."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from ...domain.models.board import Board


class BoardRepository(ABC):
    """Abstract repository for board data access."""
    
    @abstractmethod
    def get_board(self, board_id: str) -> Optional[Board]:
        """Get board by ID."""
        pass
    
    @abstractmethod
    def get_current_board(self) -> Optional[Board]:
        """Get the currently active board."""
        pass
    
    @abstractmethod
    def set_current_board(self, board_id: str) -> None:
        """Set the currently active board."""
        pass
    
    @abstractmethod
    def save_board(self, board: Board) -> None:
        """Save board data."""
        pass
    
    @abstractmethod
    def delete_board(self, board_id: str) -> bool:
        """Delete board by ID."""
        pass
    
    @abstractmethod
    def list_boards(self) -> List[Dict[str, Any]]:
        """List all available boards with metadata."""
        pass
    
    @abstractmethod
    def board_exists(self, board_id: str) -> bool:
        """Check if board exists."""
        pass