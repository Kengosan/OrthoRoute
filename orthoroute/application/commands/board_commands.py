"""Command handlers for board operations."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .routing_commands import Command, CommandHandler
from ...domain.models.board import Board, Component
from ...domain.events.board_events import (
    BoardLoaded, ComponentsChanged, NetsChanged, BoardValidated
)

logger = logging.getLogger(__name__)


@dataclass
class LoadBoardCommand(Command):
    """Command to load board data."""
    file_path: Optional[str] = None
    board_data: Optional[Dict[str, Any]] = None
    source: str = "file"  # 'file', 'kicad_api', 'memory'


@dataclass
class UpdateComponentsCommand(Command):
    """Command to update board components."""
    board_id: str
    added_components: List[Component]
    removed_component_ids: List[str]
    modified_components: List[Component]


@dataclass
class ValidateBoardCommand(Command):
    """Command to validate board integrity."""
    board_id: str
    check_drc: bool = True
    check_connectivity: bool = True


class LoadBoardCommandHandler(CommandHandler):
    """Handler for LoadBoardCommand."""
    
    def __init__(self, board_repository, board_parser, event_publisher):
        self.board_repository = board_repository
        self.board_parser = board_parser
        self.event_publisher = event_publisher
    
    def handle(self, command: LoadBoardCommand) -> Board:
        """Handle board loading command."""
        logger.info(f"Loading board from {command.source}")
        
        try:
            if command.source == "file" and command.file_path:
                # Parse board from file
                board_data = self.board_parser.parse_file(command.file_path)
                board = self.board_parser.create_board_from_data(board_data)
            elif command.source == "memory" and command.board_data:
                # Create board from provided data
                board = self.board_parser.create_board_from_data(command.board_data)
            elif command.source == "kicad_api":
                # Load from KiCad API
                board_data = self.board_parser.load_from_kicad()
                board = self.board_parser.create_board_from_data(board_data)
            else:
                raise ValueError(f"Invalid source or missing data for command: {command}")
            
            # Store board in repository
            self.board_repository.save_board(board)
            self.board_repository.set_current_board(board.id)
            
            # Publish board loaded event
            event = BoardLoaded(
                event_id=f"board_loaded_{board.id}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                board_id=board.id,
                board_name=board.name,
                component_count=len(board.components),
                net_count=len(board.nets),
                layer_count=len(board.layers)
            )
            self.event_publisher.publish(event)
            
            logger.info(f"Successfully loaded board {board.name} with {len(board.components)} components")
            return board
            
        except Exception as e:
            logger.error(f"Error handling LoadBoardCommand: {e}")
            raise


class UpdateComponentsCommandHandler(CommandHandler):
    """Handler for UpdateComponentsCommand."""
    
    def __init__(self, board_repository, event_publisher):
        self.board_repository = board_repository
        self.event_publisher = event_publisher
    
    def handle(self, command: UpdateComponentsCommand) -> bool:
        """Handle components update command."""
        logger.info(f"Updating components for board {command.board_id}")
        
        try:
            board = self.board_repository.get_board(command.board_id)
            if not board:
                raise ValueError(f"Board {command.board_id} not found")
            
            # Track changes
            added_ids = []
            removed_ids = command.removed_component_ids.copy()
            modified_ids = []
            
            # Add new components
            for component in command.added_components:
                board.add_component(component)
                added_ids.append(component.id)
            
            # Remove components
            board.components = [c for c in board.components if c.id not in command.removed_component_ids]
            
            # Update existing components
            for modified_component in command.modified_components:
                for i, existing in enumerate(board.components):
                    if existing.id == modified_component.id:
                        board.components[i] = modified_component
                        modified_ids.append(modified_component.id)
                        break
            
            # Rebuild indexes
            board._build_indexes()
            
            # Save updated board
            self.board_repository.save_board(board)
            
            # Publish components changed event
            event = ComponentsChanged(
                event_id=f"components_changed_{command.board_id}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                board_id=command.board_id,
                added_components=added_ids,
                removed_components=removed_ids,
                modified_components=modified_ids
            )
            self.event_publisher.publish(event)
            
            logger.info(f"Updated components: {len(added_ids)} added, {len(removed_ids)} removed, {len(modified_ids)} modified")
            return True
            
        except Exception as e:
            logger.error(f"Error handling UpdateComponentsCommand: {e}")
            return False


class ValidateBoardCommandHandler(CommandHandler):
    """Handler for ValidateBoardCommand."""
    
    def __init__(self, board_repository, drc_checker, event_publisher):
        self.board_repository = board_repository
        self.drc_checker = drc_checker
        self.event_publisher = event_publisher
    
    def handle(self, command: ValidateBoardCommand) -> Dict[str, Any]:
        """Handle board validation command."""
        logger.info(f"Validating board {command.board_id}")
        
        try:
            board = self.board_repository.get_board(command.board_id)
            if not board:
                raise ValueError(f"Board {command.board_id} not found")
            
            issues = []
            
            # Check board integrity
            integrity_issues = board.validate_integrity()
            issues.extend(integrity_issues)
            
            # Check DRC if requested
            drc_violations = []
            if command.check_drc:
                drc_violations = self.drc_checker.check_board(board)
                issues.extend([str(v) for v in drc_violations])
            
            # Check connectivity if requested
            connectivity_issues = []
            if command.check_connectivity:
                # This would involve more complex connectivity analysis
                pass
            
            is_valid = len(issues) == 0
            
            # Publish validation event
            event = BoardValidated(
                event_id=f"board_validated_{command.board_id}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                board_id=command.board_id,
                is_valid=is_valid,
                issues=issues
            )
            self.event_publisher.publish(event)
            
            validation_result = {
                'is_valid': is_valid,
                'issues': issues,
                'integrity_issues': integrity_issues,
                'drc_violations': drc_violations,
                'connectivity_issues': connectivity_issues
            }
            
            logger.info(f"Board validation completed: {'PASSED' if is_valid else 'FAILED'} ({len(issues)} issues)")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error handling ValidateBoardCommand: {e}")
            raise