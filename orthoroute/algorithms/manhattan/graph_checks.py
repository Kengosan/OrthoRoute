"""Shared graph validation and preflight checks for all routing engines."""
import logging

logger = logging.getLogger(__name__)

def preflight_graph(graph_state):
    """Validate graph integrity before routing begins.
    
    Args:
        graph_state: Graph state object with CSR matrices and node counts
        
    Returns:
        bool: True if all checks pass, False otherwise
    """
    try:
        logger.info("[CHK] Starting preflight validation...")
        
        # Check basic structure exists
        if not hasattr(graph_state, 'indptr_g') or not hasattr(graph_state, 'indices_g'):
            logger.error("[CHK] PREFLIGHT FAILED: Missing CSR arrays")
            return False
            
        if not hasattr(graph_state, 'lattice_node_count'):
            logger.error("[CHK] PREFLIGHT FAILED: Missing lattice_node_count")
            return False
        
        # Validate CSR structure
        indptr_len = len(graph_state.indptr_g)
        expected_indptr_len = graph_state.lattice_node_count + 1
        
        if indptr_len != expected_indptr_len:
            logger.error(f"[CHK] PREFLIGHT FAILED: indptr length {indptr_len} != expected {expected_indptr_len}")
            return False
        
        # Check indptr consistency
        if graph_state.indptr_g[-1] != len(graph_state.indices_g):
            logger.error(f"[CHK] PREFLIGHT FAILED: indptr[-1]={graph_state.indptr_g[-1]} != indices length={len(graph_state.indices_g)}")
            return False
        
        # Check node coordinates if available
        if hasattr(graph_state, 'node_coordinates_lattice'):
            coord_count = graph_state.node_coordinates_lattice.shape[0]
            if coord_count != graph_state.lattice_node_count:
                logger.error(f"[CHK] PREFLIGHT FAILED: coordinates {coord_count} != lattice nodes {graph_state.lattice_node_count}")
                return False
        
        # All checks passed
        logger.info(f"[CHK] PREFLIGHT PASSED: {graph_state.lattice_node_count} nodes, {len(graph_state.indices_g)} edges")
        return True
        
    except Exception as e:
        logger.error(f"[CHK] PREFLIGHT FAILED: Exception during validation: {e}")
        return False


def validate_lattice_integrity(graph_state):
    """Validate that lattice structure is consistent and frozen.
    
    Args:
        graph_state: Graph state object
        
    Returns:
        bool: True if lattice is properly frozen
    """
    try:
        # Check that total node count equals lattice + pads
        if hasattr(graph_state, 'pad_node_count'):
            expected_total = graph_state.lattice_node_count + graph_state.pad_node_count
            if hasattr(graph_state, 'total_node_count'):
                if graph_state.total_node_count != expected_total:
                    logger.error(f"[CHK] LATTICE INTEGRITY FAILED: total={graph_state.total_node_count} != lattice+pads={expected_total}")
                    return False
        
        logger.info(f"[CHK] LATTICE FREEZE OK: {graph_state.lattice_node_count} lattice nodes, structure frozen")
        return True
        
    except Exception as e:
        logger.error(f"[CHK] LATTICE INTEGRITY FAILED: {e}")
        return False