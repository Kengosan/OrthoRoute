"""
RRG Fabric Caching - Speed up startup by caching pre-built fabrics
"""

import pickle
import hashlib
import os
import logging
from typing import Optional, Tuple, List, Dict
from .rrg import RoutingResourceGraph
from .types import Pad

logger = logging.getLogger(__name__)

class RRGCache:
    """Cache for pre-built RRG fabrics"""
    
    def __init__(self, cache_dir: str = "rrg_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, bounds: Tuple[float, float, float, float], 
                          airwires: List[Dict], config_dict: Dict) -> str:
        """Generate cache key from fabric parameters"""
        # Create hash from key parameters
        key_data = {
            'bounds': bounds,
            'airwire_count': len(airwires),
            'airwire_bounds': self._get_airwire_bounds_hash(airwires),
            'config': config_dict
        }
        
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_airwire_bounds_hash(self, airwires: List[Dict]) -> str:
        """Get hash of airwire bounding box"""
        if not airwires:
            return "empty"
        
        all_x = []
        all_y = []
        
        for airwire in airwires:
            try:
                all_x.extend([airwire['start_x'], airwire['end_x']])
                all_y.extend([airwire['start_y'], airwire['end_y']])
            except KeyError:
                continue
        
        if all_x and all_y:
            bounds = (min(all_x), min(all_y), max(all_x), max(all_y))
            return str(bounds)
        return "invalid"
    
    def get_cached_fabric(self, bounds: Tuple[float, float, float, float],
                         airwires: List[Dict], config_dict: Dict) -> Optional[RoutingResourceGraph]:
        """Try to load cached fabric"""
        try:
            cache_key = self._generate_cache_key(bounds, airwires, config_dict)
            cache_file = os.path.join(self.cache_dir, f"rrg_{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                logger.info(f"INFO: Loading cached RRG fabric from {cache_file}")
                with open(cache_file, 'rb') as f:
                    rrg = pickle.load(f)
                logger.info(f"SUCCESS: Loaded cached fabric: {len(rrg.nodes)} nodes, {len(rrg.edges)} edges")
                return rrg
            else:
                logger.debug(f"No cached fabric found for key {cache_key}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cached fabric: {e}")
            return None
    
    def cache_fabric(self, rrg: RoutingResourceGraph, bounds: Tuple[float, float, float, float],
                    airwires: List[Dict], config_dict: Dict) -> None:
        """Cache fabric for future use"""
        try:
            cache_key = self._generate_cache_key(bounds, airwires, config_dict)
            cache_file = os.path.join(self.cache_dir, f"rrg_{cache_key}.pkl")
            
            logger.info(f"INFO: Caching RRG fabric to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(rrg, f)
            
            file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            logger.info(f"SUCCESS: Cached fabric: {file_size_mb:.1f}MB")
            
        except Exception as e:
            logger.warning(f"Failed to cache fabric: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached fabrics"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("rrg_") and filename.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("SUCCESS: Cleared RRG fabric cache")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")