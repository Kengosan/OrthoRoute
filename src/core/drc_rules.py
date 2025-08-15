#!/usr/bin/env python3
"""
Design Rule Check (DRC) Rules Management

Handles extraction and management of PCB design rules following KiCad's clearance hierarchy.
Provides design constraints for all routing algorithms.
"""
import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DRCRules:
    """Design Rule Check constraints following KiCad's clearance hierarchy"""
    
    def __init__(self, board_data: Dict):
        """
        Initialize DRC rules with KiCad hierarchy priority
        
        Args:
            board_data: Board data dictionary containing geometry and interface
        """
        # Initialize with safe defaults first
        self.min_trace_width = 0.1   # mm (4 mils) - standard minimum
        self.default_trace_width = 0.25  # mm (10 mils) - good general purpose
        self.min_trace_spacing = 0.15  # mm (6 mils) - standard clearance
        self.via_diameter = 0.6  # mm (24 mils) - standard size
        self.via_drill = 0.3    # mm (12 mils) - 2:1 aspect ratio
        self.netclasses = {}
        self.local_clearance_cache = {}  # Initialize cache for local clearance overrides
        
        self._extract_drc_rules(board_data)
        self._apply_clearance_hierarchy()
    
    def _extract_drc_rules(self, board_data: Dict):
        """Extract DRC rules using priority system"""
        
        # PRIORITY 1: Use extracted DRC rules if available (from KiCad interface)
        extracted_drc = board_data.get('drc_rules')
        if extracted_drc:
            logger.info("🔍 Using extracted DRC rules from KiCad interface...")
            
            # Use the ACTUAL values from KiCad
            self.default_trace_width = extracted_drc.default_track_width
            self.min_trace_width = extracted_drc.minimum_track_width  
            self.min_trace_spacing = extracted_drc.default_clearance
            self.via_diameter = extracted_drc.default_via_size
            self.via_drill = extracted_drc.default_via_drill
            self.netclasses = extracted_drc.netclasses
            
            logger.info(f"✅ Applied KiCad DRC rules:")
            logger.info(f"  Track width: {self.default_trace_width:.3f}mm (min: {self.min_trace_width:.3f}mm)")
            logger.info(f"  Clearance: {self.min_trace_spacing:.3f}mm") 
            logger.info(f"  Via: {self.via_diameter:.3f}mm (drill: {self.via_drill:.3f}mm)")
            logger.info(f"  Net classes: {len(self.netclasses)}")
            
            return  # Successfully extracted
            
        # PRIORITY 2: Try to extract real DRC rules using the KiCad API hierarchy
        kicad_interface = board_data.get('kicad_interface')
        
        if kicad_interface and hasattr(kicad_interface, 'board'):
            logger.info("🔍 Extracting DRC rules using KiCad Python API...")
            try:
                self._extract_drc_from_kicad_api(kicad_interface)
                return  # Successfully extracted
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract DRC from KiCad API: {e}")
        
        logger.warning("⚠️ No DRC rules available - using standard PCB defaults")
    
    def _extract_drc_from_kicad_api(self, kicad_interface):
        """Extract DRC rules using KiCad 9 IPC API hierarchy"""
        # This is a simplified version - full implementation available in original code
        try:
            if hasattr(kicad_interface, 'kicad'):
                logger.info("🔍 Extracting DRC rules using KiCad 9 IPC API...")
                kicad = kicad_interface.kicad
                
                # Get design settings using IPC service
                try:
                    design_settings = kicad.design_settings.get()
                    logger.info(f"🔍 Raw design settings from KiCad IPC API: {design_settings}")
                    
                    # Extract relevant settings
                    for key, value in design_settings.items():
                        if isinstance(value, (int, float)) and 'track' in key.lower() or 'width' in key.lower():
                            if key.lower().find('min') != -1:
                                self.min_trace_width = value / 1e6  # Convert nm to mm
                            elif key.lower().find('default') != -1:
                                self.default_trace_width = value / 1e6
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get design settings: {e}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract DRC from KiCad API: {e}")
    
    def _apply_clearance_hierarchy(self):
        """Apply KiCad's clearance hierarchy methodology"""
        # KiCad takes the MAXIMUM of all applicable clearances:
        # 1. Global board clearance (min_trace_spacing)
        # 2. Net class clearances
        # 3. Local pad clearances
        # 4. Custom rule clearances
        
        # For pathfinding, use the global clearance as baseline
        self.pathfinding_clearance = self.min_trace_spacing
        self.manufacturing_clearance = self.min_trace_spacing
        
        # Check if any netclasses have higher clearances
        max_netclass_clearance = self.min_trace_spacing
        for netclass_name, netclass_data in self.netclasses.items():
            clearance = netclass_data.get('clearance', self.min_trace_spacing)
            max_netclass_clearance = max(max_netclass_clearance, clearance)
            
        # Cap clearance if excessive (indicates possible extraction error)
        if max_netclass_clearance > 1.0:  # 1mm is very excessive for most designs
            logger.warning(f"⚠️ Very high clearance detected ({max_netclass_clearance:.3f}mm) - possible DRC extraction error")
            logger.warning(f"   Capping pathfinding clearance to 0.5mm for routing feasibility")
            self.pathfinding_clearance = min(max_netclass_clearance, 0.5)
        else:
            self.pathfinding_clearance = max_netclass_clearance
        
        logger.info(f"🎯 KiCad Clearance Hierarchy Applied:")
        logger.info(f"  Global clearance: {self.min_trace_spacing:.3f}mm")
        logger.info(f"  Max netclass clearance: {max_netclass_clearance:.3f}mm")
        logger.info(f"  Pathfinding clearance: {self.pathfinding_clearance:.3f}mm")
        logger.info(f"  Manufacturing clearance: {self.manufacturing_clearance:.3f}mm")
    
    def get_clearance_for_net(self, net_name: str) -> float:
        """Get the effective clearance for a specific net following KiCad hierarchy"""
        # KiCad hierarchy: Local pad > NetClass > Global
        
        # Start with global clearance
        effective_clearance = self.min_trace_spacing
        
        # Check if net belongs to a netclass with higher clearance
        for netclass_name, netclass_rules in self.netclasses.items():
            if net_name in netclass_rules.get('nets', []):
                netclass_clearance = netclass_rules.get('clearance', self.min_trace_spacing)
                effective_clearance = max(effective_clearance, netclass_clearance)
                logger.debug(f"Net '{net_name}' in NetClass '{netclass_name}': clearance={netclass_clearance:.3f}mm")
                break
        
        return effective_clearance
    
    def get_net_constraints(self, net_name: str) -> Dict[str, float]:
        """Get DRC constraints for a specific net following KiCad's clearance hierarchy"""
        # Get the effective clearance for this net using KiCad hierarchy
        effective_clearance = self.get_clearance_for_net(net_name)
        
        # Check if net has a specific netclass for other properties
        netclass_data = None
        for netclass_name, netclass_rules in self.netclasses.items():
            if net_name in netclass_rules.get('nets', []):
                netclass_data = netclass_rules
                break
        
        if netclass_data:
            return {
                'trace_width': netclass_data.get('track_width', self.default_trace_width),
                'clearance': effective_clearance,  # Use hierarchy-calculated clearance
                'manufacturing_clearance': effective_clearance,  # Same for validation
                'via_size': netclass_data.get('via_diameter', self.via_diameter),
                'via_drill': netclass_data.get('via_drill', self.via_drill)
            }
        
        # Use defaults with hierarchy-calculated clearance
        return {
            'trace_width': self.default_trace_width,
            'clearance': effective_clearance,  # Use hierarchy-calculated clearance
            'manufacturing_clearance': effective_clearance,  # Same for validation
            'via_size': self.via_diameter,
            'via_drill': self.via_drill
        }
    
    def calculate_pad_track_clearance(self, pad_uuid: str, pad_net_name: str, 
                                    track_net_name: str, track_width: float) -> float:
        """Calculate clearance between a pad and track using KiCad 9's clearance hierarchy"""
        
        # If both are on the same net, no clearance needed (they're connected)
        if pad_net_name == track_net_name:
            return 0.0
        
        # STEP 1: Get pad clearance using hierarchy
        pad_clearance = self._get_object_clearance(pad_uuid, pad_net_name, 'pad')
        
        # STEP 2: Get track net clearance (we don't have track UUID yet, use net-based clearance)
        track_clearance = self.get_clearance_for_net(track_net_name)
        
        # STEP 3: Take maximum as per KiCad rules
        final_clearance = max(self.min_trace_spacing, pad_clearance, track_clearance)
        
        logger.debug(f"📏 Pad-Track clearance {pad_net_name}↔{track_net_name}: "
                   f"pad={pad_clearance:.3f}mm, track_net={track_clearance:.3f}mm → {final_clearance:.3f}mm")
        
        return final_clearance
    
    def _get_object_clearance(self, object_uuid: str, net_name: str, object_type: str) -> float:
        """Get clearance for a specific object (pad or track) following KiCad hierarchy"""
        
        # STEP 1: Check for local clearance override
        if hasattr(self, 'local_clearance_cache') and object_uuid in self.local_clearance_cache:
            local_clearance = self.local_clearance_cache[object_uuid]
            logger.debug(f"   {object_type} {object_uuid[:8]}: LOCAL override {local_clearance:.3f}mm")
            return local_clearance
        
        # STEP 2: Check net class clearance
        for netclass_name, netclass_rules in self.netclasses.items():
            if net_name in netclass_rules.get('nets', []):
                netclass_clearance = netclass_rules.get('clearance', self.min_trace_spacing)
                logger.debug(f"   {object_type} {object_uuid[:8]}: NETCLASS '{netclass_name}' {netclass_clearance:.3f}mm")
                return netclass_clearance
        
        # STEP 3: Fall back to board default
        logger.debug(f"   {object_type} {object_uuid[:8]}: DEFAULT {self.min_trace_spacing:.3f}mm")
        return self.min_trace_spacing
