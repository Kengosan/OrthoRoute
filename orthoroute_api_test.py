"""
OrthoRoute API Test Plugin
Copy this file to KiCad's plugin directory to test basic functionality
"""

import pcbnew
import wx

class OrthoRouteAPITest(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "OrthoRoute API Test"
        self.category = "Test"
        self.description = "Test OrthoRoute board analysis functionality"
        self.show_toolbar_button = False
    
    def Run(self):
        try:
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found!", "Error", wx.OK)
                return
            
            # Test board analysis
            result = self.analyze_board(board)
            
            # Show results
            message = f"""OrthoRoute API Test Results:

Board: {result['width']:.1f} x {result['height']:.1f} mm
Layers: {result['layers']}
Footprints: {result['footprints']}
Total Pads: {result['total_pads']}
Total Nets: {result['total_nets']}

Net Analysis:
- Signal nets to route: {result['routeable_nets']}
- Power/ground nets: {result['power_nets']}
- Nets with copper fills: {result['filled_nets']}

{result['status']}"""
            
            wx.MessageBox(message, "OrthoRoute API Test", wx.OK)
            
        except Exception as e:
            wx.MessageBox(f"Test failed: {str(e)}", "Error", wx.OK)
    
    def analyze_board(self, board):
        # Board dimensions
        bbox = board.GetBoardEdgesBoundingBox()
        width_mm = bbox.GetWidth() / 1e6
        height_mm = bbox.GetHeight() / 1e6
        layers = board.GetCopperLayerCount()
        
        # Footprints and pads
        footprints = list(board.GetFootprints())
        total_pads = sum(len(list(fp.Pads())) for fp in footprints)
        
        # Nets
        net_count = board.GetNetCount()
        routeable_nets = 0
        power_nets = 0
        filled_nets = 0
        
        for netcode in range(1, net_count):
            net_info = board.GetNetInfo().GetNetItem(netcode)
            if not net_info:
                continue
            
            net_name = net_info.GetNetname()
            
            # Check if it's a power/ground net
            power_net_names = ['GND', 'VCC', 'VDD', '+5V', '+3V3', '+12V', '-12V', 'VEE', 'VSS', 'AGND', 'DGND']
            is_power_net = any(power_name in net_name.upper() for power_name in power_net_names)
            
            # Check for copper fills
            has_fill = self._check_copper_fill(board, netcode)
            
            # Count pads for this net
            pad_count = 0
            for fp in footprints:
                for pad in fp.Pads():
                    pad_net = pad.GetNet()
                    if pad_net and pad_net.GetNetCode() == netcode:
                        pad_count += 1
            
            if pad_count >= 2:
                if is_power_net:
                    power_nets += 1
                elif has_fill:
                    filled_nets += 1
                else:
                    routeable_nets += 1
        
        status = f"Signal nets: {routeable_nets} | Power nets: {power_nets} | Filled nets: {filled_nets}"
        if routeable_nets > 0:
            status += " | READY TO ROUTE"
        else:
            status += " | NO ROUTING NEEDED"
        
        return {
            'width': width_mm,
            'height': height_mm, 
            'layers': layers,
            'footprints': len(footprints),
            'total_pads': total_pads,
            'total_nets': net_count,
            'routeable_nets': routeable_nets,
            'power_nets': power_nets,
            'filled_nets': filled_nets,
            'status': status
        }
    
    def _check_copper_fill(self, board, netcode):
        """Check if net has copper fills"""
        try:
            for i in range(board.GetAreaCount()):
                zone = board.GetArea(i)
                if zone:
                    zone_net = zone.GetNet()
                    if zone_net and zone_net.GetNetCode() == netcode:
                        return True
            return False
        except:
            return False

# Register plugin
OrthoRouteAPITest().register()
