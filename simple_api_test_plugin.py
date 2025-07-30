"""
Simple KiCad API Test Plugin
Tests the basic API functionality step by step
"""

import pcbnew
import wx

class KiCadAPITestPlugin(pcbnew.ActionPlugin):
    """Simple test plugin to investigate KiCad API"""
    
    def defaults(self):
        self.name = "KiCad API Test"
        self.category = "Debug"
        self.description = "Test KiCad API methods systematically"
        self.show_toolbar_button = True
        
    def Run(self):
        """Run the API test"""
        print("\n" + "="*60)
        print("🧪 KiCad API Test Starting...")
        print("="*60)
        
        try:
            # Get board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board loaded!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            print("✅ Board loaded successfully")
            
            # Test 1: Board basics
            print("\n📊 Test 1: Board Basics")
            try:
                bounds = board.GetBoardEdgesBoundingBox()
                width = bounds.GetWidth()
                height = bounds.GetHeight()
                layers = board.GetCopperLayerCount()
                print(f"✅ Board size: {width/1e6:.2f} x {height/1e6:.2f} mm")
                print(f"✅ Copper layers: {layers}")
            except Exception as e:
                print(f"❌ Board basics failed: {e}")
            
            # Test 2: Footprints
            print("\n👟 Test 2: Footprints")
            try:
                footprints = list(board.GetFootprints())
                print(f"✅ Found {len(footprints)} footprints")
                
                if footprints:
                    fp = footprints[0]
                    ref = fp.GetReference()
                    pads = list(fp.Pads())
                    print(f"✅ First footprint: {ref} with {len(pads)} pads")
                    
                    if pads:
                        pad = pads[0]
                        pad_name = pad.GetName()
                        pad_net = pad.GetNet()
                        net_name = pad_net.GetNetname() if pad_net else "No net"
                        print(f"✅ First pad: {pad_name} -> '{net_name}'")
                        
            except Exception as e:
                print(f"❌ Footprints test failed: {e}")
            
            # Test 3: Nets
            print("\n🌐 Test 3: Nets")
            try:
                netcodes = board.GetNetsByNetcode()
                print(f"✅ Found {len(netcodes)} nets total")
                
                routeable_nets = 0
                for netcode, net in netcodes.items():
                    if netcode == 0:
                        continue
                    
                    net_name = net.GetNetname()
                    if not net_name:
                        continue
                    
                    # Count pads for this net
                    pad_count = 0
                    for footprint in board.GetFootprints():
                        for pad in footprint.Pads():
                            if pad.GetNet() == net:
                                pad_count += 1
                    
                    if pad_count >= 2:
                        routeable_nets += 1
                        if routeable_nets <= 3:  # Show first 3
                            print(f"✅ Routeable net: '{net_name}' ({pad_count} pads)")
                
                print(f"✅ Found {routeable_nets} routeable nets")
                
            except Exception as e:
                print(f"❌ Nets test failed: {e}")
            
            # Show results
            if routeable_nets > 0:
                message = f"SUCCESS!\n\nFound {routeable_nets} routeable nets.\nCheck console for details."
                wx.MessageBox(message, "API Test Results", wx.OK | wx.ICON_INFORMATION)
            else:
                message = "No routeable nets found.\nCheck console for details.\n\nThis board might not have unrouted nets."
                wx.MessageBox(message, "API Test Results", wx.OK | wx.ICON_WARNING)
                
        except Exception as e:
            error_msg = f"API Test failed: {str(e)}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Test Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
KiCadAPITestPlugin().register()
