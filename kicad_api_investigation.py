"""
KiCad PCBNew API Investigation Script
This script systematically tests KiCad's API to understand how to properly extract nets and board data.
"""

import pcbnew
import sys

def investigate_kicad_api():
    """Comprehensive investigation of KiCad's PCBNew API"""
    
    print("🔍 KiCad PCBNew API Investigation")
    print("=" * 50)
    
    # Get the current board
    try:
        board = pcbnew.GetBoard()
        if not board:
            print("❌ No board loaded!")
            return
        print("✅ Board loaded successfully")
    except Exception as e:
        print(f"❌ Error getting board: {e}")
        return
    
    print("\n📊 BOARD INFORMATION:")
    print("-" * 30)
    
    # Basic board info
    try:
        bounds = board.GetBoardEdgesBoundingBox()
        print(f"Board bounds: {bounds.GetWidth()/1e6:.2f} x {bounds.GetHeight()/1e6:.2f} mm")
        print(f"Board position: ({bounds.GetX()/1e6:.2f}, {bounds.GetY()/1e6:.2f}) mm")
    except Exception as e:
        print(f"❌ Error getting bounds: {e}")
    
    try:
        layer_count = board.GetCopperLayerCount()
        print(f"Copper layers: {layer_count}")
    except Exception as e:
        print(f"❌ Error getting layer count: {e}")
    
    print("\n🔍 FOOTPRINTS ANALYSIS:")
    print("-" * 30)
    
    # Investigate footprints
    try:
        footprints = board.GetFootprints()
        footprint_list = list(footprints)
        print(f"Total footprints: {len(footprint_list)}")
        
        for i, footprint in enumerate(footprint_list[:5]):  # Show first 5
            try:
                ref = footprint.GetReference()
                value = footprint.GetValue()
                pos = footprint.GetPosition()
                pads = list(footprint.Pads())
                print(f"  Footprint {i+1}: {ref} ({value}) at ({pos.x/1e6:.2f}, {pos.y/1e6:.2f})mm, {len(pads)} pads")
                
                # Show first 3 pads
                for j, pad in enumerate(pads[:3]):
                    try:
                        pad_name = pad.GetName()
                        pad_pos = pad.GetPosition()
                        pad_net = pad.GetNet()
                        pad_net_name = pad_net.GetNetname() if pad_net else "No net"
                        pad_net_code = pad_net.GetNetCode() if pad_net else 0
                        print(f"    Pad {j+1}: {pad_name} at ({pad_pos.x/1e6:.2f}, {pad_pos.y/1e6:.2f})mm -> Net {pad_net_code}: '{pad_net_name}'")
                    except Exception as e:
                        print(f"    ❌ Error reading pad {j+1}: {e}")
                        
            except Exception as e:
                print(f"  ❌ Error reading footprint {i+1}: {e}")
                
    except Exception as e:
        print(f"❌ Error getting footprints: {e}")
    
    print("\n🌐 NETS ANALYSIS:")
    print("-" * 30)
    
    # Investigate nets using different methods
    print("Method 1: GetNetsByNetcode()")
    try:
        netcodes = board.GetNetsByNetcode()
        print(f"Total nets (by netcode): {len(netcodes)}")
        
        for netcode, net in list(netcodes.items())[:10]:  # Show first 10
            try:
                net_name = net.GetNetname()
                print(f"  Net {netcode}: '{net_name}'")
            except Exception as e:
                print(f"  ❌ Error reading net {netcode}: {e}")
                
    except Exception as e:
        print(f"❌ Error with GetNetsByNetcode(): {e}")
    
    print("\nMethod 2: GetNetInfo()")
    try:
        netinfo = board.GetNetInfo()
        print(f"NetInfo object: {type(netinfo)}")
        
        # Try to iterate through nets
        try:
            net_count = netinfo.GetNetCount()
            print(f"Net count: {net_count}")
            
            for i in range(min(10, net_count)):  # First 10 nets
                try:
                    net = netinfo.GetNetItem(i)
                    if net:
                        net_name = net.GetNetname()
                        net_code = net.GetNetCode()
                        print(f"  Net {i}: Code {net_code}, Name '{net_name}'")
                except Exception as e:
                    print(f"  ❌ Error reading net {i}: {e}")
                    
        except Exception as e:
            print(f"❌ Error iterating nets: {e}")
            
    except Exception as e:
        print(f"❌ Error with GetNetInfo(): {e}")
    
    print("\n🔗 NET-PAD RELATIONSHIPS:")
    print("-" * 30)
    
    # Cross-reference nets and pads
    try:
        netcodes = board.GetNetsByNetcode()
        
        for netcode, net in list(netcodes.items())[:5]:  # First 5 nets
            if netcode == 0:
                continue
                
            net_name = net.GetNetname()
            print(f"Net {netcode}: '{net_name}'")
            
            # Find pads for this net
            pad_count = 0
            footprints = list(board.GetFootprints())
            
            for footprint in footprints:
                for pad in footprint.Pads():
                    try:
                        if pad.GetNet() == net:
                            pad_count += 1
                            if pad_count <= 3:  # Show first 3 pads
                                pad_name = pad.GetName()
                                pad_pos = pad.GetPosition()
                                footprint_ref = footprint.GetReference()
                                print(f"  Pad: {footprint_ref}.{pad_name} at ({pad_pos.x/1e6:.2f}, {pad_pos.y/1e6:.2f})mm")
                    except Exception as e:
                        print(f"  ❌ Error checking pad: {e}")
            
            print(f"  Total pads for this net: {pad_count}")
            
    except Exception as e:
        print(f"❌ Error analyzing net-pad relationships: {e}")
    
    print("\n🧪 API METHODS TESTING:")
    print("-" * 30)
    
    # Test various API methods
    test_methods = [
        ('board.GetNetCount()', lambda: board.GetNetCount()),
        ('board.GetTrackCount()', lambda: board.GetTrackCount()),
        ('board.GetAreaCount()', lambda: board.GetAreaCount()),
        ('len(list(board.GetTracks()))', lambda: len(list(board.GetTracks()))),
        ('board.GetFileName()', lambda: board.GetFileName()),
    ]
    
    for method_name, method_func in test_methods:
        try:
            result = method_func()
            print(f"✅ {method_name}: {result}")
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
    print("\n🏁 Investigation Complete!")
    print("=" * 50)

def save_board_analysis():
    """Save detailed board analysis to file"""
    try:
        import os
        from datetime import datetime
        
        output_file = "kicad_api_analysis.txt"
        
        # Redirect stdout to capture all output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            investigate_kicad_api()
        
        output = f.getvalue()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"KiCad API Analysis - {timestamp}\n{'='*60}\n\n"
        
        with open(output_file, 'w') as file:
            file.write(header + output)
        
        print(f"📄 Analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving analysis: {e}")

if __name__ == "__main__":
    # This can be run as a standalone script or imported
    investigate_kicad_api()
    save_board_analysis()
