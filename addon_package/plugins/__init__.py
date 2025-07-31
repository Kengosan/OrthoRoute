"""
OrthoRoute KiCad Plugin - Process Isolation Version
This plugin runs GPU routing in a completely separate process to avoid KiCad crashes
"""

import pcbnew
import wx
import os
import sys
import json
import time
import subprocess
import tempfile
import threading
from pathlib import Path

class OrthoRoutePlugin(pcbnew.ActionPlugin):
    """KiCad plugin that uses process isolation for GPU routing"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Routing"
        self.description = "High-performance orthogonal routing with GPU acceleration and crash protection"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
    
    def Run(self):
        """Main plugin execution with process isolation"""
        print("[START] OrthoRoute Isolated Plugin starting...")
        
        try:
            # Get board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board found!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config = self.show_config_dialog()
            if not config:
                return  # User cancelled
            
            # Create temporary work directory
            work_dir = Path(tempfile.mkdtemp(prefix="orthoroute_"))
            print(f"[DIR] Work directory: {work_dir}")
            
            try:
                # Extract board data
                print("📊 Extracting board data...")
                board_data = self.extract_board_data(board)
                
                # Start standalone server
                print("[START] Starting standalone GPU server...")
                server_process = self.start_server(work_dir)
                
                if not server_process:
                    wx.MessageBox("Failed to start GPU server!", "Error", wx.OK | wx.ICON_ERROR)
                    return
                
                try:
                    # Wait for server to be ready
                    if not self.wait_for_server_ready(work_dir):
                        wx.MessageBox("GPU server failed to start!", "Error", wx.OK | wx.ICON_ERROR)
                        return
                    
                    # Send routing request
                    print("📨 Sending routing request...")
                    self.send_routing_request(work_dir, board_data, config)
                    
                    # Monitor progress with dialog
                    result = self.monitor_routing_progress(work_dir)
                    
                    if result and result.get('success', False):
                        # Apply results to board
                        self.apply_routing_results(board, result)
                        
                        # Show success dialog
                        stats = result.get('statistics', {})
                        success_rate = stats.get('success_rate', 0)
                        routed_count = stats.get('routed_count', 0)
                        total_nets = stats.get('total_nets', 0)
                        
                        message = f"Routing completed successfully!\n\n"
                        message += f"Success rate: {success_rate:.1f}%\n"
                        message += f"Routed nets: {routed_count}/{total_nets}\n"
                        message += f"Process isolation: [OK] No crashes!"
                        
                        wx.MessageBox(message, "OrthoRoute Success", wx.OK | wx.ICON_INFORMATION)
                        
                        # Refresh display
                        pcbnew.Refresh()
                        
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No result received'
                        wx.MessageBox(f"Routing failed: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
                
                finally:
                    # Stop server
                    print("🛑 Stopping GPU server...")
                    self.stop_server(work_dir, server_process)
            
            finally:
                # Cleanup work directory
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                    print("🧹 Work directory cleaned up")
                except Exception as e:
                    print(f"[WARN] Cleanup error: {e}")
        
        except Exception as e:
            print(f"💥 Plugin error: {e}")
            import traceback
            traceback.print_exc()
            wx.MessageBox(f"Plugin error: {e}", "Error", wx.OK | wx.ICON_ERROR)
    
    def show_config_dialog(self):
        """Show configuration dialog"""
        try:
            dlg = wx.Dialog(None, title="OrthoRoute Configuration", size=(500, 600))
            panel = wx.Panel(dlg)
            sizer = wx.BoxSizer(wx.VERTICAL)
            
            # Grid size
            sizer.Add(wx.StaticText(panel, label="Grid Size:"), 0, wx.ALL, 5)
            grid_size_ctrl = wx.SpinCtrl(panel, value="100", min=50, max=1000)
            sizer.Add(grid_size_ctrl, 0, wx.ALL | wx.EXPAND, 5)
            
            # Process isolation info
            info_text = wx.StaticText(panel, label="[OK] GPU operations will run in isolated process\n[SAFE] KiCad crash protection enabled")
            sizer.Add(info_text, 0, wx.ALL, 10)
            
            # Buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            ok_btn = wx.Button(panel, wx.ID_OK, "[START] GPU ROUTING", size=(200, 40))
            ok_btn.SetDefault()
            try:
                ok_btn.SetBackgroundColour(wx.Colour(0, 120, 0))  # Green background
                ok_btn.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
            except:
                pass  # Color setting might not work on all systems
            
            cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel", size=(100, 40))
            btn_sizer.Add(ok_btn, 0, wx.ALL, 10)
            btn_sizer.Add(cancel_btn, 0, wx.ALL, 10)
            sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 20)
            
            panel.SetSizer(sizer)
            
            if dlg.ShowModal() == wx.ID_OK:
                config = {
                    'grid_size': grid_size_ctrl.GetValue()
                }
                dlg.Destroy()
                return config
            else:
                dlg.Destroy()
                return None
                
        except Exception as e:
            print(f"Config dialog error: {e}")
            return None
    
    def extract_board_data(self, board):
        """Extract board data for routing"""
        board_data = {
            'nets': [],
            'tracks': [],
            'board_size': {}
        }
        
        # Get board boundary
        bbox = board.GetBoardEdgesBoundingBox()
        board_data['board_size'] = {
            'width': bbox.GetWidth(),
            'height': bbox.GetHeight(),
            'x': bbox.GetX(),
            'y': bbox.GetY()
        }
        
        # Extract nets that need routing
        for net_id in range(board.GetNetCount()):
            net = board.FindNet(net_id)
            if net and net.GetNetname() and net.GetNetname() != "":
                net_data = {
                    'id': net_id,
                    'name': net.GetNetname(),
                    'pads': []
                }
                
                # Get pads for this net
                for footprint in board.GetFootprints():
                    for pad in footprint.Pads():
                        if pad.GetNet() == net:
                            pad_data = {
                                'x': pad.GetPosition().x,
                                'y': pad.GetPosition().y,
                                'layer': pad.GetLayerName()
                            }
                            net_data['pads'].append(pad_data)
                
                # Only include nets with 2+ pads
                if len(net_data['pads']) >= 2:
                    board_data['nets'].append(net_data)
        
        # Extract existing tracks
        for track in board.GetTracks():
            track_data = {
                'start_x': track.GetStart().x,
                'start_y': track.GetStart().y,
                'end_x': track.GetEnd().x,
                'end_y': track.GetEnd().y,
                'width': track.GetWidth(),
                'layer': track.GetLayerName(),
                'net_name': track.GetNet().GetNetname() if track.GetNet() else ""
            }
            board_data['tracks'].append(track_data)
        
        print(f"📊 Extracted {len(board_data['nets'])} nets, {len(board_data['tracks'])} tracks")
        return board_data
    
    def start_server(self, work_dir):
        """Start standalone GPU server with KiCad-safe subprocess approach"""
        try:
            # Find server script (it's in the same directory as this plugin)
            plugin_dir = Path(__file__).parent
            server_script = plugin_dir / "orthoroute_standalone_server.py"
            
            if not server_script.exists():
                print(f"[ERROR] Server script not found: {server_script}")
                return None
            
            print(f"[OK] Server script found: {server_script}")
            
            # Create a safer command that's less likely to be intercepted by KiCad
            python_exe = sys.executable
            if not python_exe:
                python_exe = "python"  # Fallback
            
            # Use absolute paths to avoid any confusion
            cmd = [
                str(python_exe),
                str(server_script.absolute()),
                "--work-dir", str(Path(work_dir).absolute())
            ]
            
            print(f"[START] Starting server command: {cmd}")
            print(f"[INFO] Python executable: {python_exe}")
            print(f"[INFO] Server script (absolute): {server_script.absolute()}")
            print(f"[INFO] Work directory (absolute): {Path(work_dir).absolute()}")
            
            # Use shell=False and try to isolate from KiCad's process completely
            try:
                # First try: Standard detached process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    cwd=str(Path(work_dir).absolute()),
                    env=dict(os.environ),  # Fresh environment copy
                    shell=False,  # Don't use shell
                    creationflags=subprocess.DETACHED_PROCESS if hasattr(subprocess, 'DETACHED_PROCESS') else 0
                )
                print(f"[OK] Server started with DETACHED_PROCESS, PID: {process.pid}")
                
            except Exception as e1:
                print(f"[WARN] DETACHED_PROCESS failed: {e1}")
                try:
                    # Second try: New console
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        cwd=str(Path(work_dir).absolute()),
                        env=dict(os.environ),
                        shell=False,
                        creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
                    )
                    print(f"[OK] Server started with CREATE_NEW_CONSOLE, PID: {process.pid}")
                    
                except Exception as e2:
                    print(f"[WARN] CREATE_NEW_CONSOLE failed: {e2}")
                    # Third try: Basic subprocess without special flags
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        cwd=str(Path(work_dir).absolute()),
                        env=dict(os.environ),
                        shell=False
                    )
                    print(f"[OK] Server started with basic flags, PID: {process.pid}")
            
            # Give the process a moment to start and check if it's still running
            time.sleep(1.0)  # Longer wait
            poll_result = process.poll()
            if poll_result is not None:
                # Process has already terminated
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                    stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                    print(f"[ERROR] Server process terminated immediately with code: {poll_result}")
                    print(f"[OUTPUT] STDOUT: {stdout_text}")
                    print(f"[OUTPUT] STDERR: {stderr_text}")
                except:
                    print(f"[ERROR] Server process terminated and couldn't get output")
                return None
            
            print(f"[SUCCESS] Server appears to be running successfully!")
            return process
            
        except Exception as e:
            print(f"[ERROR] Failed to start server: {e}")
            return None
    
    def wait_for_server_ready(self, work_dir, timeout=30):
        """Wait for server to be ready"""
        status_file = work_dir / "routing_status.json"
        
        for i in range(timeout * 10):  # Check every 0.1 seconds
            try:
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    if status.get('status') == 'idle':
                        print("[OK] Server is ready")
                        return True
                    elif status.get('status') == 'error':
                        print(f"[ERROR] Server error: {status.get('message', 'Unknown')}")
                        return False
                
                time.sleep(0.1)
                
            except Exception as e:
                time.sleep(0.1)
                continue
        
        print("[TIMEOUT] Timeout waiting for server")
        return False
    
    def send_routing_request(self, work_dir, board_data, config):
        """Send routing request to server"""
        request_data = {
            'board_data': board_data,
            'config': config,
            'timestamp': time.time()
        }
        
        input_file = work_dir / "routing_request.json"
        with open(input_file, 'w') as f:
            json.dump(request_data, f, indent=2)
        
        print("📨 Routing request sent")
    
    def monitor_routing_progress(self, work_dir):
        """Monitor routing progress with dialog"""
        status_file = work_dir / "routing_status.json"
        output_file = work_dir / "routing_result.json"
        
        # Create progress dialog
        progress_dlg = wx.ProgressDialog(
            "OrthoRoute Progress",
            "Initializing GPU routing server...",
            100,
            style=wx.PD_AUTO_HIDE | wx.PD_APP_MODAL | wx.PD_CAN_ABORT
        )
        
        try:
            last_progress = 0
            
            while True:
                try:
                    # Check if user cancelled
                    if not progress_dlg.Pulse()[0]:
                        print("[CANCEL] User cancelled routing")
                        return None
                    
                    # Check status
                    if status_file.exists():
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                        
                        current_status = status.get('status', 'unknown')
                        progress = status.get('progress', 0)
                        message = status.get('message', '')
                        
                        # Update progress dialog
                        if progress != last_progress or message:
                            progress_dlg.Update(progress, f"{current_status}: {message}")
                            last_progress = progress
                        
                        # Check for completion
                        if current_status == 'complete':
                            print("[OK] Routing completed")
                            break
                        elif current_status == 'error':
                            print(f"[ERROR] Routing error: {message}")
                            return {'success': False, 'error': message}
                    
                    # Check for result file
                    if output_file.exists():
                        with open(output_file, 'r') as f:
                            result = json.load(f)
                        
                        print("📨 Result received")
                        return result
                    
                    time.sleep(0.1)
                    
                except json.JSONDecodeError:
                    # File might be being written, try again
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"[WARN] Progress monitoring error: {e}")
                    time.sleep(0.5)
                    continue
        
        finally:
            progress_dlg.Destroy()
        
        return None
    
    def apply_routing_results(self, board, result):
        """Apply routing results to board"""
        try:
            routed_nets = result.get('routed_nets', [])
            
            print(f"📝 Applying {len(routed_nets)} routed nets to board...")
            
            for net_result in routed_nets:
                net_name = net_result.get('net_name', '')
                path = net_result.get('path', [])
                
                if len(path) < 2:
                    continue
                
                # Find the net
                net = None
                for net_id in range(board.GetNetCount()):
                    board_net = board.FindNet(net_id)
                    if board_net and board_net.GetNetname() == net_name:
                        net = board_net
                        break
                
                if not net:
                    continue
                
                # Create track segments
                for i in range(len(path) - 1):
                    start_point = path[i]
                    end_point = path[i + 1]
                    
                    # Create track
                    track = pcbnew.PCB_TRACK(board)
                    track.SetStart(pcbnew.VECTOR2I(start_point['x'], start_point['y']))
                    track.SetEnd(pcbnew.VECTOR2I(end_point['x'], end_point['y']))
                    track.SetWidth(pcbnew.FromMM(0.2))  # 0.2mm default width
                    track.SetLayer(pcbnew.F_Cu)  # Front copper layer
                    track.SetNet(net)
                    
                    board.Add(track)
            
            print(f"[OK] Applied routing results to board")
            
        except Exception as e:
            print(f"[ERROR] Error applying results: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_server(self, work_dir, process):
        """Stop the GPU server"""
        try:
            # Create shutdown signal
            shutdown_file = work_dir / "shutdown.flag"
            shutdown_file.touch()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                print("[OK] Server shutdown gracefully")
            except subprocess.TimeoutExpired:
                print("[WARN] Force terminating server...")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("[WARN] Force killing server...")
                    process.kill()
            
        except Exception as e:
            print(f"[WARN] Server shutdown error: {e}")

# Register the plugin
OrthoRoutePlugin().register()
