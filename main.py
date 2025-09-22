#!/usr/bin/env python3
"""
OrthoRoute - Main Entry Point
Advanced PCB autorouter with Manhattan routing and GPU acceleration
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add the package directory to Python path
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from orthoroute.shared.configuration import initialize_config, get_config
from orthoroute.shared.utils.logging_utils import setup_logging, init_logging


def setup_environment():
    """Setup the application environment."""
    # Initialize early logging for acceptance test
    init_logging()

    # Initialize configuration
    config = initialize_config()

    # Setup logging
    setup_logging(config.get_settings().logging)
    
    return config


def show_usage():
    """Show usage information."""
    print("OrthoRoute - KiCad PCB Autorouter")
    print("Usage:")
    print("  python main.py                      # Run KiCad plugin with GUI (default)")
    print("  python main.py plugin               # Run as KiCad plugin with GUI")
    print("  python main.py plugin --no-gui      # Run as KiCad plugin without GUI")
    print("  python main.py cli board.kicad_pcb  # Command line mode")
    print("")
    print("Alternative entry point:")
    print("  python src/orthoroute_plugin.py")
    sys.exit(0)


def run_plugin(show_gui: bool = False):
    """Run as KiCad plugin with the same GUI as orthoroute_plugin.py."""
    try:
        config = setup_environment()
        
        # Use new architecture for both GUI and non-GUI modes
        from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin
        
        plugin = KiCadPlugin()
        
        if show_gui:
            success = plugin.run_with_gui()
        else:
            success = plugin.run()
        
        if success:
            logging.info("Plugin execution completed successfully")
            sys.exit(0)
        else:
            logging.error("Plugin execution failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Plugin execution failed: {e}")
        sys.exit(1)


def run_test_manhattan():
    """Run automated Manhattan routing test without GUI."""
    try:
        config = setup_environment()
        print("Starting automated Manhattan routing test...")
        logging.info("Starting automated Manhattan routing test...")
        
        from orthoroute.presentation.plugin.kicad_plugin import KiCadPlugin
        
        plugin = KiCadPlugin()
        
        # Run with GUI for automated testing and auto-start routing
        print("Loading board from KiCad and starting GUI...")
        print("Auto-starting routing process...")
        success = plugin.run_with_gui_autostart()
        
        if success:
            logging.info("Manhattan routing test completed successfully")
            print("TEST PASSED: Manhattan routing executed without errors")
            sys.exit(0)
        else:
            logging.error("Manhattan routing test failed")
            print("TEST FAILED: Manhattan routing encountered errors")
            print("Note: Make sure KiCad is running with a board that has routable nets")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Manhattan routing test failed with exception: {e}")
        print(f"TEST FAILED: Exception occurred: {e}")
        if "division by zero" in str(e):
            print("Note: This typically occurs when the board has no routable nets")
            print("Make sure KiCad is running with a board that has components with nets to route")
        elif "No KiCad process" in str(e):
            print("Note: KiCad must be running for the test to work")
        sys.exit(1)


def run_cli(board_file: str, output_dir: str = ".", config_path: Optional[str] = None):
    """Run command line interface using unified pipeline (same as GUI)."""
    try:
        from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
        from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig

        # Initialize configuration if custom path provided
        if config_path:
            initialize_config(config_path)

        config = setup_environment()

        # Load board
        logging.info(f"Loading board from: {board_file}")
        parser = KiCadFileParser()
        board = parser.load_board(board_file)

        if not board:
            logging.error("Failed to load board file")
            sys.exit(1)

        logging.info(f"Loaded board: {board.name} with {len(board.nets)} nets")

        # Create UnifiedPathFinder (same as GUI) - FORCE CPU-ONLY
        pf = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        logging.info(f"[CLI] Created UnifiedPathFinder with instance_tag={pf._instance_tag}")

        # Use unified pipeline (SAME THREE CALLS AS GUI)
        logging.info("[CLI] Step 1: Building lattice & CSR...")
        pf.initialize_graph(board)

        logging.info("[CLI] Step 2: Mapping pads to lattice...")
        pf.map_all_pads(board)

        logging.info("[CLI] Step 3: Preparing routing runtime...")
        pf.prepare_routing_runtime()

        logging.info("[CLI] Step 4: Routing nets...")
        pf.route_multiple_nets(board.nets)

        logging.info("[CLI] Step 5: Emitting geometry...")
        tracks, vias = pf.emit_geometry(board)

        logging.info(f"[CLI] Routing completed: {tracks} tracks, {vias} vias")

        if tracks > 0 or vias > 0:
            # Save geometry results
            geom = pf.get_geometry_payload()
            logging.info(f"[CLI] Generated {len(geom.tracks)} track objects, {len(geom.vias)} via objects")
            logging.info(f"[CLI] Results would be saved to: {output_dir}")
        else:
            logging.warning("[CLI] No copper generated")
            sys.exit(1)

    except Exception as e:
        logging.error(f"CLI execution failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    import time
    run_start = time.time()

    parser = argparse.ArgumentParser(
        description="OrthoRoute - KiCad PCB Autorouter Plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run KiCad plugin with GUI (default)
  %(prog)s plugin                       # Run as KiCad plugin with GUI
  %(prog)s plugin --no-gui              # Run as KiCad plugin without GUI
  %(prog)s --test-manhattan             # Run automated Manhattan routing test
  %(prog)s cli board.kicad_pcb          # Route board via CLI
  %(prog)s cli board.kicad_pcb -o out/  # Route and save to directory
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Plugin mode
    plugin_parser = subparsers.add_parser('plugin', help='Run as KiCad plugin')
    plugin_parser.add_argument(
        '--no-gui', action='store_true',
        help='Run without GUI (default shows GUI)'
    )
    plugin_parser.add_argument(
        '--min-run-sec', type=int, default=0,
        help='Keep process alive for at least this many seconds (for CI/agents)'
    )
    
    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Command line interface')
    cli_parser.add_argument(
        'board_file',
        help='KiCad board file (.kicad_pcb)'
    )
    cli_parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    cli_parser.add_argument(
        '-c', '--config',
        help='Configuration file path'
    )
    
    # Global options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    parser.add_argument(
        '--test-manhattan',
        action='store_true',
        help='Run automated Manhattan routing test without GUI'
    )
    parser.add_argument(
        '--min-run-sec', type=int, default=0,
        help='Keep process alive for at least this many seconds (for CI/agents)'
    )
    
    # Parse arguments
    args = parser.parse_args()

    min_run = int(getattr(args, "min_run_sec", 0) or 0)
    if min_run > 0:
        logging.getLogger().info(f"[RUN-MIN] min_run_sec={min_run}")

    # Check for test mode first (overrides other modes)
    if getattr(args, 'test_manhattan', False):
        run_test_manhattan()
    elif not args.mode:
        # Handle no arguments (default to plugin mode)
        run_plugin(show_gui=True)
    else:
        # Route to appropriate handler
        try:
            if args.mode == 'plugin':
                run_plugin(show_gui=not getattr(args, 'no_gui', False))
            elif args.mode == 'cli':
                run_cli(
                    args.board_file,
                    args.output,
                    getattr(args, 'config', None)
                )
            else:
                parser.error(f"Unknown mode: {args.mode}")

        except KeyboardInterrupt:
            logging.info("Operation cancelled by user")
            sys.exit(130)

    # Keep-alive for headless/short agent runs
    if min_run > 0:
        remaining = max(0.0, min_run - (time.time() - run_start))
        if remaining > 0:
            logging.getLogger().info(f"[RUN-MIN] Sleeping {remaining:.1f}s to satisfy min runtime")
            time.sleep(remaining)


if __name__ == '__main__':
    main()