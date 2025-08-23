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
from orthoroute.shared.utils.logging_utils import setup_logging


def setup_environment():
    """Setup the application environment."""
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


def run_cli(board_file: str, output_dir: str = ".", config_path: Optional[str] = None):
    """Run command line interface."""
    try:
        from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
        from orthoroute.application.services.routing_orchestrator import RoutingOrchestrator
        from orthoroute.infrastructure.gpu.cuda_provider import CUDAProvider
        from orthoroute.infrastructure.gpu.cpu_fallback import CPUProvider
        
        # Initialize configuration if custom path provided
        if config_path:
            initialize_config(config_path)
        
        config = setup_environment()
        
        # Setup services
        try:
            gpu_provider = CUDAProvider()
            logging.info("Using CUDA GPU acceleration")
        except Exception:
            gpu_provider = CPUProvider()
            logging.info("Using CPU fallback")
        
        # Load board
        logging.info(f"Loading board from: {board_file}")
        parser = KiCadFileParser()
        board = parser.load_board(board_file)
        
        if not board:
            logging.error("Failed to load board file")
            sys.exit(1)
        
        logging.info(f"Loaded board: {board.name} with {len(board.nets)} nets")
        
        # Setup orchestrator and route
        orchestrator = RoutingOrchestrator(gpu_provider=gpu_provider)
        orchestrator.set_board(board)
        
        # Route all nets
        logging.info("Starting routing process...")
        results = orchestrator.route_all_nets_sync(timeout_per_net=30.0)
        
        if results:
            logging.info(f"Routing completed: {results.routed_nets}/{results.total_nets} nets routed")
            logging.info(f"Success rate: {results.success_rate:.1%}")
            
            # Save results (implementation would depend on desired output format)
            logging.info(f"Results saved to: {output_dir}")
            
        else:
            logging.error("Routing failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"CLI execution failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OrthoRoute - KiCad PCB Autorouter Plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run KiCad plugin with GUI (default)
  %(prog)s plugin                       # Run as KiCad plugin with GUI  
  %(prog)s plugin --no-gui              # Run as KiCad plugin without GUI
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no arguments (default to plugin mode)
    if not args.mode:
        # Default to plugin mode with GUI
        run_plugin(show_gui=True)
        return
    
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


if __name__ == '__main__':
    main()