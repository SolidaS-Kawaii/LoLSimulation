"""
LoL Draft Simulator - Main Entry Point
Terminal UI Application
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.terminal_ui import TerminalUI


def main():
    """
    Main entry point for LoL Draft Simulator Terminal UI
    
    Features:
    - 3 Game Modes: User vs AI, User vs User, AI vs AI
    - AI-powered champion recommendations
    - Real-time draft simulation
    - Winner prediction
    - Draft history export
    """
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        🎮  LEAGUE OF LEGENDS DRAFT PICK SIMULATOR  🎮           ║
║                                                                  ║
║              AI-Powered Champion Recommendations                 ║
║                                                                  ║
║                         Version 1.0.0                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize and run UI
        ui = TerminalUI()
        ui.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n" + "="*70)
        print("Error Report:")
        print("="*70)
        print("If this error persists, please check:")
        print("  1. All required data files are in data/ directory")
        print("  2. Model files (.pkl) are in models/ directory")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")
        print("\nFor support, please report this error with the full traceback.")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
