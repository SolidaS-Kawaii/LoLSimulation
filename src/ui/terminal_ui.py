"""
Terminal UI Main Controller
Coordinates all UI components and game flow
"""

import sys
import os
import time
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ui.display import Display
from src.ui.input_handler import InputHandler
from src.ui.game_controller import GameController
from src.core.champion_db import ChampionDatabase
from src.ai.recommender import ChampionRecommender
from src.ai.predictor import WinProbabilityPredictor
from src.ai.model_manager import ModelManager
from src.features.feature_extractor import FeatureExtractor
from src.features.synergy_calculator import SynergyCalculator
from src.features.matchup_calculator import MatchupCalculator
from src.features.meta_analyzer import MetaAnalyzer
import config


class TerminalUI:
    """
    Main Terminal UI Controller
    
    Coordinates:
    - Display
    - Input handling
    - Game controller
    - AI components
    """
    
    def __init__(self):
        """Initialize terminal UI"""
        self.display = Display()
        
        # Components (initialized later)
        self.champion_db: Optional[ChampionDatabase] = None
        self.input_handler: Optional[InputHandler] = None
        self.recommender: Optional[ChampionRecommender] = None
        self.predictor: Optional[WinProbabilityPredictor] = None
        self.controller: Optional[GameController] = None
        
        # Settings
        self.model_name = config.DEFAULT_MODEL
        self.auto_play_speed = 1.5  # seconds per AI turn in AI vs AI mode
        self.turn_counter = 0  # Track turn number for separator display

    def run(self):
        """Main entry point"""
        try:
            # Show welcome screen
            self.display.print_welcome()
            time.sleep(1)
            
            # Initialize components
            if not self._initialize_components():
                self.display.print_error("Failed to initialize components. Exiting.")
                return
            
            # Main menu loop
            while True:
                mode = self._select_mode()
                
                if mode == 'exit':
                    self.display.print_info("Thank you for playing! Goodbye! 👋")
                    break
                elif mode == 'settings':
                    self._show_settings()
                    continue
                
                # Run game for selected mode
                self._run_game(mode)
                
                # Ask if want to play again
                self.display.clear_screen()
                play_again = input("\nPlay again? (y/n): ").strip().lower()
                
                if play_again not in ['y', 'yes']:
                    self.display.print_info("Thank you for playing! Goodbye! 👋")
                    break
        
        except KeyboardInterrupt:
            print("\n\n")
            self.display.print_info("Interrupted by user. Exiting...")
        except Exception as e:
            print("\n")
            self.display.print_error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_components(self) -> bool:
        """Initialize all components"""
        try:
            self.display.clear_screen()
            self.display.print_header("INITIALIZING SYSTEM", '═')
            
            # Champion database
            print("\n📚 Loading champion database...")
            self.champion_db = ChampionDatabase()
            self.champion_db.load(verbose=False)
            self.display.print_success("Champion database loaded")
            
            # Input handler
            self.input_handler = InputHandler(self.champion_db)
            
            # Model manager (auto-loads all models)
            print("\n🤖 Loading AI models...")
            try:
                model_manager = ModelManager()  # Auto-loads models in __init__
                
                # Check if default model loaded successfully
                if self.model_name in model_manager.models:
                    model_info = config.AVAILABLE_MODELS[self.model_name]
                    self.display.print_success(f"Model loaded: {model_info['name']}")
                else:
                    self.display.print_warning("Default model not found, using any available model")
                    # Use first available model
                    if model_manager.models:
                        self.model_name = list(model_manager.models.keys())[0]

            except Exception as e:
                self.display.print_warning(f"Model loading failed: {e}")
                self.display.print_warning("Some features may be limited")
                model_manager = None  # Handle gracefully
            
            # Feature calculator components
            print("\n⚙️  Initializing feature calculators...")
            feature_extractor = FeatureExtractor()
            synergy_calc = SynergyCalculator()
            matchup_calc = MatchupCalculator()
            meta_analyzer = MetaAnalyzer()
            self.display.print_success("Feature calculators ready")
            
            # Predictor and recommender
            print("\n🎯 Initializing AI recommender...")
            self.predictor = WinProbabilityPredictor(model_manager, feature_extractor)
            self.recommender = ChampionRecommender(
                self.predictor,
                synergy_calc,
                matchup_calc,
                meta_analyzer,
                self.champion_db
            )
            self.display.print_success("AI recommender ready")
            
            print("\n" + "="*70)
            self.display.print_success("All systems initialized! 🚀")
            time.sleep(1.5)
            
            return True
        
        except Exception as e:
            self.display.print_error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _refresh_display(self, turn_number: Optional[int] = None, phase: Optional[str] = None):
        """
        Refresh display based on UI_CLEAR_MODE setting

        This method provides flexibility to switch between clear screen
        and separator modes without changing the rest of the code.

        Args:
            turn_number: Current turn number (for separator mode)
            phase: Current phase name (for separator mode)
        """
        if config.UI_CLEAR_MODE == 'clear':
            self.display.clear_screen()
        else:  # separator mode
            self.display.print_separator(turn_number, phase)

    def _select_mode(self) -> str:
        """Select game mode"""
        self.display.clear_screen()
        self.display.print_welcome()
        self.display.print_mode_selection()
        
        while True:
            choice = self.display.prompt("Your choice: ").strip()
            
            if choice == '1':
                return 'user_vs_ai'
            elif choice == '2':
                return 'user_vs_user'
            elif choice == '3':
                return 'ai_vs_ai'
            elif choice == '4':
                return 'settings'
            elif choice == '5':
                return 'exit'
            else:
                self.display.print_error("Invalid choice. Please select 1-5.")
    
    def _show_settings(self):
        """Show settings menu"""
        self.display.clear_screen()
        self.display.print_header("SETTINGS", '═')
        
        # Model selection
        self.display.print_model_selection(config.AVAILABLE_MODELS)
        
        choice = self.display.prompt("\nSelect model (1-3) or Enter to keep current: ").strip()
        
        if choice == '1':
            self.model_name = 'lightgbm'
        elif choice == '2':
            self.model_name = 'xgboost'
        elif choice == '3':
            self.model_name = 'random_forest'
        
        model_info = config.AVAILABLE_MODELS[self.model_name]
        self.display.print_success(f"Active model: {model_info['name']}")
        
        input("\nPress Enter to continue...")
    
    def _run_game(self, mode: str):
        """Run game for selected mode"""
        # Setup game
        if mode == 'user_vs_ai':
            user_side = self._select_side()
        elif mode == 'ai_vs_ai':
            user_side = 'blue'  # Doesn't matter for AI vs AI
        else:  # user_vs_user
            user_side = 'blue'  # User controls both
        
        # Create game controller
        self.controller = GameController(
            mode=mode,
            user_side=user_side,
            model_name=self.model_name,
            champion_db=self.champion_db,
            recommender=self.recommender,
            predictor=self.predictor
        )
        
        # Run game loop
        if mode == 'ai_vs_ai':
            self._run_ai_vs_ai()
        else:
            self._run_interactive_game()
        
        # Show final results
        self._show_results()
    
    def _select_side(self) -> str:
        """Select user's side"""
        self.display.clear_screen()
        self.display.print_side_selection()
        
        while True:
            choice = self.display.prompt("Your choice: ").strip().lower()
            
            if choice in ['b', 'blue']:
                return 'blue'
            elif choice in ['r', 'red']:
                return 'red'
            else:
                self.display.print_error("Invalid choice. Please select B or R.")

    def _run_interactive_game(self):
        """Run interactive game (User vs AI or User vs User)"""
        self.display.clear_screen()
        self.display.print_header("DRAFT PHASE STARTED", '═')
        time.sleep(1)

        self.turn_counter = 0  # Reset turn counter

        while not self.controller.is_complete():
            self.turn_counter += 1

            # Get current phase for display
            turn_info = self.controller.engine.get_turn_info()
            phase_name = turn_info.get('phase', '').replace('_', ' ').upper()

            # Display current state with separator (or clear based on config)
            self._refresh_display(self.turn_counter, phase_name)
            self.display.print_draft_state(self.controller.engine, self.champion_db)

            # Check whose turn
            if self.controller.is_user_turn():
                self._handle_user_turn()
            else:
                self._handle_ai_turn()
    
    def _handle_user_turn(self):
        """Handle user's turn"""
        # Show turn info
        turn_info = self.controller.engine.get_turn_info()
        self.display.print_turn_info(turn_info)

        if turn_info['action'] == 'pick':
            recommendations = self.controller.get_recommendations()
            if recommendations:
                print()  # Blank line
                self.display.print_recommendations(recommendations)

                    # Pause to let user read recommendations
                self.display.print_pause()
        
        # Show commands
        self.display.print_commands()
        
        # Get and handle commands
        while True:
            cmd = self.input_handler.get_command()
            
            if not cmd['valid']:
                self.display.print_error(cmd.get('error', 'Invalid command'))
                continue
            
            # Handle command
            if cmd['action'] == 'ban':
                if self._execute_user_ban(cmd):
                    break
            
            elif cmd['action'] == 'pick':
                if self._execute_user_pick(cmd):
                    break
            
            elif cmd['action'] == 'search':
                self._handle_search(cmd['query'])
            
            elif cmd['action'] == 'info':
                self._handle_info(cmd)
            
            elif cmd['action'] == 'refresh':
                self.display.print_info("Refreshing recommendations...")
                break  # Redraw screen
            
            elif cmd['action'] == 'help':
                self.display.print_commands()
            
            elif cmd['action'] == 'quit':
                confirm = input("\nAre you sure you want to quit? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    self.display.print_info("Exiting game...")
                    # Force complete to exit game loop
                    while not self.controller.is_complete():
                        self.controller.engine.state.blue_picks = [None] * 5
                        self.controller.engine.state.red_picks = [None] * 5
                    return
    
    def _execute_user_ban(self, cmd: Dict) -> bool:
        """Execute user ban command"""
        champion_id = cmd['champion_id']
        champion_name = cmd['champion']
        
        # Validate
        is_valid, error = self.input_handler.validate_ban(
            champion_id,
            self.controller.engine.state
        )
        
        if not is_valid:
            self.display.print_error(error)
            return False
        
        # Execute
        success = self.controller.execute_user_ban(champion_id, champion_name)
        
        if success:
            self.display.print_success(f"Banned {champion_name}")
            time.sleep(0.5)
            return True
        else:
            self.display.print_error("Failed to execute ban")
            return False
    
    def _execute_user_pick(self, cmd: Dict) -> bool:
        """Execute user pick command"""
        champion_id = cmd['champion_id']
        champion_name = cmd['champion']
        role = cmd.get('role')
        
        current_side = self.controller.engine.get_current_side()
        
        # Validate
        is_valid, error, suggested_role = self.input_handler.validate_pick(
            champion_id,
            role,
            self.controller.engine.state,
            current_side
        )
        
        if not is_valid:
            self.display.print_error(error)
            return False
        
        # Use suggested role if not provided
        if role is None:
            role = suggested_role
        
        # Execute
        success = self.controller.execute_user_pick(champion_id, champion_name, role)
        
        if success:
            self.display.print_success(f"Picked {champion_name} ({role})")
            time.sleep(0.5)
            return True
        else:
            self.display.print_error("Failed to execute pick")
            return False
    
    def _handle_search(self, query: str):
        """Handle search command"""
        results = self.input_handler.search_champions(query, max_results=10)
        self.display.print_search_results(query, results)
        input("\nPress Enter to continue...")
    
    def _handle_info(self, cmd: Dict):
        """Handle info command"""
        champion_name = cmd['champion']
        
        # Get recommendations to find this champion
        recommendations = self.controller.get_recommendations()
        
        # Find champion in recommendations
        rec = next((r for r in recommendations if r.champion_name == champion_name), None)
        
        if rec:
            print()  # Blank line
            self.display.print_champion_info(champion_name, rec)
        else:
            # Show basic info from database
            roles = self.champion_db.get_roles(champion_name)
            
            if roles:
                print(f"\n{champion_name}:")
                print(f"Available roles:")
                for role_data in roles:
                    print(f"  • {role_data['role']}: "
                          f"Pick Rate {role_data['pick_rate']:.1%}, "
                          f"Win Rate {role_data['win_rate']:.1%}")
            else:
                self.display.print_warning(f"No data available for {champion_name}")
        
        input("\nPress Enter to continue...")
    
    def _handle_ai_turn(self):
        """Handle AI's turn"""
        self.display.print_info("AI is thinking...")
        time.sleep(1)
        
        ai_action = self.controller.execute_ai_turn()
        
        if ai_action:
            action_str = ai_action['action'].upper()
            champ_str = ai_action['champion_name']
            
            if ai_action['action'] == 'pick':
                role_str = ai_action['role']
                self.display.print_success(f"AI {action_str}: {champ_str} ({role_str})")
            else:
                self.display.print_success(f"AI {action_str}: {champ_str}")
            
            time.sleep(1.5)

    def _run_ai_vs_ai(self):
        """Run AI vs AI demo mode"""
        self.display.clear_screen()
        self.display.print_header("AI VS AI DEMO MODE", '═')
        self.display.print_info("Watch the AI draft against itself!")
        self.display.print_info(f"Auto-play speed: {self.auto_play_speed}s per turn")
        input("\nPress Enter to start...")

        self.turn_counter = 0  # Reset turn counter

        while not self.controller.is_complete():
            self.turn_counter += 1

            # Get current phase for display
            turn_info = self.controller.engine.get_turn_info()
            phase_name = turn_info.get('phase', '').replace('_', ' ').upper()

            # Display state with separator (or clear based on config)
            self._refresh_display(self.turn_counter, phase_name)
            self.display.print_draft_state(self.controller.engine, self.champion_db)

            # Show turn
            self.display.print_turn_info(turn_info)

            # AI takes turn
            print(f"\nTurn {self.turn_counter}: AI is thinking...")
            time.sleep(self.auto_play_speed)

            ai_action = self.controller.execute_ai_turn()

            if ai_action:
                action_str = ai_action['action'].upper()
                champ_str = ai_action['champion_name']
                side_str = ai_action['side'].upper()

                if ai_action['action'] == 'pick':
                    role_str = ai_action['role']
                    self.display.print_success(f"{side_str} AI {action_str}: {champ_str} ({role_str})")
                else:
                    self.display.print_success(f"{side_str} AI {action_str}: {champ_str}")

                time.sleep(self.auto_play_speed * 0.5)
    
    def _show_results(self):
        """Show final results"""
        if not self.controller.is_complete():
            return
        
        self.display.clear_screen()
        
        # Final draft state
        self.display.print_draft_state(self.controller.engine, self.champion_db)
        
        # Winner prediction
        print("\n")
        prediction = self.controller.get_winner_prediction()
        self.display.print_winner_prediction(
            prediction['blue_win_prob'],
            prediction['red_win_prob']
        )
        
        # Offer to export
        print("\n")
        export = input("Export draft history? (y/n): ").strip().lower()
        
        if export in ['y', 'yes']:
            filepath = self.controller.export_draft()
            self.display.print_success(f"Draft exported to: {filepath}")
        
        input("\nPress Enter to continue...")


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test terminal UI"""
    
    print("This is the main Terminal UI module.")
    print("Run main.py to start the application.")
    print("\nTesting mode: Initialize components only...")
    
    try:
        ui = TerminalUI()
        
        if ui._initialize_components():
            print("\n✓ All components initialized successfully!")
            print("Ready to run full application.")
        else:
            print("\n✗ Component initialization failed.")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
