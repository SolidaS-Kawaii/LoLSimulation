"""
Display Module for Terminal UI
Handles all visual output formatting
"""

import sys
import os
from typing import List, Dict, Optional
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.draft_engine import DraftEngine, Phase
from src.core.champion_db import ChampionDatabase
import config


class Display:
    """
    Display manager for terminal UI
    
    Features:
    - Colored output
    - Formatted tables
    - ASCII art boxes
    - Draft state visualization
    """
    
    def __init__(self, use_color: bool = True, width: int = 70):
        """
        Initialize display
        
        Args:
            use_color: Whether to use colored output
            width: Display width in characters
        """
        self.use_color = use_color
        self.width = width
        
        # Color scheme
        self.colors = {
            'blue': Fore.CYAN,
            'red': Fore.RED,
            'success': Fore.GREEN,
            'error': Fore.RED,
            'warning': Fore.YELLOW,
            'highlight': Fore.YELLOW,
            'info': Fore.BLUE,
            'muted': Fore.WHITE
        }
    
    def color(self, text: str, color: str) -> str:
        """Apply color to text"""
        if not self.use_color:
            return text
        return self.colors.get(color, '') + str(text) + Style.RESET_ALL
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, text: str, char: str = '='):
        """Print fancy header box"""
        print(char * self.width)
        print(text.center(self.width))
        print(char * self.width)
    
    def print_subheader(self, text: str):
        """Print section subheader"""
        print(f"\n{self.color(text, 'highlight')}")
        print('-' * len(text))
    
    def print_welcome(self):
        """Display welcome screen"""
        self.clear_screen()
        print(self.color("╔" + "═" * (self.width - 2) + "╗", 'highlight'))
        print(self.color("║" + " " * (self.width - 2) + "║", 'highlight'))
        
        title = "🎮  LEAGUE OF LEGENDS DRAFT PICK SIMULATOR  🎮"
        print(self.color("║" + title.center(self.width - 2) + "║", 'highlight'))
        
        subtitle = "AI-Powered Champion Recommendations"
        print(self.color("║" + subtitle.center(self.width - 2) + "║", 'highlight'))
        
        print(self.color("║" + " " * (self.width - 2) + "║", 'highlight'))
        print(self.color("╚" + "═" * (self.width - 2) + "╝", 'highlight'))
    
    def print_mode_selection(self):
        """Display mode selection menu"""
        print("\n" + self.color("Select Game Mode:", 'info'))
        print("  [1] User vs AI   - You play one side, AI plays the other")
        print("  [2] User vs User - You control both sides")
        print("  [3] AI vs AI     - Watch AI play both sides (demo)")
        print("  [4] Settings")
        print("  [5] Exit")
    
    def print_side_selection(self):
        """Display side selection menu"""
        print("\n" + self.color("Select Your Side:", 'info'))
        print(f"  [{self.color('B', 'blue')}] Blue Side  - Pick first, Ban first")
        print(f"  [{self.color('R', 'red')}] Red Side   - Pick second, Counter pick last")
    
    def print_model_selection(self, models_info: Dict):
        """Display model selection menu"""
        print("\n" + self.color("Select AI Model:", 'info'))
        
        for i, (key, info) in enumerate(models_info.items(), 1):
            accuracy_str = f"{info['accuracy']:.2%}"
            print(f"  [{i}] {info['name']:<15} - {accuracy_str} accuracy")
        
        print(f"\n  Recommended: XGBoost (balanced performance)")
    
    def print_draft_state(self, engine: DraftEngine, champion_db: ChampionDatabase):
        """
        Display current draft state (bans + picks)
        
        Shows both teams side-by-side with bans and picks
        """
        state = engine.state
        
        # Header
        self.print_header("CURRENT DRAFT STATE", '═')
        
        # Phase and turn info
        turn_info = engine.get_turn_info()
        phase_display = turn_info['phase'].upper().replace('_', ' ')
        
        if turn_info['side']:
            current_side = turn_info['side'].upper()
            side_color = 'blue' if turn_info['side'] == 'blue' else 'red'
            print(f"Phase: {self.color(phase_display, 'highlight')}  |  "
                  f"Turn: {self.color(current_side, side_color)}")
        else:
            print(f"Phase: {self.color('COMPLETE', 'success')}")
        
        print()
        
        # Display bans and picks in two columns
        self._print_team_comparison(state, champion_db)
    
    def _print_team_comparison(self, state, champion_db: ChampionDatabase):
        """Display blue vs red side-by-side"""
        
        # Headers
        blue_header = self.color("┌─ BLUE SIDE " + "─" * 27 + "┐", 'blue')
        red_header = self.color("┌─ RED SIDE " + "─" * 28 + "┐", 'red')
        print(f"{blue_header}  {red_header}")
        
        # Bans
        blue_bans_str = self._format_bans(state.blue_bans, champion_db)
        red_bans_str = self._format_bans(state.red_bans, champion_db)
        
        print(f"{self.color('│', 'blue')} Bans: {blue_bans_str:<32}{self.color('│', 'blue')}  "
              f"{self.color('│', 'red')} Bans: {red_bans_str:<32}{self.color('│', 'red')}")
        
        print(f"{self.color('│', 'blue')}{' ' * 39}{self.color('│', 'blue')}  "
              f"{self.color('│', 'red')}{' ' * 39}{self.color('│', 'red')}")
        
        # Picks header
        print(f"{self.color('│ Picks:', 'blue')}{' ' * 32}{self.color('│', 'blue')}  "
              f"{self.color('│ Picks:', 'red')}{' ' * 32}{self.color('│', 'red')}")
        
        # Display each role
        for role in config.ROLES:
            blue_pick = self._get_pick_for_role(state.blue_picks, role)
            red_pick = self._get_pick_for_role(state.red_picks, role)
            
            blue_str = f"  {role:<10} {blue_pick:<20}" if blue_pick != "[Empty]" else f"  {role:<10} {self.color('[Empty]', 'muted'):<20}"
            red_str = f"  {role:<10} {red_pick:<20}" if red_pick != "[Empty]" else f"  {role:<10} {self.color('[Empty]', 'muted'):<20}"
            
            # Handle color codes in length calculation
            blue_display_len = len(f"  {role:<10} {blue_pick:<20}")
            red_display_len = len(f"  {role:<10} {red_pick:<20}")
            
            blue_padding = 39 - blue_display_len
            red_padding = 39 - red_display_len
            
            print(f"{self.color('│', 'blue')}{blue_str}{' ' * blue_padding}{self.color('│', 'blue')}  "
                  f"{self.color('│', 'red')}{red_str}{' ' * red_padding}{self.color('│', 'red')}")
        
        # Footer
        print(f"{self.color('└' + '─' * 39 + '┘', 'blue')}  "
              f"{self.color('└' + '─' * 39 + '┘', 'red')}")
    
    def _format_bans(self, bans: List[int], champion_db: ChampionDatabase) -> str:
        """Format ban list as string"""
        if not bans:
            return self.color("None", 'muted')
        
        ban_names = []
        for ban_id in bans:
            name = champion_db.get_name(ban_id)
            ban_names.append(name if name else f"ID:{ban_id}")
        
        return ", ".join(ban_names)
    
    def _get_pick_for_role(self, picks: List, role: str) -> str:
        """Get champion name for specific role"""
        for pick in picks:
            if pick.role == role:
                return pick.champion_name
        return "[Empty]"
    
    def print_recommendations(self, recommendations: List):
        """
        Display AI recommendations in table format
        
        Shows top 5 recommendations with scores
        """
        if not recommendations:
            print(self.color("\n⚠️  No recommendations available", 'warning'))
            return
        
        self.print_header("🏆 AI RECOMMENDATIONS", '═')
        
        # Table header
        header = f"{'Rank':<6}{'Champion':<16}{'Role':<10}{'Score':<8}{'Win%':<8}{'Synergy':<10}{'Counter':<10}{'Meta':<8}"
        print(self.color(header, 'highlight'))
        print('-' * len(header))
        
        # Recommendations
        for rec in recommendations:
            rank_str = f"#{rec.rank}"
            score_str = f"{rec.total_score:.3f}"
            win_str = f"{rec.win_probability*100:.1f}%"
            syn_str = f"{rec.synergy_score:.3f}"
            cnt_str = f"{rec.counter_score:.3f}"
            meta_str = f"{rec.meta_score:.3f}"
            
            row = f"{rank_str:<6}{rec.champion_name:<16}{rec.role:<10}{score_str:<8}{win_str:<8}{syn_str:<10}{cnt_str:<10}{meta_str:<8}"
            
            # Highlight #1 recommendation
            if rec.rank == 1:
                print(self.color(row, 'success'))
            else:
                print(row)
    
    def print_champion_info(self, champion_name: str, recommendation):
        """Display detailed champion information"""
        self.print_header(f"📊 {champion_name.upper()} - DETAILED INFO", '═')
        
        print(f"\n{self.color('Role:', 'info')} {recommendation.role}")
        print(f"{self.color('Total Score:', 'info')} {recommendation.total_score:.3f}")
        
        # Scores breakdown
        print(f"\n{self.color('Score Breakdown:', 'highlight')}")
        print(f"  • Win Probability: {recommendation.win_probability:.3f} ({recommendation.win_probability*100:.1f}%)")
        print(f"  • Synergy Score:   {recommendation.synergy_score:.3f}")
        print(f"  • Counter Score:   {recommendation.counter_score:.3f}")
        print(f"  • Meta Score:      {recommendation.meta_score:.3f}")
        
        # Synergies
        if recommendation.synergy_pairs:
            print(f"\n{self.color('Team Synergies:', 'highlight')}")
            for champ, score in recommendation.synergy_pairs:
                print(f"  • {champ:<20} {score:.3f} ({self._score_label(score)})")
        
        # Matchups
        if recommendation.counter_matchups:
            print(f"\n{self.color('Enemy Matchups:', 'highlight')}")
            for enemy, wr in recommendation.counter_matchups:
                wr_pct = wr * 100
                color = 'success' if wr > 0.52 else 'error' if wr < 0.48 else 'muted'
                print(f"  • vs {enemy:<20} {self.color(f'{wr_pct:.1f}%', color)}")
        
        # Meta stats
        if recommendation.meta_stats:
            stats = recommendation.meta_stats
            print(f"\n{self.color('Meta Statistics:', 'highlight')}")
            print(f"  • Pick Rate: {stats['pick_rate']:.1%}")
            print(f"  • Ban Rate:  {stats['ban_rate']:.1%}")
            print(f"  • Win Rate:  {stats['win_rate']:.1%}")
    
    def _score_label(self, score: float) -> str:
        """Get label for score value"""
        if score >= 0.55:
            return self.color("Excellent", 'success')
        elif score >= 0.52:
            return self.color("Good", 'info')
        elif score >= 0.48:
            return "Average"
        else:
            return self.color("Weak", 'warning')
    
    def print_turn_info(self, turn_info: Dict):
        """Display whose turn it is and what action"""
        action = turn_info.get('action', 'unknown')
        side = turn_info.get('side')
        
        if not side:
            return
        
        side_color = 'blue' if side == 'blue' else 'red'
        action_str = action.upper()
        side_str = side.upper()
        
        print(f"\n{self.color('═' * self.width, 'highlight')}")
        print(f"{self.color(side_str, side_color)}'s turn to {self.color(action_str, 'highlight')}")
        
        if action == 'ban':
            remaining = turn_info.get('blue_bans_remaining' if side == 'blue' else 'red_bans_remaining', 0)
            print(f"Bans remaining: {remaining}")
        elif action == 'pick':
            remaining = turn_info.get('blue_picks_remaining' if side == 'blue' else 'red_picks_remaining', 0)
            vacant = turn_info.get('vacant_roles', [])
            print(f"Picks remaining: {remaining}")
            print(f"Vacant roles: {', '.join(vacant)}")
        
        print(self.color('═' * self.width, 'highlight'))
    
    def print_commands(self):
        """Display available commands"""
        print(f"\n{self.color('Available Commands:', 'info')}")
        print("  ban <champion>     - Ban a champion (e.g., 'ban yasuo')")
        print("  pick <champion>    - Pick a champion (e.g., 'pick malphite')")
        print("  search <name>      - Search for a champion")
        print("  info <champion>    - Show detailed champion info")
        print("  refresh            - Update recommendations")
        print("  help               - Show all commands")
        print("  quit               - Exit the game")
    
    def print_success(self, message: str):
        """Print success message"""
        print(self.color(f"✓ {message}", 'success'))
    
    def print_error(self, message: str):
        """Print error message"""
        print(self.color(f"✗ {message}", 'error'))
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(self.color(f"⚠️  {message}", 'warning'))
    
    def print_info(self, message: str):
        """Print info message"""
        print(self.color(f"ℹ️  {message}", 'info'))
    
    def print_winner_prediction(self, blue_prob: float, red_prob: float):
        """Display winner prediction"""
        self.print_header("🎯 DRAFT COMPLETE - WINNER PREDICTION", '═')
        
        print(f"\n{self.color('Blue Team Win Probability:', 'blue')} {blue_prob*100:.1f}%")
        print(f"{self.color('Red Team Win Probability:', 'red')} {red_prob*100:.1f}%")
        
        print()
        winner = 'blue' if blue_prob > red_prob else 'red'
        winner_prob = max(blue_prob, red_prob)
        
        winner_color = 'blue' if winner == 'blue' else 'red'
        print(f"{self.color('Predicted Winner:', 'highlight')} "
              f"{self.color(winner.upper(), winner_color)} "
              f"({winner_prob*100:.1f}% confidence)")
    
    def print_search_results(self, query: str, matches: List[str]):
        """Display search results"""
        if not matches:
            self.print_error(f"No champions found matching '{query}'")
            return
        
        print(f"\n{self.color(f'Search results for:', 'info')} '{query}'")
        print(f"Found {len(matches)} champion(s):")
        
        for i, name in enumerate(matches, 1):
            print(f"  [{i}] {name}")
    
    def prompt(self, message: str = "> ") -> str:
        """Display prompt and get user input"""
        return input(f"\n{message}")


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test display module"""
    
    print("="*70)
    print("DISPLAY MODULE - TESTING")
    print("="*70)
    
    from src.core.draft_engine import DraftEngine
    from src.core.champion_db import ChampionDatabase
    
    # Initialize
    display = Display()
    champion_db = ChampionDatabase()
    champion_db.load(verbose=False)
    
    # Test 1: Welcome screen
    print("\n1. Testing Welcome Screen:")
    input("Press Enter to show welcome screen...")
    display.print_welcome()
    
    # Test 2: Mode selection
    print("\n2. Testing Mode Selection:")
    input("Press Enter to continue...")
    display.print_mode_selection()
    
    # Test 3: Draft state
    print("\n3. Testing Draft State Display:")
    input("Press Enter to continue...")
    
    engine = DraftEngine(user_side='blue')
    
    # Execute some bans
    bans = [157, 238, 61, 18, 11, 201]
    for ban_id in bans:
        engine.execute_ban(ban_id)
    
    # Execute some picks
    picks = [
        (222, "Jinx", "BOTTOM", "blue"),
        (498, "Kai'Sa", "BOTTOM", "red"),
        (412, "Thresh", "UTILITY", "red"),
    ]
    
    for champ_id, name, role, side in picks:
        engine.execute_pick(champ_id, name, role, side)
    
    display.print_draft_state(engine, champion_db)
    
    # Test 4: Turn info
    print("\n4. Testing Turn Info:")
    input("Press Enter to continue...")
    turn_info = engine.get_turn_info()
    display.print_turn_info(turn_info)
    
    # Test 5: Commands
    print("\n5. Testing Commands Display:")
    input("Press Enter to continue...")
    display.print_commands()
    
    # Test 6: Messages
    print("\n6. Testing Messages:")
    input("Press Enter to continue...")
    display.print_success("Successfully picked Yasuo!")
    display.print_error("Champion already banned!")
    display.print_warning("Model not loaded, using default")
    display.print_info("It's now Blue team's turn")
    
    print("\n" + "="*70)
    print("✓ Display module testing complete")
    print("="*70)
