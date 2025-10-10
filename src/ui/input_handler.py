"""
Input Handler Module for Terminal UI
Handles command parsing, validation, and fuzzy matching
"""

import sys
import os
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.draft_engine import DraftState
from src.core.champion_db import ChampionDatabase
from src.utils.text_utils import find_closest_champion, clean_champion_name, fuzzy_match
import config


class InputHandler:
    """
    Input handler for terminal UI
    
    Features:
    - Command parsing (ban, pick, search, info, etc.)
    - Input validation
    - Fuzzy champion name matching
    - Role auto-detection
    """
    
    def __init__(self, champion_db: ChampionDatabase):
        """
        Initialize input handler
        
        Args:
            champion_db: Champion database for validation and fuzzy matching
        """
        self.champion_db = champion_db
        self.command_history: List[str] = []
        
        # Command aliases
        self.aliases = {
            'b': 'ban',
            'p': 'pick',
            's': 'search',
            'i': 'info',
            'r': 'refresh',
            'h': 'help',
            'q': 'quit',
            'exit': 'quit'
        }
    
    def get_command(self, prompt: str = "> ") -> Dict:
        """
        Get and parse user command
        
        Returns:
            {
                'action': str,          # 'ban', 'pick', 'search', 'info', 'help', 'quit', 'refresh'
                'champion': str,        # Champion name (if applicable)
                'champion_id': int,     # Champion ID (if found)
                'role': str,            # Role (if applicable)
                'valid': bool,          # Whether command is valid
                'error': str            # Error message (if invalid)
            }
        """
        user_input = input(prompt).strip()
        
        if not user_input:
            return {
                'action': 'empty',
                'valid': False,
                'error': 'No command entered'
            }
        
        # Add to history
        self.command_history.append(user_input)
        
        # Parse command
        return self.parse_command(user_input)
    
    def parse_command(self, user_input: str) -> Dict:
        """
        Parse user input into structured command
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Parsed command dictionary
        """
        parts = user_input.lower().split()
        
        if not parts:
            return {
                'action': 'empty',
                'valid': False,
                'error': 'No command entered'
            }
        
        # Get action (with alias support)
        action = parts[0]
        if action in self.aliases:
            action = self.aliases[action]
        
        # Route to appropriate parser
        if action == 'ban':
            return self._parse_ban(parts[1:], user_input)
        
        elif action == 'pick':
            return self._parse_pick(parts[1:], user_input)
        
        elif action == 'search':
            return self._parse_search(parts[1:])
        
        elif action == 'info':
            return self._parse_info(parts[1:])
        
        elif action in ['refresh', 'help', 'quit']:
            return {'action': action, 'valid': True}
        
        else:
            return {
                'action': 'unknown',
                'valid': False,
                'error': f"Unknown command: '{action}'. Type 'help' for available commands."
            }
    
    def _parse_ban(self, args: List[str], original: str) -> Dict:
        """
        Parse ban command
        
        Format: ban <champion>
        Examples: 
          - ban yasuo
          - ban yas
          - b yasuo
        """
        if not args:
            return {
                'action': 'ban',
                'valid': False,
                'error': "Usage: ban <champion>  (e.g., 'ban yasuo')"
            }
        
        # Extract champion name (everything after 'ban')
        # Use original input to preserve case
        champion_query = original.split(maxsplit=1)[1].strip()
        
        # Clean and find champion
        champion_query = clean_champion_name(champion_query)
        champion_name = find_closest_champion(
            champion_query, 
            self.champion_db.get_all_champion_names(),
            threshold=config.FUZZY_MATCH_THRESHOLD
        )
        
        if not champion_name:
            # Try to provide suggestions
            matches = fuzzy_match(
                champion_query,
                self.champion_db.get_all_champion_names(),
                threshold=0.5
            )
            
            if matches:
                suggestions = ", ".join([m[0] for m in matches[:3]])
                error = f"Champion '{champion_query}' not found. Did you mean: {suggestions}?"
            else:
                error = f"Champion '{champion_query}' not found. Type 'search {champion_query}' to find similar champions."
            
            return {
                'action': 'ban',
                'valid': False,
                'error': error
            }
        
        # Get champion ID
        champion_id = self.champion_db.get_id(champion_name)
        
        return {
            'action': 'ban',
            'champion': champion_name,
            'champion_id': champion_id,
            'valid': True
        }
    
    def _parse_pick(self, args: List[str], original: str) -> Dict:
        """
        Parse pick command
        
        Format: pick <champion> [role]
        Examples:
          - pick malphite
          - pick malphite top
          - pick mal top
          - p mal
        """
        if not args:
            return {
                'action': 'pick',
                'valid': False,
                'error': "Usage: pick <champion> [role]  (e.g., 'pick malphite' or 'pick malphite top')"
            }
        
        # Extract champion name and optional role
        # Use original input to preserve case
        parts = original.split(maxsplit=1)[1].strip().split()
        
        champion_query = parts[0]
        role_query = parts[1].upper() if len(parts) > 1 else None
        
        # Clean and find champion
        champion_query = clean_champion_name(champion_query)
        champion_name = find_closest_champion(
            champion_query,
            self.champion_db.get_all_champion_names(),
            threshold=config.FUZZY_MATCH_THRESHOLD
        )
        
        if not champion_name:
            # Try to provide suggestions
            matches = fuzzy_match(
                champion_query,
                self.champion_db.get_all_champion_names(),
                threshold=0.5
            )
            
            if matches:
                suggestions = ", ".join([m[0] for m in matches[:3]])
                error = f"Champion '{champion_query}' not found. Did you mean: {suggestions}?"
            else:
                error = f"Champion '{champion_query}' not found. Type 'search {champion_query}' to find similar champions."
            
            return {
                'action': 'pick',
                'valid': False,
                'error': error
            }
        
        # Validate role if provided
        if role_query:
            if role_query not in config.ROLES:
                return {
                    'action': 'pick',
                    'champion': champion_name,
                    'valid': False,
                    'error': f"Invalid role: '{role_query}'. Valid roles: {', '.join(config.ROLES)}"
                }
            
            role = role_query
        else:
            # Role will be auto-detected by game controller
            role = None
        
        # Get champion ID
        champion_id = self.champion_db.get_id(champion_name)
        
        return {
            'action': 'pick',
            'champion': champion_name,
            'champion_id': champion_id,
            'role': role,
            'valid': True
        }
    
    def _parse_search(self, args: List[str]) -> Dict:
        """
        Parse search command
        
        Format: search <query>
        Examples:
          - search lee
          - search ori
          - s yas
        """
        if not args:
            return {
                'action': 'search',
                'valid': False,
                'error': "Usage: search <name>  (e.g., 'search lee')"
            }
        
        query = " ".join(args)
        
        return {
            'action': 'search',
            'query': query,
            'valid': True
        }
    
    def _parse_info(self, args: List[str]) -> Dict:
        """
        Parse info command
        
        Format: info <champion>
        Examples:
          - info yasuo
          - info lee sin
          - i yas
        """
        if not args:
            return {
                'action': 'info',
                'valid': False,
                'error': "Usage: info <champion>  (e.g., 'info yasuo')"
            }
        
        champion_query = " ".join(args)
        champion_query = clean_champion_name(champion_query)
        
        # Find champion
        champion_name = find_closest_champion(
            champion_query,
            self.champion_db.get_all_champion_names(),
            threshold=config.FUZZY_MATCH_THRESHOLD
        )
        
        if not champion_name:
            return {
                'action': 'info',
                'valid': False,
                'error': f"Champion '{champion_query}' not found."
            }
        
        champion_id = self.champion_db.get_id(champion_name)
        
        return {
            'action': 'info',
            'champion': champion_name,
            'champion_id': champion_id,
            'valid': True
        }
    
    def validate_ban(self, champion_id: int, draft_state: DraftState) -> Tuple[bool, Optional[str]]:
        """
        Validate a ban command
        
        Args:
            champion_id: Champion ID to ban
            draft_state: Current draft state
            
        Returns:
            (is_valid, error_message)
        """
        # Check if champion already banned or picked
        if not draft_state.is_champion_available(champion_id):
            champion_name = self.champion_db.get_name(champion_id)
            return False, f"{champion_name} is already banned or picked"
        
        # Check if ban phase is complete
        if draft_state.is_bans_complete():
            return False, "Ban phase is complete"
        
        return True, None
    
    def validate_pick(self, champion_id: int, role: Optional[str], 
                     draft_state: DraftState, side: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a pick command
        
        Args:
            champion_id: Champion ID to pick
            role: Role for this pick (None for auto-detect)
            draft_state: Current draft state
            side: Which side is picking
            
        Returns:
            (is_valid, error_message, suggested_role)
        """
        champion_name = self.champion_db.get_name(champion_id)
        
        # Check if bans complete
        if not draft_state.is_bans_complete():
            return False, "Cannot pick during ban phase", None
        
        # Check if champion available
        if not draft_state.is_champion_available(champion_id):
            return False, f"{champion_name} is already banned or picked", None
        
        # Get vacant roles
        vacant_roles = draft_state.get_vacant_roles(side)
        
        if not vacant_roles:
            return False, "All roles are filled", None
        
        # Auto-detect role if not provided
        if role is None:
            role = self.champion_db.get_best_role(champion_name, vacant_roles)
            
            if role is None:
                return False, f"{champion_name} has no available roles for vacant positions", None
        
        # Validate role
        if role not in vacant_roles:
            return False, f"Role {role} is already filled. Vacant roles: {', '.join(vacant_roles)}", None
        
        return True, None, role
    
    def search_champions(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search for champions matching query
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching champion names
        """
        matches = fuzzy_match(
            query,
            self.champion_db.get_all_champion_names(),
            threshold=0.4  # Lower threshold for search
        )
        
        return [name for name, score in matches[:max_results]]
    
    def get_command_history(self, n: int = 10) -> List[str]:
        """Get last n commands from history"""
        return self.command_history[-n:]


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test input handler"""
    
    print("="*70)
    print("INPUT HANDLER - TESTING")
    print("="*70)
    
    from src.core.draft_engine import DraftEngine
    
    # Initialize
    champion_db = ChampionDatabase()
    champion_db.load(verbose=False)
    
    handler = InputHandler(champion_db)
    engine = DraftEngine(user_side='blue')
    
    # Test commands
    test_commands = [
        "ban yasuo",
        "ban yas",
        "b orianna",
        "pick malphite",
        "pick malphite top",
        "pick mal",
        "p lee sin jungle",
        "search lee",
        "info yasuo",
        "refresh",
        "help",
        "quit",
        "invalid command",
        "ban",  # Missing argument
        "pick",  # Missing argument
    ]
    
    print("\n1. Testing Command Parsing:")
    print("-" * 70)
    
    for cmd in test_commands:
        print(f"\nInput: '{cmd}'")
        result = handler.parse_command(cmd)
        
        print(f"  Action: {result['action']}")
        print(f"  Valid: {result['valid']}")
        
        if 'champion' in result:
            print(f"  Champion: {result.get('champion')}")
        if 'role' in result:
            print(f"  Role: {result.get('role')}")
        if 'error' in result and not result['valid']:
            print(f"  Error: {result['error']}")
    
    # Test validation
    print("\n\n2. Testing Validation:")
    print("-" * 70)
    
    # Execute some bans
    bans = [157, 238, 61]  # Yasuo, Zed, Orianna
    for ban_id in bans:
        engine.execute_ban(ban_id)
        champion_name = champion_db.get_name(ban_id)
        print(f"Banned: {champion_name}")
    
    # Try to ban already banned champion
    print(f"\nTrying to ban Yasuo (already banned):")
    is_valid, error = handler.validate_ban(157, engine.state)
    print(f"  Valid: {is_valid}")
    print(f"  Error: {error}")
    
    # Complete bans
    more_bans = [18, 11, 201, 22, 64, 69, 110]
    for ban_id in more_bans:
        engine.execute_ban(ban_id)
    
    # Test pick validation
    print(f"\nTrying to pick Malphite:")
    is_valid, error, role = handler.validate_pick(54, None, engine.state, 'blue')
    print(f"  Valid: {is_valid}")
    print(f"  Suggested role: {role}")
    print(f"  Error: {error}")
    
    # Test search
    print("\n\n3. Testing Search:")
    print("-" * 70)
    
    search_queries = ["lee", "ori", "yas", "tf"]
    
    for query in search_queries:
        results = handler.search_champions(query, max_results=5)
        print(f"\nSearch: '{query}'")
        print(f"  Results: {', '.join(results)}")
    
    print("\n" + "="*70)
    print("✓ Input handler testing complete")
    print("="*70)
