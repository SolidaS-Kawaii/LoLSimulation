"""
Draft Engine
Manages draft state, turn progression, and validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config


class Phase(Enum):
    """Draft phases"""
    BAN_ROUND_1 = "ban_round_1"
    BAN_ROUND_2 = "ban_round_2"
    PICK = "pick"
    COMPLETE = "complete"


@dataclass
class Pick:
    """Represents a champion pick"""
    champion_id: int
    champion_name: str
    role: str
    
    def __str__(self):
        return f"{self.champion_name} ({self.role})"


@dataclass
class DraftState:
    """
    Current state of a draft
    
    Tracks all picks, bans, and current phase/turn information
    """
    # Bans (champion IDs)
    blue_bans: List[int] = field(default_factory=list)
    red_bans: List[int] = field(default_factory=list)
    
    # Picks (Pick objects with role information)
    blue_picks: List[Pick] = field(default_factory=list)
    red_picks: List[Pick] = field(default_factory=list)
    
    # Current state
    phase: Phase = Phase.BAN_ROUND_1
    current_side: str = 'blue'
    
    def add_ban(self, champion_id: int, side: str):
        """
        Add a ban for the specified side
        
        Args:
            champion_id: Champion ID to ban
            side: 'blue' or 'red'
        """
        if side == 'blue':
            self.blue_bans.append(champion_id)
        else:
            self.red_bans.append(champion_id)
    
    def add_pick(self, champion_id: int, champion_name: str, 
                 role: str, side: str):
        """
        Add a pick for the specified side
        
        Args:
            champion_id: Champion ID
            champion_name: Champion name
            role: Role (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
            side: 'blue' or 'red'
        """
        pick = Pick(champion_id, champion_name, role)
        
        if side == 'blue':
            self.blue_picks.append(pick)
        else:
            self.red_picks.append(pick)
    
    def get_all_banned_ids(self) -> List[int]:
        """Get all banned champion IDs"""
        return self.blue_bans + self.red_bans
    
    def get_all_picked_ids(self) -> List[int]:
        """Get all picked champion IDs"""
        blue_ids = [p.champion_id for p in self.blue_picks]
        red_ids = [p.champion_id for p in self.red_picks]
        return blue_ids + red_ids
    
    def get_unavailable_ids(self) -> List[int]:
        """
        Get all unavailable champion IDs (banned + picked)
        
        Returns:
            List of champion IDs that cannot be picked
        """
        return self.get_all_banned_ids() + self.get_all_picked_ids()
    
    def get_team_roles(self, side: str) -> List[str]:
        """
        Get roles already picked for a team
        
        Args:
            side: 'blue' or 'red'
            
        Returns:
            List of role strings
        """
        picks = self.blue_picks if side == 'blue' else self.red_picks
        return [pick.role for pick in picks]
    
    def get_vacant_roles(self, side: str) -> List[str]:
        """
        Get roles not yet filled for a team
        
        Args:
            side: 'blue' or 'red'
            
        Returns:
            List of vacant role strings
        """
        filled_roles = self.get_team_roles(side)
        return [role for role in config.ROLES if role not in filled_roles]
    
    def is_champion_available(self, champion_id: int) -> bool:
        """
        Check if champion is available (not banned/picked)
        
        Args:
            champion_id: Champion ID to check
            
        Returns:
            True if available, False otherwise
        """
        return champion_id not in self.get_unavailable_ids()
    
    def is_bans_complete(self) -> bool:
        """Check if ban phase is complete (5 bans per team)"""
        return len(self.blue_bans) == 5 and len(self.red_bans) == 5
    
    def is_draft_complete(self) -> bool:
        """Check if draft is finished (5 picks per team)"""
        return len(self.blue_picks) == 5 and len(self.red_picks) == 5
    
    def get_summary(self) -> Dict:
        """
        Get summary of current draft state
        
        Returns:
            Dictionary with comprehensive draft information
        """
        return {
            'phase': self.phase.value,
            'current_side': self.current_side,
            'blue': {
                'bans': self.blue_bans,
                'picks': [(p.champion_name, p.role) for p in self.blue_picks],
                'vacant_roles': self.get_vacant_roles('blue')
            },
            'red': {
                'bans': self.red_bans,
                'picks': [(p.champion_name, p.role) for p in self.red_picks],
                'vacant_roles': self.get_vacant_roles('red')
            },
            'complete': self.is_draft_complete()
        }


class DraftEngine:
    """
    Draft Engine - Manages draft flow and turn progression
    
    Handles:
    - Ban phase (Round 1: 3-3, Round 2: 2-2)
    - Pick phase (1-2-2-2-2-1)
    - Turn progression
    - State validation
    """
    
    def __init__(self, user_side: str = 'blue'):
        """
        Initialize draft engine
        
        Args:
            user_side: Which side user is playing ('blue' or 'red')
        """
        self.state = DraftState()
        self.user_side = user_side.lower()
        
        if self.user_side not in ['blue', 'red']:
            raise ValueError(f"Invalid side: {user_side}. Must be 'blue' or 'red'")
        
        # Build sequences
        self.ban_sequence = self._build_ban_sequence()
        self.pick_sequence = config.PICK_PHASE_SEQUENCE.copy()
        
        # Track current position in sequences
        self.ban_index = 0
        self.pick_index = 0
    
    def _build_ban_sequence(self) -> List[str]:
        """
        Build ban sequence: B-R-B-R-B-R (round 1), B-R-B-R (round 2)
        Total: 10 bans (5 per team)
        
        Returns:
            List of sides in ban order
        """
        sequence = []
        
        # Round 1: 3 bans each (alternating)
        for _ in range(config.BAN_PHASE_CONFIG['round1']['blue']):
            sequence.extend(['blue', 'red'])
        
        # Round 2: 2 bans each (alternating)
        for _ in range(config.BAN_PHASE_CONFIG['round2']['blue']):
            sequence.extend(['blue', 'red'])
        
        return sequence
    
    def get_current_phase(self) -> Phase:
        """
        Get current draft phase
        
        Returns:
            Phase enum value
        """
        if self.state.is_draft_complete():
            return Phase.COMPLETE
        elif not self.state.is_bans_complete():
            # Determine which ban round
            total_bans = len(self.state.blue_bans) + len(self.state.red_bans)
            if total_bans < 6:  # First 6 bans (3 per team)
                return Phase.BAN_ROUND_1
            else:
                return Phase.BAN_ROUND_2
        else:
            return Phase.PICK
    
    def get_current_side(self) -> Optional[str]:
        """
        Get which side's turn it is
        
        Returns:
            'blue', 'red', or None if draft complete
        """
        if self.state.is_draft_complete():
            return None
        
        phase = self.get_current_phase()
        
        if phase in [Phase.BAN_ROUND_1, Phase.BAN_ROUND_2]:
            if self.ban_index < len(self.ban_sequence):
                return self.ban_sequence[self.ban_index]
        else:  # PICK phase
            total_picks = len(self.state.blue_picks) + len(self.state.red_picks)
            if total_picks < len(self.pick_sequence):
                return self.pick_sequence[total_picks]
        
        return None
    
    def is_user_turn(self) -> bool:
        """
        Check if it's user's turn
        
        Returns:
            True if current turn belongs to user
        """
        current = self.get_current_side()
        return current == self.user_side if current else False
    
    def execute_ban(self, champion_id: int, side: Optional[str] = None) -> bool:
        """
        Execute a ban
        
        Args:
            champion_id: Champion ID to ban
            side: Which side is banning (optional, uses current turn if None)
            
        Returns:
            True if ban successful, False otherwise
        """
        # Determine side
        if side is None:
            side = self.get_current_side()
        
        if not side:
            return False
        
        # Validate
        if not self._validate_ban(champion_id, side):
            return False
        
        # Execute
        self.state.add_ban(champion_id, side)
        self.ban_index += 1
        
        # Update phase
        self.state.phase = self.get_current_phase()
        self.state.current_side = self.get_current_side() or side
        
        return True
    
    def execute_pick(self, champion_id: int, champion_name: str, 
                     role: str, side: Optional[str] = None) -> bool:
        """
        Execute a pick
        
        Args:
            champion_id: Champion ID
            champion_name: Champion name
            role: Role for this champion
            side: Which side is picking (optional, uses current turn if None)
            
        Returns:
            True if pick successful, False otherwise
        """
        # Determine side
        if side is None:
            side = self.get_current_side()
        
        if not side:
            return False
        
        # Validate
        if not self._validate_pick(champion_id, role, side):
            return False
        
        # Execute
        self.state.add_pick(champion_id, champion_name, role, side)
        
        # Update phase
        self.state.phase = self.get_current_phase()
        self.state.current_side = self.get_current_side() or side
        
        return True
    
    def _validate_ban(self, champion_id: int, side: str) -> bool:
        """
        Validate a ban
        
        Args:
            champion_id: Champion ID to validate
            side: Side attempting to ban
            
        Returns:
            True if valid, False otherwise
        """
        # Check if it's this side's turn
        current = self.get_current_side()
        if current != side:
            return False
        
        # Check if ban phase is complete
        if self.state.is_bans_complete():
            return False
        
        # Check if champion already banned/picked
        if not self.state.is_champion_available(champion_id):
            return False
        
        return True
    
    def _validate_pick(self, champion_id: int, role: str, side: str) -> bool:
        """
        Validate a pick
        
        Args:
            champion_id: Champion ID to validate
            role: Role for this champion
            side: Side attempting to pick
            
        Returns:
            True if valid, False otherwise
        """
        # Check if it's this side's turn
        current = self.get_current_side()
        if current != side:
            return False
        
        # Check if still in ban phase
        if not self.state.is_bans_complete():
            return False
        
        # Check if pick phase is complete
        if self.state.is_draft_complete():
            return False
        
        # Check if champion available
        if not self.state.is_champion_available(champion_id):
            return False
        
        # Check if role is vacant
        vacant_roles = self.state.get_vacant_roles(side)
        if role not in vacant_roles:
            return False
        
        return True
    
    def get_turn_info(self) -> Dict:
        """
        Get information about current turn
        
        Returns:
            Dictionary with turn information
        """
        phase = self.get_current_phase()
        side = self.get_current_side()
        
        info = {
            'phase': phase.value if phase else 'complete',
            'side': side,
            'is_user_turn': self.is_user_turn(),
            'user_side': self.user_side
        }
        
        if phase in [Phase.BAN_ROUND_1, Phase.BAN_ROUND_2]:
            # Ban phase info
            blue_bans_left = 5 - len(self.state.blue_bans)
            red_bans_left = 5 - len(self.state.red_bans)
            
            info.update({
                'action': 'ban',
                'blue_bans_remaining': blue_bans_left,
                'red_bans_remaining': red_bans_left,
                'total_bans': len(self.state.blue_bans) + len(self.state.red_bans)
            })
        elif phase == Phase.PICK:
            # Pick phase info
            blue_picks_left = 5 - len(self.state.blue_picks)
            red_picks_left = 5 - len(self.state.red_picks)
            
            info.update({
                'action': 'pick',
                'blue_picks_remaining': blue_picks_left,
                'red_picks_remaining': red_picks_left,
                'total_picks': len(self.state.blue_picks) + len(self.state.red_picks),
                'vacant_roles': self.state.get_vacant_roles(side) if side else []
            })
        
        return info
    
    def reset(self):
        """Reset draft to initial state"""
        self.state = DraftState()
        self.ban_index = 0
        self.pick_index = 0


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test draft engine"""
    
    print("="*70)
    print("DRAFT ENGINE - TESTING")
    print("="*70)
    
    # Test 1: Ban sequence
    print("\n1. Testing Ban Sequence:")
    print("-"*70)
    
    engine = DraftEngine(user_side='blue')
    print(f"Ban sequence: {' → '.join(engine.ban_sequence)}")
    print(f"Expected: B-R-B-R-B-R-B-R-B-R (10 bans total)")
    
    # Test 2: Phase progression
    print("\n2. Testing Phase Progression:")
    print("-"*70)
    
    # Simulate some bans
    test_bans = [157, 238, 61, 18, 11, 201]  # Sample champion IDs
    
    for i, champ_id in enumerate(test_bans):
        current_side = engine.get_current_side()
        phase = engine.get_current_phase()
        
        print(f"\nTurn {i+1}: {phase.value} - {current_side.upper()}'s turn")
        success = engine.execute_ban(champ_id)
        print(f"  Ban champion {champ_id}: {'✓' if success else '✗'}")
        
        if i == 5:  # After 6 bans (round 1 complete)
            print(f"  Phase: {engine.get_current_phase().value}")
    
    # Test 3: Turn info
    print("\n3. Testing Turn Information:")
    print("-"*70)
    
    turn_info = engine.get_turn_info()
    print(f"Current phase: {turn_info['phase']}")
    print(f"Current side: {turn_info['side']}")
    print(f"Is user turn: {turn_info['is_user_turn']}")
    print(f"Action: {turn_info.get('action', 'N/A')}")
    
    # Test 4: Validation
    print("\n4. Testing Validation:")
    print("-"*70)
    
    # Try to ban already banned champion
    already_banned = test_bans[0]
    print(f"\nTrying to ban already banned champion ({already_banned}):")
    result = engine.execute_ban(already_banned)
    print(f"  Result: {'✗ Correctly rejected' if not result else '✓ Error - should reject'}")
    
    # Try to pick before bans complete
    print(f"\nTrying to pick before bans complete:")
    result = engine.execute_pick(1, "Annie", "MIDDLE")
    print(f"  Result: {'✗ Correctly rejected' if not result else '✓ Error - should reject'}")
    
    # Test 5: Complete draft simulation
    print("\n5. Testing Complete Draft:")
    print("-"*70)
    
    engine.reset()
    
    # Complete all bans
    ban_ids = [157, 238, 61, 18, 11, 201, 22, 64, 69, 110]
    for champ_id in ban_ids:
        engine.execute_ban(champ_id)
    
    print(f"Bans complete: {engine.state.is_bans_complete()}")
    print(f"Current phase: {engine.get_current_phase().value}")
    
    # Do some picks
    picks = [
        (222, "Jinx", "BOTTOM", "blue"),
        (498, "Kai'Sa", "BOTTOM", "red"),
        (412, "Thresh", "UTILITY", "red"),
    ]
    
    for champ_id, name, role, side in picks:
        success = engine.execute_pick(champ_id, name, role, side)
        print(f"Pick {name} ({role}) for {side}: {'✓' if success else '✗'}")
    
    # Test 6: Draft summary
    print("\n6. Draft Summary:")
    print("-"*70)
    
    summary = engine.state.get_summary()
    print(f"\nBlue team:")
    print(f"  Bans: {len(summary['blue']['bans'])}")
    print(f"  Picks: {summary['blue']['picks']}")
    print(f"  Vacant roles: {summary['blue']['vacant_roles']}")
    
    print(f"\nRed team:")
    print(f"  Bans: {len(summary['red']['bans'])}")
    print(f"  Picks: {summary['red']['picks']}")
    print(f"  Vacant roles: {summary['red']['vacant_roles']}")
    
    print("\n" + "="*70)
    print("✓ Draft engine testing complete")
    print("="*70)
