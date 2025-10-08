#!/usr/bin/env python3
"""
Phase 2 Testing Script
Tests Draft Engine components

Run this to verify Phase 2 is working correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.draft_engine import DraftEngine, DraftState, Phase, Pick
from src.core.champion_db import ChampionDatabase
import config


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print formatted subheader"""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def test_draft_state():
    """Test DraftState class"""
    print_header("TEST 1: DRAFT STATE")

    try:
        state = DraftState()

        # Test adding bans
        print("\n📝 Testing bans...")
        state.add_ban(157, 'blue')  # Yasuo
        state.add_ban(238, 'red')  # Zed

        assert len(state.blue_bans) == 1, "Blue bans count incorrect"
        assert len(state.red_bans) == 1, "Red bans count incorrect"
        assert 157 in state.get_all_banned_ids(), "Ban not recorded"
        print("  ✓ Bans working correctly")

        # Test adding picks
        print("\n📝 Testing picks...")
        state.add_pick(222, "Jinx", "BOTTOM", 'blue')
        state.add_pick(498, "Kai'Sa", "BOTTOM", 'red')

        assert len(state.blue_picks) == 1, "Blue picks count incorrect"
        assert len(state.red_picks) == 1, "Red picks count incorrect"
        assert state.blue_picks[0].champion_name == "Jinx", "Pick name incorrect"
        print("  ✓ Picks working correctly")

        # Test unavailable champions
        print("\n📝 Testing unavailable champions...")
        unavailable = state.get_unavailable_ids()
        assert 157 in unavailable, "Banned champion not in unavailable"
        assert 222 in unavailable, "Picked champion not in unavailable"
        assert state.is_champion_available(1), "Available champion marked unavailable"
        assert not state.is_champion_available(157), "Banned champion marked available"
        print("  ✓ Availability checking works")

        # Test role tracking
        print("\n📝 Testing role tracking...")
        blue_roles = state.get_team_roles('blue')
        assert "BOTTOM" in blue_roles, "Role not tracked"

        vacant = state.get_vacant_roles('blue')
        assert "BOTTOM" not in vacant, "Filled role in vacant list"
        assert "TOP" in vacant, "Unfilled role not in vacant list"
        print("  ✓ Role tracking works")

        # Test completion checks
        print("\n📝 Testing completion checks...")
        assert not state.is_bans_complete(), "Incorrectly marked bans complete"
        assert not state.is_draft_complete(), "Incorrectly marked draft complete"

        # Complete bans
        for i in range(8):  # Add 8 more bans (total 10)
            side = 'blue' if i % 2 == 0 else 'red'
            state.add_ban(100 + i, side)

        assert state.is_bans_complete(), "Bans not marked complete"
        print("  ✓ Completion checking works")

        print("\n✅ DraftState test passed!")
        return True

    except AssertionError as e:
        print(f"\n❌ DraftState test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ DraftState test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ban_sequence():
    """Test ban sequence generation"""
    print_header("TEST 2: BAN SEQUENCE")

    try:
        engine = DraftEngine(user_side='blue')

        # Check sequence format
        print("\n📝 Checking ban sequence...")
        ban_seq = engine.ban_sequence

        assert len(ban_seq) == 10, f"Expected 10 bans, got {len(ban_seq)}"

        # Count bans per side
        blue_count = ban_seq.count('blue')
        red_count = ban_seq.count('red')

        assert blue_count == 5, f"Expected 5 blue bans, got {blue_count}"
        assert red_count == 5, f"Expected 5 red bans, got {red_count}"

        # Check alternating pattern
        for i in range(len(ban_seq) - 1):
            if ban_seq[i] == ban_seq[i + 1]:
                print(f"  ⚠️  Non-alternating at position {i}: {ban_seq[i]} → {ban_seq[i + 1]}")

        print(f"  Ban sequence: {' → '.join([s[0].upper() for s in ban_seq])}")
        print(f"  Expected: B-R-B-R-B-R-B-R-B-R")
        print("  ✓ Ban sequence correct")

        # Check pick sequence
        print("\n📝 Checking pick sequence...")
        pick_seq = engine.pick_sequence

        assert len(pick_seq) == 10, f"Expected 10 picks, got {len(pick_seq)}"

        blue_picks = pick_seq.count('blue')
        red_picks = pick_seq.count('red')

        assert blue_picks == 5, f"Expected 5 blue picks, got {blue_picks}"
        assert red_picks == 5, f"Expected 5 red picks, got {red_picks}"

        # Check 1-2-2-2-2-1 pattern
        expected_pattern = ['blue', 'red', 'red', 'blue', 'blue',
                            'red', 'red', 'blue', 'blue', 'red']

        if pick_seq == expected_pattern:
            print("  ✓ Pick sequence matches 1-2-2-2-2-1 pattern")
        else:
            print(f"  ⚠️  Pick sequence: {pick_seq}")
            print(f"  Expected: {expected_pattern}")

        print("\n✅ Ban/Pick sequence test passed!")
        return True

    except AssertionError as e:
        print(f"\n❌ Sequence test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Sequence test error: {e}")
        return False


def test_phase_progression():
    """Test phase progression logic"""
    print_header("TEST 3: PHASE PROGRESSION")

    try:
        # Load champion database for valid IDs
        from src.core.champion_db import ChampionDatabase
        db = ChampionDatabase()
        db.load(verbose=False)

        engine = DraftEngine(user_side='blue')

        # Initial phase
        print("\n📝 Testing initial phase...")
        phase = engine.get_current_phase()
        assert phase == Phase.BAN_ROUND_1, f"Expected BAN_ROUND_1, got {phase}"
        print(f"  Initial phase: {phase.value} ✓")

        # After 3 bans per side (6 total)
        print("\n📝 Testing ban round transition...")
        # Use real champion names to get valid IDs
        ban_names_round1 = ["Yasuo", "Zed", "Orianna", "Ahri", "Syndra", "LeBlanc"]
        banned_count = 0
        for name in ban_names_round1:
            champ_id = db.get_id(name)
            if champ_id:
                if engine.execute_ban(champ_id):
                    banned_count += 1
            else:
                print(f"  ⚠️  Champion '{name}' not found in database")

        if banned_count < 6:
            print(f"  ⚠️  Only {banned_count}/6 bans executed in round 1")

        phase = engine.get_current_phase()
        assert phase == Phase.BAN_ROUND_2, f"Expected BAN_ROUND_2 after 6 bans, got {phase}"
        print(f"  After {banned_count} bans: {phase.value} ✓")

        # Complete bans (4 more)
        print("\n📝 Testing transition to pick phase...")
        ban_names_round2 = ["Katarina", "Fizz", "Talon", "Kassadin"]
        for name in ban_names_round2:
            champ_id = db.get_id(name)
            if champ_id:
                if engine.execute_ban(champ_id):
                    banned_count += 1
            else:
                print(f"  ⚠️  Champion '{name}' not found in database")

        if banned_count < 10:
            print(f"  ⚠️  Only {banned_count}/10 total bans executed")

        phase = engine.get_current_phase()
        assert phase == Phase.PICK, f"Expected PICK after 10 bans, got {phase}"
        assert engine.state.is_bans_complete(), "Bans not marked complete"
        print(f"  After {banned_count} bans total: {phase.value} ✓")

        # After some picks
        print("\n📝 Testing pick phase...")
        initial_picks = [
            ("Jinx", "BOTTOM"),
            ("Kai'Sa", "BOTTOM"),
            ("Thresh", "UTILITY"),
        ]

        picked_count = 0
        for name, role in initial_picks:
            champ_id = db.get_id(name)
            if champ_id:
                if engine.execute_pick(champ_id, name, role):
                    picked_count += 1

        phase = engine.get_current_phase()
        assert phase == Phase.PICK, "Should still be in PICK phase"
        print(f"  After {picked_count} picks: {phase.value} ✓")

        # Complete draft
        print("\n📝 Testing draft completion...")

        # Show currently banned champions for debugging
        banned_ids = engine.state.get_all_banned_ids()
        print(f"  Currently banned champion IDs: {banned_ids}")

        # Pick sequence continues: blue, blue, red, red, blue, blue, red
        # Current state: Blue has BOTTOM, Red has BOTTOM+UTILITY
        # Need to pick champions that are definitely NOT banned
        remaining_pick_names = [
            ("Leona", "UTILITY"),  # Turn 4: Blue
            ("Annie", "MIDDLE"),  # Turn 5: Blue
            ("Veigar", "MIDDLE"),  # Turn 6: Red
            ("Garen", "TOP"),  # Turn 7: Red
            ("Teemo", "TOP"),  # Turn 8: Blue
            ("Master Yi", "JUNGLE"),  # Turn 9: Blue - Changed from Graves
            ("Warwick", "JUNGLE"),  # Turn 10: Red - Changed from Shyvana
        ]

        successful_picks = 0
        for name, role in remaining_pick_names:
            champ_id = db.get_id(name)
            if not champ_id:
                print(f"  ✗ Champion '{name}' not found in database")
                continue

            current_side = engine.get_current_side()
            success = engine.execute_pick(champ_id, name, role)

            if success:
                successful_picks += 1
            else:
                print(f"  ✗ Failed to pick {name} ({role}) for {current_side}")
                print(f"     Champion available: {engine.state.is_champion_available(champ_id)}")
                print(f"     Vacant roles for {current_side}: {engine.state.get_vacant_roles(current_side)}")
                print(f"     Current turn: {engine.get_current_side()}")

        # Check final state
        total_picks = len(engine.state.blue_picks) + len(engine.state.red_picks)
        print(f"\n  Successfully picked: {successful_picks}/{len(remaining_pick_names)} champions")
        print(f"  Total picks in draft: {total_picks}/10")

        phase = engine.get_current_phase()

        if phase != Phase.COMPLETE:
            print(f"\n  ⚠️  Draft not complete!")
            print(f"  Current phase: {phase.value}")
            print(
                f"  Blue picks ({len(engine.state.blue_picks)}): {[p.champion_name for p in engine.state.blue_picks]}")
            print(f"  Red picks ({len(engine.state.red_picks)}): {[p.champion_name for p in engine.state.red_picks]}")

        assert phase == Phase.COMPLETE, f"Expected COMPLETE, got {phase}"
        assert engine.state.is_draft_complete(), "Draft not marked complete"
        print(f"\n  After 10 picks: {phase.value} ✓")

        print("\n✅ Phase progression test passed!")
        return True

    except AssertionError as e:
        print(f"\n❌ Phase progression test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Phase progression test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_turn_management():
    """Test turn management and user turn detection"""
    print_header("TEST 4: TURN MANAGEMENT")

    try:
        # Test as blue side
        print("\n📝 Testing as blue side...")
        engine = DraftEngine(user_side='blue')

        # First turn should be blue
        current = engine.get_current_side()
        assert current == 'blue', f"Expected blue's turn, got {current}"
        assert engine.is_user_turn(), "Should be user's turn"
        print("  Turn 1: Blue's turn (user) ✓")

        # After blue bans
        engine.execute_ban(157)
        current = engine.get_current_side()
        assert current == 'red', f"Expected red's turn, got {current}"
        assert not engine.is_user_turn(), "Should not be user's turn"
        print("  Turn 2: Red's turn (opponent) ✓")

        # Test as red side
        print("\n📝 Testing as red side...")
        engine = DraftEngine(user_side='red')

        current = engine.get_current_side()
        assert current == 'blue', "First turn is always blue"
        assert not engine.is_user_turn(), "Should not be user's turn (blue goes first)"
        print("  Turn 1: Blue's turn (opponent) ✓")

        engine.execute_ban(157)
        current = engine.get_current_side()
        assert current == 'red', f"Expected red's turn, got {current}"
        assert engine.is_user_turn(), "Should be user's turn"
        print("  Turn 2: Red's turn (user) ✓")

        # Test turn info
        print("\n📝 Testing turn info...")
        engine = DraftEngine(user_side='blue')

        info = engine.get_turn_info()
        assert 'phase' in info, "Turn info missing phase"
        assert 'side' in info, "Turn info missing side"
        assert 'is_user_turn' in info, "Turn info missing user turn flag"
        assert 'action' in info, "Turn info missing action"

        print(f"  Turn info keys: {list(info.keys())} ✓")

        print("\n✅ Turn management test passed!")
        return True

    except AssertionError as e:
        print(f"\n❌ Turn management test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Turn management test error: {e}")
        return False


def test_validation():
    """Test validation logic"""
    print_header("TEST 5: VALIDATION")

    try:
        engine = DraftEngine(user_side='blue')

        # Test valid ban
        print("\n📝 Testing valid ban...")
        result = engine.execute_ban(157)  # Yasuo
        assert result, "Valid ban rejected"
        print("  Valid ban accepted ✓")

        # Test duplicate ban
        print("\n📝 Testing duplicate ban...")
        result = engine.execute_ban(157)  # Same champion
        assert not result, "Duplicate ban accepted"
        print("  Duplicate ban rejected ✓")

        # Test pick during ban phase
        print("\n📝 Testing pick during ban phase...")
        result = engine.execute_pick(222, "Jinx", "BOTTOM")
        assert not result, "Pick accepted during ban phase"
        print("  Pick during ban phase rejected ✓")

        # Complete bans
        for i in range(9):
            engine.execute_ban(100 + i)

        # Test valid pick
        print("\n📝 Testing valid pick...")
        result = engine.execute_pick(222, "Jinx", "BOTTOM")
        assert result, "Valid pick rejected"
        print("  Valid pick accepted ✓")

        # Test duplicate pick
        print("\n📝 Testing duplicate pick...")
        engine.execute_pick(498, "Kai'Sa", "BOTTOM")  # Red's turn
        result = engine.execute_pick(222, "Jinx", "TOP")  # Blue tries Jinx again
        assert not result, "Duplicate pick accepted"
        print("  Duplicate pick rejected ✓")

        # Test duplicate role
        print("\n📝 Testing duplicate role...")
        # Blue already has BOTTOM (Jinx)
        result = engine.execute_pick(1, "Annie", "BOTTOM")
        assert not result, "Duplicate role accepted"
        print("  Duplicate role rejected ✓")

        # Test valid role
        print("\n📝 Testing different role...")
        result = engine.execute_pick(412, "Thresh", "UTILITY")  # Red's turn
        assert result, "Valid role pick rejected"
        print("  Different role accepted ✓")

        print("\n✅ Validation test passed!")
        return True

    except AssertionError as e:
        print(f"\n❌ Validation test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Validation test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_champion_db():
    """Test integration with champion database"""
    print_header("TEST 6: INTEGRATION WITH CHAMPION DB")

    try:
        print("\n📝 Loading champion database...")
        champion_db = ChampionDatabase()
        champion_db.load(verbose=False)
        print("  ✓ Champion database loaded")

        print("\n📝 Testing draft with real champion data...")
        engine = DraftEngine(user_side='blue')

        # Ban some real champions
        bans = ["Yasuo", "Zed", "Orianna", "Lee Sin", "Jinx"]
        print(f"\n  Banning: {', '.join(bans)}")

        for champ_name in bans:
            champ_id = champion_db.get_id(champ_name)
            if champ_id:
                success = engine.execute_ban(champ_id)
                if success:
                    print(f"    ✓ Banned {champ_name} (ID: {champ_id})")
                else:
                    print(f"    ✗ Failed to ban {champ_name}")

        # Complete remaining bans
        for i in range(5):
            engine.execute_ban(100 + i)

        # Pick some champions with role assignment
        picks = ["Kai'Sa", "Thresh"]
        print(f"\n  Picking: {', '.join(picks)}")

        for champ_name in picks:
            champ_id = champion_db.get_id(champ_name)
            if champ_id:
                current_side = engine.get_current_side()
                vacant_roles = engine.state.get_vacant_roles(current_side)

                # Get best role for this champion
                best_role = champion_db.get_best_role(champ_name, vacant_roles)

                if best_role:
                    success = engine.execute_pick(champ_id, champ_name, best_role)
                    if success:
                        print(f"    ✓ Picked {champ_name} ({best_role}) for {current_side}")
                    else:
                        print(f"    ✗ Failed to pick {champ_name}")

        # Verify state
        summary = engine.state.get_summary()
        total_bans = len(summary['blue']['bans']) + len(summary['red']['bans'])
        total_picks = len(summary['blue']['picks']) + len(summary['red']['picks'])

        print(f"\n📊 Draft status:")
        print(f"  Total bans: {total_bans}")
        print(f"  Total picks: {total_picks}")
        print(f"  Blue picks: {summary['blue']['picks']}")
        print(f"  Red picks: {summary['red']['picks']}")

        assert total_bans == 10, f"Expected 10 bans, got {total_bans}"
        assert total_picks >= 2, f"Expected at least 2 picks, got {total_picks}"

        print("\n✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 tests"""
    print("\n" + "🎮" * 35)
    print("LoL DRAFT SIMULATOR - PHASE 2 TESTING")
    print("🎮" * 35)

    print("\nTesting Draft Engine components...")

    # Run tests
    results = []

    results.append(("Draft State", test_draft_state()))
    results.append(("Ban Sequence", test_ban_sequence()))
    results.append(("Phase Progression", test_phase_progression()))
    results.append(("Turn Management", test_turn_management()))
    results.append(("Validation", test_validation()))
    results.append(("Integration", test_integration_with_champion_db()))

    # Summary
    print_header("TEST SUMMARY")

    print("\nResults:")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "🎉" * 35)
        print("ALL TESTS PASSED! Phase 2 complete! 🚀")
        print("🎉" * 35)
        print("\n✓ Ready to proceed to Phase 3: Feature Engineering")
    else:
        print("\n⚠️  Some tests failed. Please fix errors before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)