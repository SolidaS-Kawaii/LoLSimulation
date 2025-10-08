"""
Text Processing Utilities
Functions for fuzzy matching and text suggestions
"""

from difflib import SequenceMatcher
from typing import List, Tuple, Optional


def fuzzy_match(query: str, candidates: List[str], 
                threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find similar strings using fuzzy matching
    
    Uses SequenceMatcher for similarity scoring with special handling
    for substring matches.
    
    Args:
        query: String to match
        candidates: List of candidate strings to search
        threshold: Minimum similarity score (0-1) to include in results
        
    Returns:
        List of (candidate, score) tuples, sorted by score descending
        
    Examples:
        >>> fuzzy_match("yasuo", ["Yasuo", "Yone", "Aatrox"])
        [('Yasuo', 0.95)]
        
        >>> fuzzy_match("ori", ["Orianna", "Ornn", "Pyke"])
        [('Orianna', 0.95), ('Ornn', 0.95)]
    """
    query_lower = query.lower().strip()
    matches = []
    
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        
        # Direct substring match gets bonus score
        if query_lower in candidate_lower:
            # Score based on how much of the candidate the query represents
            # "ori" in "Orianna" = 3/7 = 0.43 → bonus to 0.95
            base_score = len(query_lower) / len(candidate_lower)
            score = 0.90 + min(base_score * 0.1, 0.1)  # 0.90 to 1.0
        else:
            # Use sequence matcher for non-substring matches
            score = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        
        if score >= threshold:
            matches.append((candidate, score))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def find_closest_champion(query: str, champion_names: List[str], 
                          threshold: float = 0.8) -> Optional[str]:
    """
    Find the closest matching champion name
    
    Args:
        query: User input string
        champion_names: List of valid champion names
        threshold: Minimum similarity threshold (default 0.8)
        
    Returns:
        Closest matching champion name, or None if no match above threshold
        
    Examples:
        >>> find_closest_champion("yas", ["Yasuo", "Yone", "Zed"])
        'Yasuo'
        
        >>> find_closest_champion("xyz", ["Yasuo", "Yone", "Zed"])
        None
    """
    matches = fuzzy_match(query, champion_names, threshold)
    
    if matches:
        return matches[0][0]  # Return best match
    
    return None


def format_suggestions(query: str, champion_names: List[str], 
                      max_suggestions: int = 5) -> str:
    """
    Format suggestion text for user when champion not found
    
    Args:
        query: User's invalid input
        champion_names: List of valid champion names
        max_suggestions: Maximum number of suggestions to show (default 5)
        
    Returns:
        Formatted suggestion string
        
    Examples:
        >>> format_suggestions("yas", ["Yasuo", "Yone", "Zed"])
        'Did you mean: Yasuo, Yone?'
    """
    # Use lower threshold for suggestions (0.5 instead of 0.8)
    matches = fuzzy_match(query, champion_names, threshold=0.5)
    
    if not matches:
        return "❌ No similar champions found. Please check spelling or type 'help' to see all champions."
    
    # Get top N suggestions
    suggestions = [name for name, score in matches[:max_suggestions]]
    
    if len(suggestions) == 1:
        return f"❓ Did you mean: {suggestions[0]}?"
    else:
        suggestion_list = ", ".join(suggestions)
        return f"❓ Did you mean: {suggestion_list}?"


def clean_champion_name(name: str) -> str:
    """
    Clean and normalize champion name
    
    Args:
        name: Champion name to clean
        
    Returns:
        Cleaned champion name
        
    Examples:
        >>> clean_champion_name("  yasuo  ")
        'Yasuo'
        
        >>> clean_champion_name("LEE SIN")
        'Lee Sin'
    """
    # Strip whitespace and capitalize properly
    cleaned = name.strip()
    
    # Handle special cases with spaces (Lee Sin, Twisted Fate, etc.)
    words = cleaned.split()
    cleaned = " ".join(word.capitalize() for word in words)
    
    return cleaned


def get_fuzzy_threshold(query_length: int) -> float:
    """
    Get adaptive fuzzy match threshold based on query length
    
    Shorter queries need higher threshold to avoid false positives
    
    Args:
        query_length: Length of user query
        
    Returns:
        Recommended threshold (0.0 to 1.0)
        
    Examples:
        >>> get_fuzzy_threshold(2)  # Very short
        0.85
        
        >>> get_fuzzy_threshold(5)  # Medium
        0.8
        
        >>> get_fuzzy_threshold(10)  # Long
        0.7
    """
    if query_length <= 2:
        return 0.85  # Very strict for short queries
    elif query_length <= 4:
        return 0.8   # Standard threshold
    else:
        return 0.7   # More lenient for longer queries


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test text utilities"""
    
    print("="*70)
    print("TEXT UTILS - TESTING")
    print("="*70)
    
    # Sample champion names
    champions = [
        "Yasuo", "Yone", "Zed", "Orianna", "Ornn", 
        "Lee Sin", "Jinx", "Leona", "Kai'Sa", "Thresh",
        "Twisted Fate", "Annie", "Ahri", "Syndra"
    ]
    
    # Test cases
    test_queries = [
        ("yas", "Should match Yasuo"),
        ("ori", "Should match Orianna"),
        ("lee", "Should match Lee Sin"),
        ("tf", "Should match Twisted Fate"),
        ("kisa", "Should match Kai'Sa"),
        ("xyz", "Should find no match"),
        ("  YASUO  ", "Should normalize to Yasuo")
    ]
    
    print("\n1. Testing Fuzzy Matching:")
    print("-" * 70)
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}' ({description})")
        
        # Test fuzzy match
        matches = fuzzy_match(query, champions, threshold=0.6)
        if matches:
            print(f"  Matches found: {len(matches)}")
            for name, score in matches[:3]:
                print(f"    • {name:<20} (score: {score:.2f})")
        else:
            print(f"  No matches found")
        
        # Test closest champion
        closest = find_closest_champion(query, champions, threshold=0.7)
        if closest:
            print(f"  ✓ Best match: {closest}")
        else:
            print(f"  ✗ No match above threshold")
    
    print("\n\n2. Testing Suggestions:")
    print("-" * 70)
    
    for query, _ in test_queries[:5]:
        suggestions = format_suggestions(query, champions, max_suggestions=3)
        print(f"\nQuery: '{query}'")
        print(f"  {suggestions}")
    
    print("\n\n3. Testing Name Cleaning:")
    print("-" * 70)
    
    dirty_names = [
        "  yasuo  ",
        "LEE SIN",
        "twisted fate",
        "KAI'SA"
    ]
    
    for dirty in dirty_names:
        cleaned = clean_champion_name(dirty)
        print(f"'{dirty}' → '{cleaned}'")
    
    print("\n\n4. Testing Adaptive Threshold:")
    print("-" * 70)
    
    for length in [2, 3, 5, 8, 12]:
        threshold = get_fuzzy_threshold(length)
        print(f"Query length {length}: threshold = {threshold}")
    
    print("\n" + "="*70)
    print("✓ Text utils testing complete")
    print("="*70)
