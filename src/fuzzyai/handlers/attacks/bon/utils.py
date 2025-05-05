import random
import string
from typing import Optional


def character_scrambling(text: str, scramble_prob: float = 0.6) -> str:
    def scramble_word(word: str) -> str:
        if len(word) > 3 and random.random() < scramble_prob:
            middle = list(word[1:-1])
            random.shuffle(middle)
            return word[0] + ''.join(middle) + word[-1]
        return word

    words = text.split()
    scrambled_words = [scramble_word(word) for word in words]
    return ' '.join(scrambled_words)

def random_capitalization(text: str, cap_prob: float = 0.6) -> str:
    return ''.join(
        char.upper() if random.random() < cap_prob else char.lower() for char in text
    )

def character_noising(text: str, noise_prob: float = 0.06, seed: Optional[int] = None) -> str:
    """
    Add random noise to text by slightly modifying characters while preserving case.
    
    Args:
        text: Input text to be noised
        noise_prob: Probability (0-1) of modifying each character
        seed: Optional random seed for reproducibility
    
    Returns:
        String with random character modifications
    
    Raises:
        ValueError: If noise_prob is not between 0 and 1
    """
    if not 0 <= noise_prob <= 1:
        raise ValueError("noise_prob must be between 0 and 1")
    
    if seed is not None:
        random.seed(seed)
    
    def alter_char(char: str) -> str:
        """Modify a single character while preserving case."""
        if not char.isalpha():
            return char
            
        # Determine character set and boundaries based on case
        if char.isupper():
            charset = string.ascii_uppercase
        else:
            charset = string.ascii_lowercase
            
        if random.random() >= noise_prob:
            return char
            
        # Get adjacent characters while wrapping around
        char_idx = charset.index(char)
        possible_chars = [
            charset[(char_idx - 1) % len(charset)],
            charset[(char_idx + 1) % len(charset)]
        ]
        
        return random.choice(possible_chars)
    
    return ''.join(alter_char(c) for c in text)