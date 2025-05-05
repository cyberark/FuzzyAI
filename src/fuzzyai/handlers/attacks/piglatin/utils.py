import string

VOWELS = "aeiouAEIOU"

def english_to_pig_latin(sentence: str) -> str:
    def convert_to_pig_latin(word: str) -> str:
        if word[0] in VOWELS:
            return word + "yay"
        else:
            for i, letter in enumerate(word):
                if letter in VOWELS:
                    return word[i:] + word[:i] + "ay"
            return word + "ay"  # In case the word doesn't have any vowels
    words = sentence.split()
    pig_latin_words = [convert_to_pig_latin(word) for word in words]
    return " ".join(pig_latin_words)

def pig_latin_to_english(sentence: str) -> str:
    def convert_to_english(word: str) -> str:
        if word.endswith("yay"):
            return word[:-3]
        elif word.endswith("ay"):
            word = word[:-2]
            for i in range(len(word) - 1, -1, -1):
                if word[i] in VOWELS:
                    return word[i:] + word[:i]
        return word
    words = sentence.split()
    english_words = [convert_to_english(word) for word in words]
    return " ".join(english_words)

def is_pig_latin(sentence: str) -> bool:
    vowels = "aeiouAEIOU"
    punctuation = string.punctuation

    def check_word(word: str) -> bool:
        # Separate the word from its punctuation
        word_core = word.rstrip(punctuation)
        
        # Check if word_core ends with "yay" and starts with a vowel
        if word_core.endswith("yay") and word_core[:-3][0] in vowels:
            return True
        # Check if word_core ends with "ay" and has at least one vowel before "ay"
        elif word_core.endswith("ay"):
            word_core = word_core[:-2]
            for char in word_core:
                if char in vowels:
                    return True
        return False

    words = sentence.split()
    return all(check_word(word) for word in words)
