from dp.phonemizer import Phonemizer
import re
from collections import defaultdict

class G2PConverter:
    def __init__(self, remove_stress=False, fallback=None, word_boundaries=False):
        """
        :param remove_stress: If True, remove stress markers from phonemes (e.g., 'AA1' -> 'AA').
        :param fallback: A callback for unknown words. Takes a word and returns a list of ARPABET phonemes.
        :param word_boundaries: If True, insert space markers between words in the output.
        """
        self.remove_stress = remove_stress
        self.fallback = fallback
        self.word_boundaries = word_boundaries
        self.dictionary = defaultdict(list)

    def load_dictionary(self, *file_paths):
        """
        Load one or more pronunciation dictionary files.
        """
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split("  ")  # Double-space separator
                    if len(parts) == 2:
                        word, phonemes = parts
                        self.dictionary[word.upper()].append(phonemes.split())

    def _normalize_phonemes(self, phonemes):
        if self.remove_stress:
            return [re.sub(r'\d', '', p) for p in phonemes]
        return phonemes

    def g2p(self, sentence):
        """
        Convert a sentence into a list of ARPABET phonemes.
        :return: List of phonemes, optionally including spaces between words.
        """
        result = []
        words = re.findall(r"\b[\w']+\b", sentence)
        for idx, word in enumerate(words):
            if word.upper() in self.dictionary:
                phonemes = self.dictionary[word.upper()][0]
            elif self.fallback:
                phonemes = self.fallback(word)
            else:
                raise ValueError(f"Word '{word}' not found in dictionary and no fallback provided.")
            result.extend(self._normalize_phonemes(phonemes))
            if self.word_boundaries and idx < len(words) - 1:
                result.append(' ')  # Mark word boundary
        return result

class DefaultG2PFallback:
    def __init__(self):
        self.fallback_phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_forward.pt')   
    
    def __call__(self, word):
        if all(c.isupper() for c in word):
            # Assume that a word with all caps is an acronym
            return re.findall(r"\[(\w+)\]", self.fallback_phonemizer(word, lang='en_us', expand_acronyms=True))
        else:
            return re.findall(r"\[(\w+)\]", self.fallback_phonemizer(word, lang='en_us', expand_acronyms=False))

if __name__ == "__main__":
    g2p = G2PConverter(remove_stress=True, word_boundaries=True, fallback=DefaultG2PFallback())
    g2p.load_dictionary("g2p/cmudict.clean", "g2p/new_horsewords.clean")
    print(g2p.g2p("I'm down. Everypony's a pony down in Tenochtitlan. Ay ay! Welcome to the DMV. The SCC FCC. florble gorble."))
