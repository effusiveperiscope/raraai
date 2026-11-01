import nltk
nltk.download('averaged_perceptron_tagger_eng')

from g2p_en import G2p
import re

g2p = G2p()

# g2p_en's heteronym ("homograph") table: lowercase word -> (pronunciation
# when the word's Penn tag starts with `pos1`, pronunciation otherwise,
# `pos1`).  e.g. READ -> (R IY1 D, R EH1 D, VBD).
HOMOGRAPHS = g2p.homograph2features

# Characters kept when reducing a token to its dictionary key (mirrors what
# g2p_en's own `__call__` sanitisation keeps).
_WORD_CHARS = re.compile(r"[^a-z']")
# Splits a token into (leading punctuation, word, trailing punctuation).
_TOKEN_PARTS = re.compile(r"^([^A-Za-z']*)(.*?)([^A-Za-z']*)$")


class HorsePhonemizer:
    """g2p_en + a hand-written ARPAbet dictionary, with heteronym support.

    The dictionary file has one ``WORD  PHONES`` entry per line (two spaces
    between the two fields, PHONES in ARPAbet with stress digits). Heteronyms
    that g2p_en gets wrong even in context can be pinned per part of speech
    with a POS-qualified key, e.g.::

        USE/VB  Y UW1 Z
        USE/NN  Y UW1 S
        LEAD/VB  L IY1 D

    The qualifier is matched exactly (``VB``, ``VBD``, ``NN``, ``JJ`` ...) and
    then by its two-letter prefix (``VB`` also matches ``VBD``/``VBP``/``VBZ``),
    so ``USE/VB`` covers every verb form. POS-qualified entries win over plain
    ones, which win over g2p_en, which wins over its neural fallback.
    """

    def __init__(self, horsewords_dictionary = 'new_horsewords.clean'):
        self.horsedict = {}
        with open(horsewords_dictionary, 'r') as f:
            while line := f.readline():
                if not line.strip():
                    continue
                baseword, transcription = line.split('  ')
                self.horsedict[baseword] = transcription

    @staticmethod
    def tag_words(words):
        """Return Penn POS tags for `words`, 1:1 and in order.

        g2p_en resolves heteronyms ("read", "refuse", "present", ...) from the
        Penn tag of the word *in its sentence* (`pos_tag` over the whole
        input). Tagging a single word at a time throws that context away, so a
        caller that phonemises word by word must tag the whole utterance once
        with this helper and pass each tag to `phonemize(..., tag=...)`.
        """
        cleaned = [_WORD_CHARS.sub('', str(w).lower()) or '_' for w in words]
        return [tag for _, tag in nltk.pos_tag(cleaned)]

    def phonemize(self, text, tag=None):
        """Uses g2p_en + a dictionary to convert a string into ARPAbet chars.

        Args:
            text: Whitespace-separated words (normally one forced-alignment
                token, e.g. "read" / "read.").
            tag: Optional Penn POS tag for the word in `text` (from
                `tag_words` over the whole sentence). When given, heteronyms
                get the pronunciation matching this part of speech; when
                `None` the historical context-free behaviour is reproduced
                (g2p_en then tags the word in isolation, which defaults to its
                most likely standalone tag).

        Returns:
            Space-separated per-word phoneme strings, ARPAbet with stress
            digits and no separators inside a word ("R IY1 D" -> "RIY1D").
        """
        spl = text.split()
        l = ''
        for s in spl:
            l += self._phonemize_word(s, tag) + ' '
        return l.strip()

    def phonemize_sentence(self, text, tags=None):
        """Phonemize a whole utterance, tagging its words in context.

        Equivalent to joining `phonemize(word, tag=tag)` over the utterance's
        whitespace tokens, i.e. its output differs from a per-word loop only
        for heteronyms.
        """
        words = str(text).split()
        if tags is None:
            tags = self.tag_words(words)
        return ' '.join(self._phonemize_word(w, t) for w, t in zip(words, tags))

    def _phonemize_word(self, s, tag=None):
        """Phonemize one token: hand dictionary > POS-aware heteronym > g2p_en."""
        pron = self._dictionary_pronunciation(s, tag)
        if pron is not None:
            # Hand-written overrides always win, exactly as before.
            return pron
        if tag is not None:
            pron = self._homograph_pronunciation(s, tag)
            if pron is not None:
                return pron
        p = [arp for arp in g2p(s) if arp != ' ']
        return ''.join(p)

    def _dictionary_pronunciation(self, s, tag=None):
        """Hand-dictionary lookup for `s`, preferring a POS-qualified key."""
        s_up = s.strip().upper()
        if tag is not None:
            for key in (f'{s_up}/{tag}', f'{s_up}/{tag[:2]}'):
                if key in self.horsedict:
                    return ''.join(self.horsedict[key].split())
        if s_up in self.horsedict:
            return ''.join(self.horsedict[s_up].split())
        return None

    def _homograph_pronunciation(self, s, tag):
        """POS-conditioned pronunciation for a heteronym, else None.

        Mirrors the homograph branch of `g2p_en.G2p.__call__`, but with the
        caller-supplied tag instead of a single-word `pos_tag`.
        """
        match = _TOKEN_PARTS.match(s)
        if match is None:
            return None
        prefix, word, suffix = match.groups()
        key = _WORD_CHARS.sub('', word.lower())
        if not key or key not in HOMOGRAPHS:
            return None
        pron1, pron2, pos1 = HOMOGRAPHS[key]
        pron = pron1 if tag.startswith(pos1) else pron2
        # g2p_en emits punctuation as literal tokens, which the caller joins
        # without separators; reproduce that so the phoneme string keeps the
        # same shape as the plain g2p path.
        return prefix + ''.join(pron) + suffix
