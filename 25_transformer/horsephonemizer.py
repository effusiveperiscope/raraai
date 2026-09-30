import nltk
nltk.download('averaged_perceptron_tagger_eng')

from g2p_en import G2p
import time
import re
import string

g2p = G2p()

class HorsePhonemizer:
    def __init__(self, horsewords_dictionary = 'new_horsewords.clean'):
        self.horsedict = {}
        with open(horsewords_dictionary, 'r') as f:
            while line := f.readline():
                baseword, transcription = line.split('  ')
                self.horsedict[baseword] = transcription

    def phonemize(self, text):
        """Uses g2p_en + a dictionary to convert a string into ARPAbet characters"""
        spl = text.split()
        l = ''
        for s in spl:
            s_up = s.strip().upper()
            if s_up in self.horsedict:
                arpabet = ''.join(self.horsedict[s_up].split())
                l += arpabet + ' '
            else:
                p = [arp for arp in g2p(s) if arp != ' ']
                arpabet_string = ''.join(p)
                l += arpabet_string + ' '
        return l.strip()