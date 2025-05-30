from g2p.g2p_arpabet import G2PConverter, DefaultG2PFallback

g2p = G2PConverter(remove_stress=True, word_boundaries=True, fallback=DefaultG2PFallback())
g2p.load_dictionary("g2p/cmudict.clean", "g2p/new_horsewords.clean")

print("Enter text to convert to phonemes (type 'exit' to quit):")

while True:
    text = input(">>> ")
    if text.lower() == "exit":
        break
    phonemes = g2p.g2p(text)
    print("Phonemes:", phonemes)
