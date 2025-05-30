from g2p_en import G2p
g2p = G2p()

print("Enter text to convert to phonemes (type 'exit' to quit):")

while True:
    text = input(">>> ")
    if text.lower() == "exit":
        break
    phonemes = g2p(text)
    print("Phonemes:", phonemes)
