from g2p import make_g2p

print("Enter text to convert to phonemes (type 'exit' to quit):")
g2p = make_g2p('eng', 'eng-arpabet')

while True:
    text = input(">>> ")
    if text.lower() == "exit":
        break
    phonemes = g2p(text).output_string
    print("Phonemes:", phonemes)
