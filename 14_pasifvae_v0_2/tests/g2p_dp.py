from dp.phonemizer import Phonemizer
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_forward.pt')
print("Enter text to convert to phonemes (type 'exit' to quit):")

while True:
    text = input(">>> ")
    if text.lower() == "exit":
        break
    phonemes = phonemizer(text, lang='en_us')
    print("Phonemes:", phonemes)
