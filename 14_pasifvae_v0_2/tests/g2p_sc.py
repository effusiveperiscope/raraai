from speechbrain.inference.text import GraphemeToPhoneme

# Load the G2P model
g2p = GraphemeToPhoneme.from_hparams(
    source="speechbrain/soundchoice-g2p",
    savedir="pretrained_models/soundchoice-g2p"
)

print("Enter text to convert to phonemes (type 'exit' to quit):")

while True:
    text = input(">>> ")
    if text.lower() == "exit":
        break
    phonemes = g2p(text)
    print("Phonemes:", phonemes)
