import time
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

import re
from nltk import pos_tag, word_tokenize


#This part is the function of rephasing the caption

#Removes adjectives that are not necessary and keeping those that are used meaningfully
def remove_redundant_adjectives(tokens):
    tagged = pos_tag(tokens)
    filtered = []
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith("JJ"):
            prev_word = tagged[i - 1][0].lower() if i > 0 else ""
            if prev_word in {"is", "are", "was", "were", "looks", "seems"}:
                filtered.append(word)
        else:
            filtered.append(word)
    return filtered

#Finds the first noun in the sentence, to use as a fallback replacement for pronouns if no provide custom noun
def find_main_noun(tokens):
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        if tag in {"NN", "NNS", "NNP", "NNPS"}:
            return word
    return "object"  

#Replaces ambiguous pronouns with a more specific noun
def resolve_pronouns(tokens, replacement=None):
    if replacement is None:
        replacement = find_main_noun(tokens)
    return [
        replacement if word.lower() in {"it", "they", "them", "this", "that"} else word
        for word in tokens
    ]

#Moves a location phrase to the beginning of the sentence
def reorder_location_phrase(tokens):
    sentence = " ".join(tokens)
    match = re.search(r"\b(on|in|at|by|under|next to)\b [^,\.]+", sentence)
    if match:
        loc_phrase = match.group()
        rest = sentence.replace(loc_phrase, "").strip(", ")
        return (loc_phrase + ", " + rest).split()
    return tokens

#Removes parts of the sentence that follow "and" if they are likely not important
def shorten_ending(tokens):
    if "and" in tokens:
        idx = tokens.index("and")
        if len(tokens) - idx <= 3:
            return tokens[:idx]
    return tokens

#Cleans up any extra or incorrect punctuation
def clean_punctuation(text):
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?]){2,}", r"\1", text)
    return text.strip()


def rephrase_caption(caption, pronoun_ref=None):
    tokens = word_tokenize(caption, preserve_line=True)
    tokens = remove_redundant_adjectives(tokens)
    tokens = resolve_pronouns(tokens, replacement=pronoun_ref)
    tokens = reorder_location_phrase(tokens)
    tokens = shorten_ending(tokens)
    return clean_punctuation(" ".join(tokens))


#BLIP Model
BLIP_PATH = "Salesforce/blip-image-captioning-base"
#TTS_MODEL
TTS_MODEL_TAG = "kan-bayashi/ljspeech_fastspeech2"

IMAGE_PATH = "test_04.jpg"

#Devices
device_tts = torch.device("cuda")
device_blip = torch.device("cpu")  

#Load BLIP model (on CPU)
print("Loading BLIP...")
processor = BlipProcessor.from_pretrained(BLIP_PATH)
model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH).to(device_blip)
model.eval()

#Load image
raw_image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(raw_image, return_tensors="pt").to(device_blip)

#Start timer to calculate latency
start_time = time.time()

#Caption generation
print("Generating Caption...")
with torch.no_grad():
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Caption: {caption}")

original = caption
rephrased = rephrase_caption(original)
print("Original:", original)
print("Rephrased:", rephrased)

device = device_tts

#Load FS2 model
text2speech = Text2Speech.from_pretrained(model_tag=TTS_MODEL_TAG)


if device.type == "cuda":
    text2speech.model = text2speech.model.to(device)
    if hasattr(text2speech, "normalize") and text2speech.normalize is not None:
        text2speech.normalize = text2speech.normalize.to(device)
    if hasattr(text2speech, "spc2wav") and text2speech.spc2wav is not None:
        text2speech.spc2wav = text2speech.spc2wav.to(device)
    text2speech.device = device


print("Synthesizing speech...")
with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
    wav = text2speech(rephrased)["wav"]

#Save output and convert to float32 before saving
sf.write("pipeline_3.wav", wav.cpu().to(torch.float32).numpy(), 22050)


#Finish the timer
end_time = time.time()
print(f"Saved: pipeline_3.wav")
print(f"Total Latency: {end_time - start_time:.2f} seconds")
