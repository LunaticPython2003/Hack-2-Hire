import json
import language_tool_python
# from spellchecker import SpellChecker
import re
##processing of english statement
# Read the JSON file

def remove_repetitions(text):
    words = text.split()
    unique_words = []
    previous_word = None

    for word in words:
        if word != previous_word:
            unique_words.append(word)
        previous_word = word

    return ' '.join(unique_words)

def correct_grammar(text):
        tool = language_tool_python.LanguageTool('en-US')
        corrected_text = tool.correct(text)
        return corrected_text

def ret_text():
    with open("predicted_characters.json", "r") as json_file:
        data = json.load(json_file)
    input_sentence = data['key']
    original_text = input_sentence
    # def correct_spelling(text):
    #     spell = SpellChecker()
    #     words = text.split()  # Split the text into words
    #     corrected_words = [spell.correction(word) for word in words]
    #     corrected_text = ' '.join(corrected_words)
    #     return corrected_text

    


        # Correct spelling
    # corrected_text = correct_spelling(original_text)

        # Remove repetitions
    text_without_repetitions = remove_repetitions(original_text)

        # Correct grammar
    final_text = correct_grammar(text_without_repetitions)
    return final_text


## generating audio
# import torch
# from TTS.api import TTS

# device = "cuda" if torch.cuda.is_available() else "cpu"

# tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts").to(device)
# tts.tts_to_file(final_text, speaker_wav="test/test.wav", language="en", file_path="/Users/thushara/Documents/Thushara/SMIT/PROJECTS/Akhil/Final_Model/output.wavi")
