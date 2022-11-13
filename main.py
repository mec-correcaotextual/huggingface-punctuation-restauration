import json

from transformers import pipeline, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer

from utils import remove_punctuation

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

MODEL_PATH = 'tiagoblima/punctuation-nilc-bert'


def main():
    # Load the model
    annotator1 = json.load(open('raw_datasets/annotator1.json', 'r'))

    classifier_ner = pipeline("ner", MODEL_PATH, grouped_entities=True, use_auth_token=True)
    tokens = remove_punctuation(annotator1[0]['text'])
    outputs = classifier_ner(' '.join(tokens))
    print(outputs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    checkpoint = 'tiagoblima/punctuation-nilc-t5-base'
    import torch
    print(torch.__version__)
    T5ForConditionalGeneration.from_pretrained(checkpoint, use_auth_token=True, returns='py').save_pretrained('models/tiagoblima/punctuation-nilc-t5-base')
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, use_auth_token=True).save_pretrained('models/tiagoblima/punctuation-nilc-t5-base')
    #main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
