import json

from transformers import pipeline, AutoModelForSequenceClassification

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
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
