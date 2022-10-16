from transformers import pipeline, AutoModelForSequenceClassification


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

MODEL_PATH = 'models/punctuation-nilc-bert'
def main():

    # Load the model

    classifier_ner = pipeline("ner", MODEL_PATH, grouped_entities=True)
    outputs = classifier_ner("Olá como vai você")
    print(outputs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
