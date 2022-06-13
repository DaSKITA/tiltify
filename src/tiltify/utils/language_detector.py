import os
import fasttext

from tiltify.config import Path


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = os.path.join(Path.data_path, "pretrained_language_model/fasttext/lid.176.bin")
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        text = text.replace("\n", " ")
        predictions = self.model.predict(text, k=1)
        language = predictions[0][0].split("__")[-1]
        return language


if __name__ == '__main__':
    LANGUAGE = LanguageIdentification()
    lang = LANGUAGE.predict_lang("Hallo\n")
    print(lang)
