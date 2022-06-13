import os
import fasttext
import urllib.request

from tiltify.config import Path


class LanguageDetector:

    def __init__(self):
        self.pretrained_lang_model_path = os.path.join(Path.data_path, "pretrained_language_model/fasttext/lid.176.bin")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            self.pretrained_lang_model_path)
        self.model = fasttext.load_model(self.pretrained_lang_model_path)

    def predict_lang(self, text):
        text = text.replace("\n", " ")
        predictions = self.model.predict(text, k=1)
        language = predictions[0][0].split("__")[-1]
        return language


if __name__ == '__main__':
    LANGUAGE = LanguageDetector()
    lang = LANGUAGE.predict_lang("Hallo\n")
    print(lang)
