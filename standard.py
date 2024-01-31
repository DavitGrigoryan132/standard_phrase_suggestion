import spacy
import pandas as pd
from embeddings import EmbeddingsModel
from torch import Tensor


class StandardisedPhrases:
    def __init__(self):
        self.phrases: list[str] = []
        self.phrases_embeddings: Tensor = Tensor()
        self.embeddings_model: EmbeddingsModel = EmbeddingsModel()
        self.spacy_nlp = spacy.load('en_core_web_sm')

    def read_phrases(self, phrases_file_path):
        self.phrases = pd.read_csv(phrases_file_path)["Optimal performance"].tolist()
        self.phrases_embeddings = self.embeddings_model.get_embeddings(self.phrases)

    def find_similar_phrase(self, phrase):
        encoded_phrase = self.embeddings_model.get_embeddings(phrase)

        scores = self.embeddings_model.cosine_similarity(encoded_phrase, self.phrases_embeddings)
        phrase_score_pair = zip(self.phrases, scores)

        return max(phrase_score_pair, key=lambda x: x[1])

    def give_standardised_suggestions(self, user_text, threshold=0.45, window_size=3):
        sentences = self.spacy_nlp(user_text).sents
        suggestions = []
        for sentence in sentences:
            words = sentence.text.split()
            window_start = 0
            while window_start <= (len(words) - window_size):
                input_phrase = " ".join(words[window_start: window_start + window_size])

                suggested_phrase, score = self.find_similar_phrase(
                    input_phrase)

                if score >= threshold:
                    if len(suggestions) > 0 and len(
                            set(input_phrase.split()).intersection(suggestions[-1][0].split())) >= 1:
                        if score > suggestions[-1][2]:
                            suggestions.pop(-1)
                            suggestions.append((input_phrase, suggested_phrase, score))
                    else:
                        suggestions.append((input_phrase, suggested_phrase, score))

                window_start += 1

        return suggestions
