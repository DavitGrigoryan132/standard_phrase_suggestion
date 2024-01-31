import spacy
import pandas as pd
from torch import Tensor
from embeddings import EmbeddingsModel
from typing import List, Tuple


class StandardisedPhrases:
    """
    A class for handling standardized phrases and providing suggestions based on embeddings.

    Attributes:
        phrases (List[str]): List of standardized phrases.
        phrases_embeddings (Tensor): Embeddings of standardized phrases.
        embeddings_model (EmbeddingsModel): Instance of the EmbeddingsModel class.
        spacy_nlp: Spacy NLP model for natural language processing.

    Methods:
        read_phrases(phrases_file_path: str) -> None:
            Read standardized phrases from a file and compute their embeddings.

        find_similar_phrase(phrase: str) -> Tuple[str, float]:
            Find the most similar standardized phrase to the given input phrase.

        give_standardised_suggestions(user_text: str, threshold: float = 0.45,
                                      window_size: int = 3) -> List[Tuple[str, str, float]]:
            Provide standardized suggestions for phrases in the user's text.
    """

    def __init__(self):
        """
        Initialize StandardisedPhrases class.
        """
        self.phrases: list[str] = []
        self.phrases_embeddings: Tensor = Tensor()
        self.embeddings_model: EmbeddingsModel = EmbeddingsModel()
        self.spacy_nlp = spacy.load('en_core_web_sm')

    def read_phrases(self, phrases_file_path: str) -> None:
        """
        Read standardized phrases from a file and compute their embeddings.

        Parameters:
            phrases_file_path (str): Path to the file containing standardized phrases.
        """
        self.phrases = pd.read_csv(phrases_file_path)["Optimal performance"].tolist()
        self.phrases_embeddings = self.embeddings_model.get_embeddings(self.phrases)

    def find_similar_phrase(self, phrase: str) -> Tuple[str, float]:
        """
        Find the most similar standardized phrase to the given input phrase.

        Parameters:
            phrase (str): Input phrase.

        Returns:
            Tuple[str, float]: The most similar standardized phrase and its cosine similarity score.
        """
        encoded_phrase = self.embeddings_model.get_embeddings(phrase)

        scores = self.embeddings_model.cosine_similarity(encoded_phrase, self.phrases_embeddings)
        phrase_score_pair = zip(self.phrases, scores)

        return max(phrase_score_pair, key=lambda x: x[1])

    def give_standardised_suggestions(self, user_text: str, threshold: float = 0.45,
                                      window_size: int = 3) -> List[Tuple[str, str, float]]:
        """
        Provide standardized suggestions for phrases in the user's text.

        Parameters:
            user_text (str): User's input text.
            threshold (float, optional): Cosine similarity threshold for considering a suggestion. Default is 0.45.
            window_size (int, optional): Size of the sliding window for extracting phrases. Default is 3.

        Returns:
            List[Tuple[str, str, float]]: List of standardized suggestions along with their corresponding input phrases and scores.
        """
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
