import torch
import torch.nn.functional as F
from typing import List, Union
from transformers import AutoTokenizer, AutoModel


class EmbeddingsModel:
    """
    A class for generating sentence embeddings using pre-trained transformer models.

    Parameters:
        model_url (str, optional): The Hugging Face model identifier or URL.
            Default is "sentence-transformers/all-MiniLM-L6-v2".

    Members:
        tokenizer (transformers.AutoTokenizer): The tokenizer for tokenizing input sentences.
        model (transformers.AutoModel): The transformer model for generating embeddings.

    Methods:
        get_embeddings(sentences: Union[str, List[str]]) -> torch.Tensor:
            Tokenizes and embeds a list of sentences.

        cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
            Computes the cosine similarity between two embeddings.
    """
    def __init__(self, model_url: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingsModel.

        Parameters:
            model_url (str, optional): The Hugging Face model identifier or URL.
                Default is "sentence-transformers/all-MiniLM-L6-v2".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModel.from_pretrained(model_url)

    @staticmethod
    def __mean_pooling(input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-pooling of token embeddings.

        Parameters:
            input_tensor (torch.Tensor): The token embeddings.
            attention_mask (torch.Tensor): The attention mask for input sentences.

        Returns:
            torch.Tensor: Mean-pooled embeddings.
        """
        token_embeddings = input_tensor[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenize and embed a list of sentences.

        Parameters:
            sentences (Union[str, List[str]]): Input sentences.

        Returns:
            torch.Tensor: Normalized embeddings of input sentences.
        """
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            output = self.model(**encoded_sentences)

        embeddings = self.__mean_pooling(output, encoded_sentences["attention_mask"])

        return F.normalize(embeddings, p=2, dim=1)

    @staticmethod
    def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> int:
        """
        Compute cosine similarity between two embeddings.

        Parameters:
            embedding1 (torch.Tensor): First embedding.
            embedding2 (torch.Tensor): Second embedding.

        Returns:
            float: Cosine similarity between the two embeddings.
        """
        return F.cosine_similarity(embedding1, embedding2).numpy()
