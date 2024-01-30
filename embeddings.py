from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class EmbeddingsModel:
    def __init__(self, model_url="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModel.from_pretrained(model_url)

    @staticmethod
    def __mean_pooling(input_tensor, attention_mask):
        token_embeddings = input_tensor[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            output = self.model(**encoded_sentences)

        embeddings = self.__mean_pooling(output, encoded_sentences["attention_mask"])

        return F.normalize(embeddings, p=2, dim=1)

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return F.cosine_similarity(embedding1, embedding2)
