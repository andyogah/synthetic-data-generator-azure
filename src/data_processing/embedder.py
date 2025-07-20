from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):
        return self.model.encode(chunks, convert_to_tensor=True)

    def save_embeddings(self, embeddings, file_path):
        import torch
        torch.save(embeddings, file_path)

    def load_embeddings(self, file_path):
        import torch
        return torch.load(file_path)