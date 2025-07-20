from typing import List

class Chunker:
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= self.max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_data(self, data: List[str]) -> List[List[str]]:
        return [self.chunk_text(text) for text in data]