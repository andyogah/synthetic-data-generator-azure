from typing import List, Dict

class TextSearch:
    def __init__(self, original_data: List[str]):
        self.original_data = original_data

    def search(self, query: str) -> List[Dict[str, str]]:
        results = []
        for index, text in enumerate(self.original_data):
            if query.lower() in text.lower():
                results.append({"index": index, "text": text})
        return results

    def advanced_search(self, query: str, context: str) -> List[Dict[str, str]]:
        results = []
        for index, text in enumerate(self.original_data):
            if query.lower() in text.lower() and context.lower() in text.lower():
                results.append({"index": index, "text": text})
        return results

    def fuzzy_search(self, query: str) -> List[Dict[str, str]]:
        from difflib import get_close_matches
        results = []
        for index, text in enumerate(self.original_data):
            matches = get_close_matches(query, text.split(), n=5, cutoff=0.6)
            if matches:
                results.append({"index": index, "text": text})
        return results

    def get_all_texts(self) -> List[str]:
        return self.original_data

    def count_results(self, query: str) -> int:
        return len(self.search(query))