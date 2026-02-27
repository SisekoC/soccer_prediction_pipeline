# features/llm_scorer.py (new module)
class LLMScorer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Initialize LLM client
        pass
    
    def score_anchor(self, news_text: str, anchor_description: str) -> float:
        """Ask LLM to rate relevance of news to this anchor (0-1)"""
        prompt = f"""
        News text: {news_text}
        
        Does this news indicate that: {anchor_description}
        
        Rate from 0 (completely unrelated) to 1 (strongly indicates this factor).
        Return only a number.
        """
        response = self.llm.complete(prompt)
        return float(response.strip())