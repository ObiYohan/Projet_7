"""
Évaluation avec DeepEval - Alternative à Ragas
"""
import sys
from pathlib import Path
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

sys.path.append(str(Path(__file__).parent.parent))
from src.chatbot import EventChatbot


class MistralModel(DeepEvalBaseLLM):
    """Wrapper Mistral pour DeepEval"""
    
    def __init__(self):
        from mistralai import Mistral
        import os
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.model = "mistral-large-latest"
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.model


def evaluate_with_deepeval():
    """Évaluation avec DeepEval"""
    print("🤖 Initialisation...")
    chatbot = EventChatbot()
    mistral_model = MistralModel()
    
    # Charger les questions de test
    import json
    with open(Path(__file__).parent / "test_queries.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_cases = []
    
    for item in test_data:
        query = item['question']
        ground_truth = item['ground_truth']
        
        print(f"\n📝 {query}")
        
        # Récupérer contexte et réponse
        relevant_chunks, _ = chatbot.search_relevant_events(query, k=5)
        context = relevant_chunks['text'].tolist()
        context_str = chatbot.format_context(relevant_chunks)
        answer = chatbot.generate_response(query, context_str)
        
        # Créer un test case DeepEval
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=context
        )
        test_cases.append(test_case)
    
    # Définir les métriques
    metrics = [
        AnswerRelevancyMetric(model=mistral_model, threshold=0.7),
        FaithfulnessMetric(model=mistral_model, threshold=0.7),
        ContextualPrecisionMetric(model=mistral_model, threshold=0.7),
        ContextualRecallMetric(model=mistral_model, threshold=0.7)
    ]
    
    # Évaluer
    print("\n🧪 Évaluation en cours...")
    results = evaluate(test_cases, metrics)
    
    print("\n📊 Résultats:")
    print(results)


if __name__ == "__main__":
    evaluate_with_deepeval()