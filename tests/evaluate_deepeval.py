"""
Évaluation avec DeepEval - Alternative à Ragas
"""
import sys
from pathlib import Path
import time
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
    """Wrapper Mistral pour DeepEval avec rate limiting et retry"""
    
    def __init__(self, delay: float = 5.0):
        from mistralai import Mistral
        import os
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.model = "mistral-large-latest"
        self.delay = delay
        self.last_call_time = 0
    
    def load_model(self):
        return self.client
    
    def _wait_if_needed(self):
        """Attendre si nécessaire pour respecter le rate limit"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.delay:
            wait_time = self.delay - time_since_last_call
            print(f"  ⏸️  Rate limit: pause de {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_call_time = time.time()
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Génère avec retry en cas de rate limit"""
        for attempt in range(max_retries):
            try:
                self._wait_if_needed()
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"  ⚠️  Rate limit! Attente de {wait_time}s... (tentative {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ Échec après {max_retries} tentatives")
                        raise
                else:
                    raise
        return ""
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.model


def evaluate_with_deepeval(delay_between_calls: float = 5.0, api_delay: float = 3.0):
    """Évaluation avec DeepEval"""
    print("🤖 Initialisation...")
    chatbot = EventChatbot()
    mistral_model = MistralModel(delay=api_delay)
    
    # Charger les questions de test
    import json
    with open(Path(__file__).parent / "test_queries.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"⏱️  Délai entre les test cases: {delay_between_calls}s")
    print(f"⏱️  Délai entre les appels API: {api_delay}s")
    print(f"⚠️  Note: DeepEval lance 4 métriques en parallèle par test case")
    
    test_cases = []
    
    for i, item in enumerate(test_data, 1):
        query = item['query']
        ground_truth = item['ground_truth']
        
        print(f"\n📝 [{i}/{len(test_data)}] {query}")
        
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
        
        # Pause entre les appels
        if i < len(test_data):
            print(f"  ⏸️  Pause de {delay_between_calls}s...")
            time.sleep(delay_between_calls)
    
    # Définir les métriques (elles s'exécuteront en parallèle)
    metrics = [
        AnswerRelevancyMetric(model=mistral_model, threshold=0.7),
        FaithfulnessMetric(model=mistral_model, threshold=0.7),
        ContextualPrecisionMetric(model=mistral_model, threshold=0.7),
        ContextualRecallMetric(model=mistral_model, threshold=0.7)
    ]
    
    # Évaluer un test case à la fois
    print("\n🧪 Évaluation en cours (mode séquentiel)...")
    print("⚠️  Chaque test case prendra ~30-60s (4 métriques × délais)")
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Évaluation du test case {i}/{len(test_cases)}...")
        try:
            result = evaluate([test_case], metrics)
            all_results.append(result)
            print(f"  ✅ Test case {i} terminé")
        except Exception as e:
            print(f"  ❌ Erreur sur test case {i}: {e}")
            all_results.append(None)
        
        # Pause entre chaque test case
        if i < len(test_cases):
            print(f"  ⏸️  Pause de {delay_between_calls}s avant le prochain test...")
            time.sleep(delay_between_calls)
    
    print("\n📊 Résultats finaux:")
    for i, result in enumerate(all_results, 1):
        print(f"\n{'='*50}")
        print(f"Test case {i}:")
        if result:
            print(result)
        else:
            print("❌ Échec de l'évaluation")


if __name__ == "__main__":
    evaluate_with_deepeval()