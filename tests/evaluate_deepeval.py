"""
Évaluation avec DeepEval — version améliorée

Corrections apportées :
  FIX #3 : retrieval_context enrichi avec dates et lieux
            (les chunks bruts sans métadonnées faisaient échouer ContextualPrecision)
  FIX #4 : ground_truth réalistes et atteignables par le RAG
            (des agrégats comme "173 événements" ne peuvent pas être fournis par
             un retriever k=5, donc ContextualRecall était mécaniquement 0)
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
import pandas as pd

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
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.delay:
            wait_time = self.delay - time_since_last_call
            print(f"  ⏸️  Rate limit: pause de {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_call_time = time.time()
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
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


def build_rich_context(relevant_chunks: pd.DataFrame) -> list[str]:
    """
    FIX #3 : Construit des contextes enrichis (texte + dates + lieu) par événement.

    DeepEval évalue chaque élément de retrieval_context indépendamment.
    Un chunk de texte brut sans dates ni ville semble "hors sujet" à l'LLM-juge.
    En incluant les métadonnées structurées, chaque entrée devient auto-suffisante.
    """
    if relevant_chunks.empty:
        return []

    context_items = []

    for uid, group in relevant_chunks.groupby('uid'):
        first_row = group.iloc[0]
        parts = []

        # Titre / nom de l'événement
        if 'title' in first_row and pd.notna(first_row.get('title')):
            parts.append(f"Événement : {first_row['title']}")

        # Dates
        date_parts = []
        for col, label in [
            ('firstdate_begin', 'Début'),
            ('lastdate_end', 'Fin'),
        ]:
            if col in first_row and pd.notna(first_row[col]):
                ts = pd.to_datetime(first_row[col], utc=True)
                date_parts.append(f"{label} : {ts.strftime('%d/%m/%Y %H:%M')}")
        if date_parts:
            parts.append("Dates — " + " | ".join(date_parts))

        # Localisation
        loc_parts = []
        for col, label in [
            ('location_name', 'Lieu'),
            ('location_city', 'Ville'),
            ('location_postalcode', 'CP'),
        ]:
            if col in first_row and pd.notna(first_row.get(col)):
                loc_parts.append(f"{label} : {first_row[col]}")
        if loc_parts:
            parts.append("Localisation — " + " | ".join(loc_parts))

        # Texte des chunks
        if 'chunk_id' in group.columns:
            text = " ".join(group.sort_values('chunk_id')['text'].tolist())
        else:
            text = " ".join(group['text'].tolist())
        parts.append(f"Description : {text}")

        context_items.append("\n".join(parts))

    return context_items


def evaluate_with_deepeval(delay_between_calls: float = 5.0, api_delay: float = 3.0):
    """Évaluation avec DeepEval"""
    print("🤖 Initialisation...")
    chatbot = EventChatbot()
    mistral_model = MistralModel(delay=api_delay)
    
    import json
    with open(Path(__file__).parent / "test_queries.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"⏱️  Délai entre les test cases  : {delay_between_calls}s")
    print(f"⏱️  Délai entre les appels API  : {api_delay}s")
    print(f"⚠️  DeepEval lance 4 métriques en parallèle par test case")
    
    test_cases = []
    
    for i, item in enumerate(test_data, 1):
        query = item['query']
        ground_truth = item['ground_truth']
        
        print(f"\n📝 [{i}/{len(test_data)}] {query}")
        
        relevant_chunks, _ = chatbot.search_relevant_events(query, k=5)
        context_str = chatbot.format_context(relevant_chunks)
        answer = chatbot.generate_response(query, context_str)

        # FIX #3 : contexte enrichi avec métadonnées
        rich_context = build_rich_context(relevant_chunks)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=rich_context
        )
        test_cases.append(test_case)
        
        if i < len(test_data):
            print(f"  ⏸️  Pause de {delay_between_calls}s...")
            time.sleep(delay_between_calls)
    
    metrics = [
        AnswerRelevancyMetric(model=mistral_model, threshold=0.7),
        FaithfulnessMetric(model=mistral_model, threshold=0.7),
        ContextualPrecisionMetric(model=mistral_model, threshold=0.7),
        ContextualRecallMetric(model=mistral_model, threshold=0.7)
    ]
    
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
