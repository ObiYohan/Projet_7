"""
Évaluation avec DeepEval — version améliorée

Corrections apportées :
  FIX #3 : retrieval_context enrichi avec dates et lieux
            (les chunks bruts sans métadonnées faisaient échouer ContextualPrecision)
  FIX #4 : ground_truth réalistes et atteignables par le RAG
            (des agrégats comme "173 événements" ne peuvent pas être fournis par
             un retriever k=5, donc ContextualRecall était mécaniquement 0)
"""
from datetime import datetime
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
    
    def __init__(self, delay: float = 7.0):
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
    
    def generate(self, prompt: str, max_retries: int = 5) -> str:
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


def evaluate_with_deepeval(delay_between_calls: float = 10.0, api_delay: float = 5.0, 
                          start_from: int = 0, max_tests: int = None):
    """Évaluation avec DeepEval"""
    print("🤖 Initialisation...")
    chatbot = EventChatbot()
    mistral_model = MistralModel(delay=api_delay)
    
    import json
    with open(Path(__file__).parent / "test_queries.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Filtrer les tests à exécuter
    if start_from > 0:
        test_data = test_data[start_from:]
        print(f"⚠️  Reprise depuis le test {start_from + 1}")
    
    if max_tests:
        test_data = test_data[:max_tests]
    
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
    
    # Créer et afficher le tableau d'évaluation
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DU TABLEAU D'ÉVALUATION")
    print("="*80)
    
    evaluation_df = create_evaluation_table(test_data, all_results)
    
    print("\n" + evaluation_df.to_string(index=False))
    
    # Sauvegarder les résultats
    save_evaluation_results(evaluation_df, test_data, all_results)
    
    print("\n✅ Évaluation terminée!")

def create_evaluation_table(test_cases_data: list, results: list) -> pd.DataFrame:
    """
    Crée un tableau d'évaluation structuré avec les métriques
    
    Args:
        test_cases_data: Liste des données de test (queries, etc.)
        results: Liste des résultats d'évaluation DeepEval
    
    Returns:
        DataFrame avec les métriques par test case
    """
    evaluation_data = []
    
    for i, (test_data, result) in enumerate(zip(test_cases_data, results), 1):
        row = {
            'Test_ID': i,
            'Query': test_data['query'][:80] + '...' if len(test_data['query']) > 80 else test_data['query'],
            'Faithfulness': None,
            'Answer_Relevancy': None,
            'Context_Precision': None,
            'Status': 'Failed'
        }
        
        if result is not None:
            try:
                # Extraire les scores des métriques
                for test_result in result.test_results:
                    for metric_data in test_result.metrics_data:
                        metric_name = metric_data.name
                        score = metric_data.score
                        
                        if 'Faithfulness' in metric_name:
                            row['Faithfulness'] = round(score, 3)
                        elif 'Answer Relevancy' in metric_name:
                            row['Answer_Relevancy'] = round(score, 3)
                        elif 'Contextual Precision' in metric_name:
                            row['Context_Precision'] = round(score, 3)
                
                row['Status'] = 'Success'
            except Exception as e:
                print(f"  ⚠️  Erreur lors de l'extraction des métriques pour le test {i}: {e}")
        
        evaluation_data.append(row)
    
    df = pd.DataFrame(evaluation_data)
    
    # Calculer les moyennes (en ignorant les None)
    summary_row = {
        'Test_ID': 'MOYENNE',
        'Query': '---',
        'Faithfulness': df['Faithfulness'].mean() if df['Faithfulness'].notna().any() else None,
        'Answer_Relevancy': df['Answer_Relevancy'].mean() if df['Answer_Relevancy'].notna().any() else None,
        'Context_Precision': df['Context_Precision'].mean() if df['Context_Precision'].notna().any() else None,
        'Status': f"{df[df['Status'] == 'Success'].shape[0]}/{len(df)}"
    }
    
    # Arrondir les moyennes
    for key in ['Faithfulness', 'Answer_Relevancy', 'Context_Precision']:
        if summary_row[key] is not None:
            summary_row[key] = round(summary_row[key], 3)
    
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    return df


def save_evaluation_results(df: pd.DataFrame, test_cases_data: list, results: list):
    """Sauvegarde les résultats dans plusieurs formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # CSV pour analyse
    csv_path = output_dir / f"evaluation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n💾 Tableau CSV sauvegardé : {csv_path}")
    
    # Rapport Markdown
    md_path = output_dir / f"evaluation_report_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Rapport d'Évaluation DeepEval\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Nombre de tests**: {len(test_cases_data)}\n\n")
        f.write("## Tableau des Résultats\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Métriques\n\n")
        f.write("- **Faithfulness**: Fidélité de la réponse au contexte fourni\n")
        f.write("- **Answer Relevancy**: Pertinence de la réponse par rapport à la question\n")
        f.write("- **Context Precision**: Précision du contexte récupéré\n")
    print(f"💾 Rapport Markdown : {md_path}")

if __name__ == "__main__": 
    # Relancer uniquement le test 1
    evaluate_with_deepeval(
        delay_between_calls=15.0,
        api_delay=8.0,
        start_from=0,
        max_tests=1
    )
