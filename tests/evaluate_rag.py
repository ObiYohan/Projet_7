"""
Évaluation automatique du RAG avec Ragas
Métriques : Answer Relevancy, Faithfulness, Context Precision, Context Recall
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import time
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent))
from src.chatbot import EventChatbot

# Import Ragas
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from langchain_mistralai import ChatMistralAI


class RagasEvaluator:
    """Évaluateur utilisant Ragas pour des métriques précises"""
    
    def __init__(self, chatbot: EventChatbot, delay_between_calls: float = 5.0):
        self.chatbot = chatbot
        self.delay_between_calls = delay_between_calls
        
        # Initialiser le modèle Mistral pour Ragas
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0
        )
    
    def load_test_dataset(self, test_file: str = None):
        """Charge le dataset de test"""
        if test_file is None:
            test_file = Path(__file__).parent / "test_queries.json"
        
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_response_with_retry(self, query: str, context: str, max_retries: int = 3) -> str:
        """Génère une réponse avec retry"""
        for attempt in range(max_retries):
            try:
                return self.chatbot.generate_response(query, context)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = (attempt + 1) * 15
                    print(f"  ⚠️  Rate limit. Attente {wait_time}s... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        return "[ERREUR: Rate limit]"
                else:
                    raise
        return "[ERREUR: Impossible de générer]"
    
    def prepare_ragas_dataset(self, test_data: list) -> Dataset:
        """Prépare les données au format Ragas"""
        print("\n🔄 Génération des réponses pour l'évaluation Ragas...")
        
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for i, item in enumerate(test_data, 1):
            query = item['question']
            ground_truth = item['ground_truth']
            
            print(f"\n📝 [{i}/{len(test_data)}] {query}")
            
            # Récupérer contexte et générer réponse
            relevant_chunks, _ = self.chatbot.search_relevant_events(query, k=5)
            context_list = relevant_chunks['text'].tolist()
            context_str = self.chatbot.format_context(relevant_chunks)
            
            answer = self.generate_response_with_retry(query, context_str)
            
            questions.append(query)
            answers.append(answer)
            contexts.append(context_list)
            ground_truths.append([ground_truth])  # Ragas attend une liste
            
            print(f"  ✓ Réponse générée ({len(answer.split())} mots)")
            
            if i < len(test_data):
                print(f"  ⏸️  Pause {self.delay_between_calls}s...")
                time.sleep(self.delay_between_calls)
        
        # Créer le dataset Ragas
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def evaluate_with_ragas(self, test_file: str = None) -> dict:
        """Évalue avec Ragas"""
        print("\n" + "="*80)
        print("🎯 ÉVALUATION AVEC RAGAS")
        print("="*80)
        
        test_data = self.load_test_dataset(test_file)
        print(f"\n✓ {len(test_data)} requêtes de test chargées")
        
        # Préparer le dataset
        dataset = self.prepare_ragas_dataset(test_data)
        
        print("\n🧪 Calcul des métriques Ragas...")
        print("  (Cela peut prendre plusieurs minutes)")
        
        # Évaluer avec Ragas
        try:
            results = evaluate(
                dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall
                ],
                llm=self.llm
            )
            
            return results
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l'évaluation Ragas: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_ragas_results(self, results):
        """Affiche les résultats Ragas"""
        if results is None:
            print("\n❌ Pas de résultats à afficher")
            return
        
        print("\n" + "="*80)
        print("📊 RÉSULTATS RAGAS")
        print("="*80)
        
        # Convertir en DataFrame pour affichage
        df = results.to_pandas()
        
        print("\n📈 Métriques moyennes:")
        print(f"  • Answer Relevancy:   {df['answer_relevancy'].mean():.2%}")
        print(f"  • Faithfulness:       {df['faithfulness'].mean():.2%}")
        print(f"  • Context Precision:  {df['context_precision'].mean():.2%}")
        print(f"  • Context Recall:     {df['context_recall'].mean():.2%}")
        
        # Score global
        avg_score = df[['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall']].mean().mean()
        
        print(f"\n🎯 Score global: {avg_score:.2%}")
        
        if avg_score >= 0.7:
            print("✅ Excellent - Le système RAG est performant")
        elif avg_score >= 0.5:
            print("⚠️  Acceptable - Des améliorations sont recommandées")
        else:
            print("❌ Insuffisant - Le système nécessite des ajustements majeurs")
        
        # Détails par question
        print("\n📋 Détails par question:")
        for idx, row in df.iterrows():
            print(f"\n  Question {idx + 1}:")
            print(f"    Relevancy:  {row['answer_relevancy']:.2%}")
            print(f"    Faithfulness: {row['faithfulness']:.2%}")
            print(f"    Precision:  {row['context_precision']:.2%}")
            print(f"    Recall:     {row['context_recall']:.2%}")
    
    def save_ragas_results(self, results, output_file: str = None):
        """Sauvegarde les résultats Ragas"""
        if results is None:
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(__file__).parent / f"ragas_results_{timestamp}.json"
        
        df = results.to_pandas()
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'answer_relevancy': float(df['answer_relevancy'].mean()),
                'faithfulness': float(df['faithfulness'].mean()),
                'context_precision': float(df['context_precision'].mean()),
                'context_recall': float(df['context_recall'].mean()),
                'global_score': float(df[['answer_relevancy', 'faithfulness', 'context_precision', 'context_recall']].mean().mean())
            },
            'details': df.to_dict(orient='records')
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés : {output_file}")


def main():
    """Fonction principale"""
    try:
        print("🤖 Initialisation du chatbot...")
        chatbot = EventChatbot()
        
        print("🔧 Configuration de Ragas...")
        evaluator = RagasEvaluator(chatbot, delay_between_calls=15.0)
        
        results = evaluator.evaluate_with_ragas()
        evaluator.print_ragas_results(results)
        evaluator.save_ragas_results(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Évaluation interrompue")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()