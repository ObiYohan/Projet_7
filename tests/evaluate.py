"""
Script d'évaluation simplifié du chatbot RAG (sans Ragas)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mistralai.client import Mistral

sys.path.append(str(Path(__file__).parent.parent))
from src.chatbot import EventChatbot


class SimpleRAGEvaluator:
    """Évaluateur simplifié pour le système RAG"""
    
    def __init__(self, chatbot: EventChatbot, delay_between_calls: float = 3.0):
        self.chatbot = chatbot
        self.delay_between_calls = delay_between_calls
    
    def load_test_dataset(self, test_file: str = None):
        """Charge le dataset de test annoté"""
        if test_file is None:
            test_file = Path(__file__).parent / "test_queries.json"
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        return test_data
    
    def evaluate_response(self, response: str, ground_truth: str, contexts: list) -> dict:
        """Évaluation manuelle basique"""
        ground_truth_lower = ground_truth.lower()
        response_lower = response.lower()
        
        keywords = [word for word in ground_truth_lower.split() if len(word) > 4]
        matches = sum(1 for keyword in keywords if keyword in response_lower)
        
        relevance_score = matches / len(keywords) if keywords else 0
        
        context_usage = any(
            any(word in response_lower for word in context.lower().split()[:10])
            for context in contexts
        )
        
        return {
            'relevance_score': relevance_score,
            'uses_context': context_usage,
            'response_length': len(response.split())
        }
    
    def evaluate_response_semantic(self, response: str, ground_truth: str, contexts: list) -> dict:
        """Évaluation sémantique basée sur les embeddings"""
        
        # 1. Vectoriser la réponse et la ground truth
        with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as mistral:
            embeddings_res = mistral.embeddings.create(
                model="mistral-embed",
                inputs=[response, ground_truth]
            )
            response_emb = np.array([embeddings_res.data[0].embedding])
            ground_truth_emb = np.array([embeddings_res.data[1].embedding])
        
        # 2. Calculer la similarité cosinus
        semantic_similarity = cosine_similarity(response_emb, ground_truth_emb)[0][0]
        
        # 3. Vérifier l'utilisation du contexte
        context_usage = any(
            any(word in response.lower() for word in context.lower().split()[:20])
            for context in contexts
        )
        
        return {
            'relevance_score': float(semantic_similarity),  # 0 à 1
            'uses_context': context_usage,
            'response_length': len(response.split())
        }
    
    def generate_response_with_retry(self, query: str, context: str, max_retries: int = 3) -> str:
        """Génère une réponse avec gestion du rate limit"""
        for attempt in range(max_retries):
            try:
                response = self.chatbot.generate_response(query, context)
                return response
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    wait_time = (attempt + 1) * 10
                    print(f"  ⚠️  Rate limit atteint. Attente de {wait_time}s... (tentative {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        print(f"  ❌ Échec après {max_retries} tentatives")
                        return "[ERREUR: Rate limit dépassé]"
                else:
                    print(f"  ❌ Erreur inattendue: {error_msg}")
                    raise
        
        return "[ERREUR: Impossible de générer la réponse]"
    
    def compute_semantic_similarity(self, response: str, ground_truth: str) -> float:
        """Calcule la similarité sémantique entre deux textes"""
        import os
        
        with Mistral(api_key=os.getenv("MISTRAL_API_KEY")) as mistral:
            embeddings_res = mistral.embeddings.create(
                model="mistral-embed",
                inputs=[response, ground_truth]
            )
            response_emb = np.array([embeddings_res.data[0].embedding])
            ground_truth_emb = np.array([embeddings_res.data[1].embedding])
        
        similarity = cosine_similarity(response_emb, ground_truth_emb)[0][0]
        return float(similarity)
    
    def uses_context(self, response: str, contexts: list) -> bool:
        """Vérifie si la réponse utilise le contexte fourni"""
        response_lower = response.lower()
        return any(
            any(word in response_lower for word in context.lower().split()[:20])
            for context in contexts
        )
    
    def evaluate_response_advanced(self, response: str, ground_truth: str, 
                                   contexts: list, expected_data: dict = None) -> dict:
        """Évaluation multi-critères"""
        
        # 1. Similarité sémantique (40% du score)
        semantic_score = self.compute_semantic_similarity(response, ground_truth)
        
        # 2. Présence des mots-clés attendus (30% du score)
        if expected_data and 'expected_keywords' in expected_data:
            keywords = expected_data['expected_keywords']
            keyword_matches = sum(1 for kw in keywords if kw.lower() in response.lower())
            keyword_score = keyword_matches / len(keywords) if keywords else 0.5
        else:
            keyword_score = 0.5  # Score neutre si pas de keywords
        
        # 3. Nombre d'événements mentionnés (20% du score)
        if expected_data and 'expected_count' in expected_data:
            import re
            numbers = re.findall(r'\b\d+\b', response)
            count_score = 1.0 if str(expected_data['expected_count']) in numbers else 0.5
        else:
            count_score = 0.5
        
        # 4. Utilisation du contexte (10% du score)
        context_score = 1.0 if self.uses_context(response, contexts) else 0.0
        
        # Score final pondéré
        final_score = (
            semantic_score * 0.4 +
            keyword_score * 0.3 +
            count_score * 0.2 +
            context_score * 0.1
        )
        
        return {
            'relevance_score': final_score,
            'semantic_similarity': semantic_score,
            'keyword_match': keyword_score,
            'count_accuracy': count_score,
            'uses_context': context_score > 0,
            'response_length': len(response.split())
        }
    
    def evaluate(self, test_file: str = None, use_advanced: bool = False) -> dict:
        """Évalue le système RAG"""
        print("\n" + "="*80)
        print("🎯 ÉVALUATION SIMPLIFIÉE DU RAG")
        print("="*80)
        
        test_data = self.load_test_dataset(test_file)
        print(f"\n✓ {len(test_data)} requêtes de test chargées")
        print(f"⏱️  Délai entre les appels API: {self.delay_between_calls}s")
        
        if use_advanced:
            print("🔬 Mode d'évaluation : AVANCÉ (sémantique + multi-critères)")
        else:
            print("📝 Mode d'évaluation : BASIQUE (mots-clés)")
        
        results = []
        
        for i, item in enumerate(test_data, 1):
            query = item['question']
            ground_truth = item['ground_truth']
            expected_data = {
                'expected_count': item.get('expected_count'),
                'expected_keywords': item.get('expected_keywords', [])
            }
            
            print(f"\n🔍 [{i}/{len(test_data)}] Évaluation : {query}")
            
            relevant_chunks, distances = self.chatbot.search_relevant_events(query, k=5)
            context = self.chatbot.format_context(relevant_chunks)
            
            response = self.generate_response_with_retry(query, context)
            
            contexts = relevant_chunks['text'].tolist()
            
            # Choisir la méthode d'évaluation
            if use_advanced:
                eval_result = self.evaluate_response_advanced(
                    response, ground_truth, contexts, expected_data
                )
                print(f"  ✓ Score de pertinence: {eval_result['relevance_score']:.2f}")
                print(f"    - Similarité sémantique: {eval_result['semantic_similarity']:.2f}")
                print(f"    - Mots-clés: {eval_result['keyword_match']:.2f}")
                print(f"    - Précision du compte: {eval_result['count_accuracy']:.2f}")
            else:
                eval_result = self.evaluate_response(response, ground_truth, contexts)
                print(f"  ✓ Score de pertinence: {eval_result['relevance_score']:.2f}")
            
            results.append({
                'question': query,
                'response': response,
                'ground_truth': ground_truth,
                **eval_result
            })
            
            if i < len(test_data):
                print(f"  ⏸️  Pause de {self.delay_between_calls}s...")
                time.sleep(self.delay_between_calls)
        
        return results
    
    def print_summary(self, results: list):
        """Affiche un résumé des résultats"""
        print("\n" + "="*80)
        print("📈 RÉSULTATS DE L'ÉVALUATION")
        print("="*80)
        
        valid_results = [r for r in results if not r['response'].startswith('[ERREUR')]
        failed_count = len(results) - len(valid_results)
        
        if failed_count > 0:
            print(f"\n⚠️  {failed_count}/{len(results)} requêtes ont échoué")
        
        if not valid_results:
            print("\n❌ Aucun résultat valide à analyser")
            return
        
        avg_relevance = sum(r['relevance_score'] for r in valid_results) / len(valid_results)
        context_usage_rate = sum(1 for r in valid_results if r['uses_context']) / len(valid_results)
        avg_length = sum(r['response_length'] for r in valid_results) / len(valid_results)
        
        print(f"\n📊 Métriques moyennes ({len(valid_results)} requêtes valides):")
        print(f"  • Pertinence: {avg_relevance:.2%}")
        print(f"  • Utilisation du contexte: {context_usage_rate:.2%}")
        print(f"  • Longueur moyenne: {avg_length:.0f} mots")
        
        if avg_relevance >= 0.7:
            print("\n✅ Excellent - Le système RAG fonctionne bien")
        elif avg_relevance >= 0.5:
            print("\n⚠️  Acceptable - Des améliorations sont possibles")
        else:
            print("\n❌ Insuffisant - Le système nécessite des ajustements")
    
    def save_results(self, results: list, output_file: str = None):
        """Sauvegarde les résultats"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(__file__).parent / f"evaluation_results_{timestamp}.json"
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(results),
            'successful_queries': len([r for r in results if not r['response'].startswith('[ERREUR')]),
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés : {output_file}")


def main():
    """Fonction principale d'évaluation"""
    try:
        print("🤖 Initialisation du chatbot...")
        chatbot = EventChatbot()
        
        evaluator = SimpleRAGEvaluator(chatbot, delay_between_calls=10.0)
        
        # ✅ Utiliser l'évaluation avancée
        results = evaluator.evaluate(use_advanced=True)
        
        evaluator.print_summary(results)
        evaluator.save_results(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Évaluation interrompue")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'évaluation : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()