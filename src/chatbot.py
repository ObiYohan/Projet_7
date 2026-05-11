from mistralai.client import Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
import pandas as pd
import faiss
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY", "")

class EventChatbot:
    """Chatbot intelligent pour recommander des événements"""
    
    def __init__(self, index_path="faiss_index.idx", chunks_path="chunks_with_embeddings.json"):
        """Initialise le chatbot avec l'index FAISS et les métadonnées"""
        # Charger l'index FAISS
        self.index = faiss.read_index(index_path)
        
        # Charger les métadonnées des chunks
        self.chunks_df = pd.read_json(chunks_path)
        
        # Initialiser le client Mistral
        self.mistral = Mistral(api_key=api_key)
        
        # Template de prompt pour le RAG
        self.system_prompt = """Tu es un assistant spécialisé dans la recommandation d'événements culturels.

                                DATE ACTUELLE : {current_date}

                                Ton rôle est de :
                                1. Analyser les préférences de l'utilisateur ET les contraintes temporelles (dates, "ce week-end", "bientôt", etc.)
                                2. Recommander des événements pertinents basés sur le contexte fourni
                                3. TOUJOURS mentionner les dates des événements dans tes recommandations
                                4. Filtrer les événements passés si l'utilisateur cherche des événements futurs
                                5. Fournir des informations détaillées et personnalisées
                                6. Être enthousiaste et encourageant

                                Contexte des événements disponibles :
                                {context}

                                IMPORTANT : Vérifie systématiquement les dates avant de recommander un événement. 
                                Si aucune date n'est fournie dans le contexte, mentionne-le clairement.

                                Réponds de manière naturelle et conversationnelle en français."""

    def vectorize_query(self, query):
        """Vectorise la requête utilisateur"""
        query_res = self.mistral.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        return np.array([query_res.data[0].embedding])
    
    def search_relevant_events(self, query, k=5):
        """Recherche les événements les plus pertinents"""
        # Vectoriser la requête
        query_embedding = self.vectorize_query(query)
        
        # Rechercher dans FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Récupérer les chunks correspondants
        relevant_chunks = self.chunks_df.iloc[indices[0]]
        
        return relevant_chunks, distances[0]
    
    def format_context(self, chunks_df):
        """Formate les chunks en contexte lisible"""
        context_parts = []
        
        for idx, row in chunks_df.iterrows():
            # Extract temporal metadata if available
            event_info = f"Événement {row['uid']} (partie {row['chunk_id']})"
            
            # Add date information if present in the dataframe
            if 'date_debut' in row and pd.notna(row['firstdate_begin']):
                event_info += f" - Date: {row['firstdate_begin']}"
            if 'date_fin' in row and pd.notna(row['lastdate_end']):
                event_info += f" au {row['lastdate_end']}"
                
            context_parts.append(f"{event_info}:\n{row['text']}\n")
        
        return "\n---\n".join(context_parts)
    
    def generate_response(self, query, context):
        """Génère une réponse augmentée avec Mistral"""
        # Construire le prompt complet
        current_date = datetime.now().strftime("%d/%m/%Y")
        full_prompt = self.system_prompt.format(
            current_date=current_date,
            context=context
            )
        
        # Appeler l'API Mistral pour la génération
        response = self.mistral.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def chat(self, user_query, k=5, show_sources=True):
        """Fonction principale du chatbot"""
        print(f"\n🤖 Recherche d'événements pour : '{user_query}'")
        
        # 1. Rechercher les événements pertinents
        relevant_chunks, distances = self.search_relevant_events(user_query, k=k)
        
        # 2. Formater le contexte
        context = self.format_context(relevant_chunks)
        
        # 3. Générer la réponse
        print("\n💭 Génération de la réponse...")
        response = self.generate_response(user_query, context)
        
        # 4. Afficher les résultats
        print("\n" + "="*80)
        print("RÉPONSE DU CHATBOT")
        print("="*80)
        print(response)
        
        if show_sources:
            print("\n" + "="*80)
            print("SOURCES UTILISÉES")
            print("="*80)
            for idx, (_, row) in enumerate(relevant_chunks.iterrows()):
                print(f"\n📍 Source {idx+1} - Événement {row['uid']} (distance: {distances[idx]:.4f})")
                print(f"   {row['text'][:200]}...")
        
        return {
            "response": response,
            "sources": relevant_chunks,
            "distances": distances
        }
    
    def interactive_mode(self):
        """Mode interactif pour converser avec le chatbot"""
        print("\n" + "="*80)
        print("🎭 CHATBOT D'ÉVÉNEMENTS CULTURELS")
        print("="*80)
        print("Tapez 'quit' ou 'exit' pour quitter\n")
        
        while True:
            user_input = input("Vous : ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir !")
                break
            
            if not user_input:
                continue
            
            try:
                self.chat(user_input)
            except Exception as e:
                print(f"\n❌ Erreur : {e}")
            
            print("\n" + "-"*80 + "\n")


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le chatbot
    chatbot = EventChatbot()
    
    # Exemples de requêtes
    test_queries = [
        "Je cherche un concert de musique classique ce week-end",
        "Quels événements pour enfants sont disponibles le mois prochain ?",
        "Je veux découvrir des expositions d'art contemporain",
        "Recommande-moi des activités en plein air"
    ]
    
    print("\n🧪 TEST DES REQUÊTES")
    print("="*80)
    
    for query in test_queries:
        chatbot.chat(query, k=3, show_sources=False)
        print("\n" + "="*80 + "\n")
    
    # Lancer le mode interactif
    chatbot.interactive_mode()