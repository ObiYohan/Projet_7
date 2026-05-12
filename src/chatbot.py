from mistralai.client import Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
import pandas as pd
import faiss
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import re


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY", "")

class EventChatbot:
    """Chatbot intelligent pour recommander des événements"""
    
    def __init__(self, index_path=None, chunks_path=None):
        """Initialise le chatbot avec l'index FAISS et les métadonnées"""
        
        # Déterminer le répertoire racine du projet
        project_root = Path(__file__).parent.parent
        
        # Chemins par défaut relatifs au projet
        if index_path is None:
            index_path = project_root / "data/faiss_index.idx"
        if chunks_path is None:
            chunks_path = project_root / "data/chunks_with_embeddings.json"
        
        # Vérifier que les fichiers existent
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index FAISS introuvable : {index_path}\n"
                f"Veuillez exécuter le notebook data_process.ipynb pour générer l'index."
            )
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Fichier chunks introuvable : {chunks_path}\n"
                f"Veuillez exécuter le notebook data_process.ipynb pour générer les chunks."
            )
        
        print(f"📂 Chargement de l'index depuis : {index_path}")
        print(f"📂 Chargement des chunks depuis : {chunks_path}")
        
        # Charger l'index FAISS
        self.index = faiss.read_index(str(index_path))
        
        # Charger les métadonnées des chunks
        self.chunks_df = pd.read_json(chunks_path)
        
        # Initialiser le client Mistral
        self.mistral = Mistral(api_key=api_key)
        
        # Template de prompt pour le RAG
        self.system_prompt = """Tu es un assistant spécialisé dans la recommandation d'événements culturels.

                                DATE ACTUELLE : {current_date}

                                Ton rôle est de :
                                1. Analyser les préférences de l'utilisateur ET les contraintes temporelles (dates, "ce week-end", "bientôt", etc.)
                                2. Recommander UNIQUEMENT les événements présents dans le contexte fourni
                                3. VÉRIFIER que les événements sont FUTURS par rapport à la date actuelle ({current_date})
                                4. NE JAMAIS recommander un événement dont toutes les dates sont passées
                                5. TOUJOURS mentionner les dates EXACTES des événements telles qu'elles apparaissent dans le contexte
                                6. Si une information (date, lieu, horaire) n'est pas dans le contexte, indique clairement "information non disponible"
                                7. Citer les descriptions TEXTUELLEMENT sans les modifier ou les interpréter

                                Contexte des événements disponibles :
                                {context}

                                RÈGLES STRICTES :
                                - Utilise UNIQUEMENT les informations présentes dans le contexte ci-dessus
                                - VÉRIFIE que lastdate_end >= {current_date} avant de recommander
                                - Ne recommande JAMAIS un événement passé
                                - Si aucun événement ne correspond aux critères temporels, dis-le clairement
                                - Cite les événements avec leur UID exact
                                - Reproduis les dates et horaires EXACTEMENT comme ils apparaissent dans le contexte

                                Réponds de manière naturelle et conversationnelle en français."""

    def vectorize_query(self, query):
        """Vectorise la requête utilisateur"""
        query_res = self.mistral.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        return np.array([query_res.data[0].embedding])
    
    def extract_temporal_filter(self, query):
        """Extrait les contraintes temporelles de la requête"""
        query_lower = query.lower()
        
        # Mapping des mois
        months_map = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }
        
        # Détecter le mois demandé
        for month_name, month_num in months_map.items():
            if month_name in query_lower:
                # Déterminer l'année (année actuelle par défaut)
                current_year = datetime.now().year
                
                # Chercher une année dans la requête (ex: "juillet 2026")
                import re
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    year = int(year_match.group(1))
                else:
                    year = current_year
                
                return {
                    'type': 'month',
                    'month': month_num,
                    'year': year
                }
        
        return None

    def search_relevant_events(self, query, k=5):
        """Recherche les événements les plus pertinents"""
        
        # 1. Extraire les filtres temporels
        temporal_filter = self.extract_temporal_filter(query)
        
        # 2. Filtrer d'abord par date si nécessaire
        if temporal_filter and temporal_filter['type'] == 'month':
            print(f"🗓️  Filtre temporel détecté : {temporal_filter['month']}/{temporal_filter['year']}")
            
            # Filtrer les chunks par mois
            filtered_df = self.chunks_df.copy()
            
            # Convertir les dates
            filtered_df['firstdate_begin_dt'] = pd.to_datetime(filtered_df['firstdate_begin'], utc=True)
            filtered_df['lastdate_end_dt'] = pd.to_datetime(filtered_df['lastdate_end'], utc=True)
            
            # Créer les bornes du mois demandé
            start_of_month = pd.Timestamp(
                year=temporal_filter['year'], 
                month=temporal_filter['month'], 
                day=1, 
                tz='UTC'
            )
            
            if temporal_filter['month'] == 12:
                end_of_month = pd.Timestamp(
                    year=temporal_filter['year'] + 1, 
                    month=1, 
                    day=1, 
                    tz='UTC'
                )
            else:
                end_of_month = pd.Timestamp(
                    year=temporal_filter['year'], 
                    month=temporal_filter['month'] + 1, 
                    day=1, 
                    tz='UTC'
                )
            
            # Filtrer : événements qui se déroulent au moins partiellement dans le mois
            mask = (
                (filtered_df['firstdate_begin_dt'] < end_of_month) & 
                (filtered_df['lastdate_end_dt'] >= start_of_month)
            )
            
            filtered_df = filtered_df[mask]
            
            # DÉDUPLICATION PAR UID
            # Grouper par UID et garder tous les chunks de chaque événement unique
            unique_uids = filtered_df['uid'].unique()
            print(f"✓ {len(unique_uids)} événements uniques trouvés en {temporal_filter['month']}/{temporal_filter['year']}")
            
            # Limiter au nombre d'événements demandés
            selected_uids = unique_uids[:k]
            result_df = filtered_df[filtered_df['uid'].isin(selected_uids)]
            
            # Nettoyer les colonnes temporaires
            result_df = result_df.drop(['firstdate_begin_dt', 'lastdate_end_dt'], axis=1)
            
            # Distances fictives (tous également pertinents)
            distances = np.zeros(len(result_df))
            
            return result_df, distances
        
        # 3. Sinon, recherche sémantique classique
        query_embedding = self.vectorize_query(query)
        
        # AUGMENTER k pour avoir plus de candidats
        search_k = min(k * 20, len(self.chunks_df))  # 20x plus de chunks
        distances, indices = self.index.search(query_embedding, search_k)
        
        relevant_chunks = self.chunks_df.iloc[indices[0]].copy()
        relevant_chunks['distance'] = distances[0]
        
        # Filtrer les événements futurs
        current_date = pd.Timestamp.now(tz='UTC')
        relevant_chunks = relevant_chunks[
            pd.to_datetime(relevant_chunks["lastdate_end"], utc=True) >= current_date
        ]
        
        # DÉDUPLICATION : Garder k événements uniques
        unique_events = []
        seen_uids = set()
        
        for _, row in relevant_chunks.sort_values('distance').iterrows():
            uid = row['uid']
            if uid not in seen_uids:
                seen_uids.add(uid)
                # Récupérer TOUS les chunks de cet événement
                event_chunks = relevant_chunks[relevant_chunks['uid'] == uid]
                unique_events.append(event_chunks)
                
                if len(seen_uids) >= k:
                    break
        
        # Reconstruire le DataFrame
        if unique_events:
            result_df = pd.concat(unique_events, ignore_index=True)
            filtered_distances = result_df['distance'].values
            result_df = result_df.drop('distance', axis=1)
        else:
            result_df = pd.DataFrame()
            filtered_distances = np.array([])
        
        print(f"✓ {len(seen_uids)} événements uniques trouvés sur {search_k} candidats")
        
        return result_df, filtered_distances
    
    def format_context(self, chunks_df):
        """Formate les chunks en contexte lisible avec métadonnées temporelles"""
        context_parts = []
        current_date = pd.Timestamp.now(tz='UTC')
        
        # Grouper par UID pour éviter les doublons
        grouped = chunks_df.groupby('uid')
        
        for uid, group in grouped:
            # En-tête de l'événement
            event_info = f"=== ÉVÉNEMENT {uid} ==="
            
            # Prendre les dates du premier chunk (elles sont identiques pour tous les chunks)
            first_row = group.iloc[0]
            
            # Ajouter les dates si disponibles
            date_info = []
            if 'firstdate_begin' in first_row and pd.notna(first_row['firstdate_begin']):
                firstdate = pd.to_datetime(first_row['firstdate_begin'], utc=True)
                date_info.append(f"Première date début: {firstdate.strftime('%d/%m/%Y %H:%M')}")
            if 'firstdate_end' in first_row and pd.notna(first_row['firstdate_end']):
                firstdate_end = pd.to_datetime(first_row['firstdate_end'], utc=True)
                date_info.append(f"Première date fin: {firstdate_end.strftime('%d/%m/%Y %H:%M')}")
            if 'lastdate_begin' in first_row and pd.notna(first_row['lastdate_begin']):
                lastdate = pd.to_datetime(first_row['lastdate_begin'], utc=True)
                date_info.append(f"Dernière date début: {lastdate.strftime('%d/%m/%Y %H:%M')}")
            if 'lastdate_end' in first_row and pd.notna(first_row['lastdate_end']):
                lastdate_end = pd.to_datetime(first_row['lastdate_end'], utc=True)
                date_info.append(f"Dernière date fin: {lastdate_end.strftime('%d/%m/%Y %H:%M')}")
                
                # Vérification de validité temporelle
                if lastdate_end < current_date:
                    date_info.append("⚠️ ÉVÉNEMENT PASSÉ - NE PAS RECOMMANDER")
            
            # Reconstruire la description complète à partir de tous les chunks
            full_description = " ".join(group.sort_values('chunk_id')['text'].tolist())
            
            # Construire le contexte
            context_block = [event_info]
            if date_info:
                context_block.append("DATES: " + " | ".join(date_info))
            context_block.append(f"DESCRIPTION:\n{full_description}")
            
            context_parts.append("\n".join(context_block))
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
        
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
            temperature=0.3,
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
            
            # CRÉER UN MAPPING ENTRE INDEX ET DISTANCE
            distance_map = dict(zip(relevant_chunks.index, distances)) if len(distances) > 0 else {}
            
            # REGROUPER PAR UID POUR ÉVITER LES DOUBLONS
            grouped_sources = relevant_chunks.groupby('uid')
            
            for idx, (uid, group) in enumerate(grouped_sources, 1):
                # Prendre le premier chunk pour les métadonnées
                first_chunk = group.iloc[0]
                
                # Distance moyenne pour cet événement (utiliser le mapping)
                if distance_map:
                    event_distances = [distance_map.get(i, 0.0) for i in group.index if i in distance_map]
                    avg_distance = np.mean(event_distances) if event_distances else 0.0
                else:
                    avg_distance = 0.0
                
                # Reconstruire la description complète
                full_text = " ".join(group.sort_values('chunk_id')['text'].tolist())
                
                # Afficher les dates si disponibles
                date_info = []
                if 'firstdate_begin' in first_chunk and pd.notna(first_chunk['firstdate_begin']):
                    date_info.append(f"Début: {pd.to_datetime(first_chunk['firstdate_begin'], utc=True).strftime('%d/%m/%Y')}")
                if 'lastdate_end' in first_chunk and pd.notna(first_chunk['lastdate_end']):
                    date_info.append(f"Fin: {pd.to_datetime(first_chunk['lastdate_end'], utc=True).strftime('%d/%m/%Y')}")
                
                print(f"\n📍 Source {idx} - Événement {uid} (distance: {avg_distance:.4f})")
                if date_info:
                    print(f"   📅 {' | '.join(date_info)}")
                print(f"   {full_text[:300]}...")
        
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
    # test_queries = [
    #     "Je cherche un concert de musique classique ce week-end",
    #     "Quels événements pour enfants sont disponibles le mois prochain ?",
    #     "Je veux découvrir des expositions d'art contemporain",
    #     "Recommande-moi des activités en plein air"
    # ]
    
    # print("\n TEST DES REQUÊTES")
    # print("="*80)
    
    # for query in test_queries:
    #     chatbot.chat(query, k=3, show_sources=False)
    #     print("\n" + "="*80 + "\n")
    
    # Lancer le mode interactif
    chatbot.interactive_mode()