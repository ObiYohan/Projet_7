from mistralai import Mistral
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
        
        # ✅ CORRECTION : Charger avec conversion automatique des dates
        self.chunks_df = pd.read_json(chunks_path)
        
        # ✅ FORCER la conversion des colonnes de dates (timestamps en millisecondes)
        date_columns = ['firstdate_begin', 'firstdate_end', 'lastdate_begin', 'lastdate_end']
        for col in date_columns:
            if col in self.chunks_df.columns:
                # Convertir les timestamps (millisecondes) en datetime UTC
                self.chunks_df[col] = pd.to_datetime(
                    self.chunks_df[col], 
                    unit='ms',  # ✅ Timestamps en millisecondes
                    utc=True, 
                    errors='coerce'
                )
        
        # ✅ VÉRIFICATION : Afficher les statistiques de dates
        current_date = pd.Timestamp.now(tz='UTC')
        future_events = self.chunks_df[self.chunks_df['lastdate_end'] >= current_date]
        
        print(f"\n📊 Statistiques des données :")
        print(f"  - Total chunks : {len(self.chunks_df)}")
        print(f"  - Événements futurs : {len(future_events['uid'].unique())} événements uniques")
        print(f"  - Date la plus proche : {self.chunks_df['firstdate_begin'].min()}")
        print(f"  - Date la plus lointaine : {self.chunks_df['lastdate_end'].max()}")

        # Extraire dynamiquement toutes les villes disponibles
        self.available_cities = self._extract_available_cities()
        self.available_postal_codes = self._extract_available_postal_codes()
        
        print(f"📍 {len(self.available_cities)} villes détectées dans les données")
        print(f"📮 {len(self.available_postal_codes)} codes postaux détectés")
        
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
    
    def _extract_available_cities(self):
        """Extrait toutes les villes uniques présentes dans les données"""
        if 'location_city' not in self.chunks_df.columns:
            return {}  # Retourner un dictionnaire vide, pas un set
        
        cities = self.chunks_df['location_city'].dropna().unique()
        
        # Normaliser les noms (minuscules, sans accents pour la recherche)
        import unicodedata
        normalized_cities = {}
        
        for city in cities:
            # Normaliser pour la recherche
            normalized = unicodedata.normalize('NFD', city.lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            normalized_cities[normalized] = city  # Garder l'original
        
        return normalized_cities  # ✅ Retourne un dictionnaire
    
    def _extract_available_postal_codes(self):
        """Extrait tous les codes postaux uniques présents dans les données"""
        if 'location_postalcode' not in self.chunks_df.columns:
            return set()
        
        postal_codes = self.chunks_df['location_postalcode'].dropna().unique()
        
        # Convertir en set pour recherche rapide
        return set(str(code) for code in postal_codes)

    def extract_location_filter(self, query):
        """Extrait les contraintes géographiques de la requête"""
        import unicodedata
        
        # Normaliser la requête
        query_normalized = unicodedata.normalize('NFD', query.lower())
        query_normalized = ''.join(c for c in query_normalized if unicodedata.category(c) != 'Mn')
        
        # 1. Chercher une ville dans la requête
        # self.available_cities est un DICTIONNAIRE {normalized: original}
        for normalized_city, original_city in self.available_cities.items():
            if normalized_city in query_normalized:
                print(f"🏙️  Ville détectée : {original_city}")
                return {'type': 'city', 'value': original_city}
        
        # 2. Chercher un code postal (tous les codes postaux français : 5 chiffres)
        import re
        postal_match = re.search(r'\b(\d{5})\b', query)
        if postal_match:
            postal_code = postal_match.group(1)
            # Vérifier que ce code postal existe dans nos données
            if postal_code in self.available_postal_codes:
                print(f"📮 Code postal détecté : {postal_code}")
                return {'type': 'postal', 'value': postal_code}
            else:
                print(f"⚠️  Code postal {postal_code} non trouvé dans les données")
        
        return None

    def search_relevant_events(self, query, k=10):
        """Recherche les événements les plus pertinents"""
        
        # Extraire les filtres temporels et géographiques
        temporal_filter = self.extract_temporal_filter(query)
        location_filter = self.extract_location_filter(query)
        
        # Partir du DataFrame complet
        filtered_df = self.chunks_df.copy()
        
        # Filtrer d'abord par date si nécessaire
        if temporal_filter and temporal_filter['type'] == 'month':
            print(f"🗓️  Filtre temporel détecté : {temporal_filter['month']}/{temporal_filter['year']}")
            
            # ✅ Les dates sont déjà en datetime, pas besoin de conversion
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
                (filtered_df['firstdate_begin'] < end_of_month) & 
                (filtered_df['lastdate_end'] >= start_of_month)
            )
            
            filtered_df = filtered_df[mask]
            
            # DÉDUPLICATION PAR UID
            unique_uids = filtered_df['uid'].unique()
            print(f"✓ {len(unique_uids)} événements uniques trouvés en {temporal_filter['month']}/{temporal_filter['year']}")
            
            # Limiter au nombre d'événements demandés
            selected_uids = unique_uids[:k]
            result_df = filtered_df[filtered_df['uid'].isin(selected_uids)]
            
            # Distances fictives (tous également pertinents)
            distances = np.zeros(len(result_df))
            
            return result_df, distances
        
        # Filtrer par localisation si nécessaire
        if location_filter:
            print(f"📍 Filtre géographique détecté : {location_filter['value']}")
            
            if location_filter['type'] == 'city':
                filtered_df = filtered_df[
                    filtered_df['location_city'].str.contains(
                        location_filter['value'], 
                        case=False, 
                        na=False
                    )
                ]
            elif location_filter['type'] == 'postal':
                filtered_df = filtered_df[
                    filtered_df['location_postalcode'] == location_filter['value']
                ]
            
            # DÉDUPLICATION PAR UID après filtrage géographique
            unique_uids = filtered_df['uid'].unique()
            print(f"✓ {len(unique_uids)} événements uniques trouvés pour {location_filter['value']}")
            
            # Limiter au nombre d'événements demandés
            selected_uids = unique_uids[:k]
            result_df = filtered_df[filtered_df['uid'].isin(selected_uids)]
            
            # Distances fictives
            distances = np.zeros(len(result_df))
            
            return result_df, distances
        
        # Recherche sémantique classique
        query_embedding = self.vectorize_query(query)
        
        # Rechercher dans FAISS
        search_k = min(k * 20, len(self.chunks_df))
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Récupérer les chunks candidats
        relevant_chunks = self.chunks_df.iloc[indices[0]].copy()
        relevant_chunks['distance'] = distances[0]
        
        print(f"📊 Chunks candidats avant filtrage : {len(relevant_chunks)}")
        
        # ✅ Filtrer les événements futurs (dates déjà converties)
        current_date = pd.Timestamp.now(tz='UTC')
        future_mask = relevant_chunks['lastdate_end'] >= current_date
        relevant_chunks = relevant_chunks[future_mask]
        
        print(f"📊 Chunks après filtrage temporel : {len(relevant_chunks)}")
        
        # Déduplication par UID
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
        
        # Vérifier que le DataFrame n'est pas vide et contient la colonne 'uid'
        if chunks_df.empty:
            return "Aucun événement trouvé."
        
        if 'uid' not in chunks_df.columns:
            print(f"⚠️ Colonnes disponibles : {chunks_df.columns.tolist()}")
            return "Erreur : colonne 'uid' manquante dans les données."
        
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

            location_info = []
            if 'location_name' in first_row and pd.notna(first_row['location_name']):
                location_info.append(f"Lieu: {first_row['location_name']}")
            if 'location_city' in first_row and pd.notna(first_row['location_city']):
                location_info.append(f"Ville: {first_row['location_city']}")
            if 'location_postalcode' in first_row and pd.notna(first_row['location_postalcode']):
                location_info.append(f"Code postal: {first_row['location_postalcode']}")
            if 'location_phone' in first_row and pd.notna(first_row['location_phone']):
                location_info.append(f"Téléphone: {first_row['location_phone']}")
            if 'location_website' in first_row and pd.notna(first_row['location_website']):
                location_info.append(f"Site web: {first_row['location_website']}")
            
            # Reconstruire la description complète à partir de tous les chunks
            if 'chunk_id' in group.columns:
                full_description = " ".join(group.sort_values('chunk_id')['text'].tolist())
            else:
                full_description = " ".join(group['text'].tolist())
            
            # Construire le contexte
            context_block = [event_info]
            if date_info:
                context_block.append("DATES: " + " | ".join(date_info))
            if location_info:
                context_block.append("LOCALISATION: " + " | ".join(location_info))
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
# if __name__ == "__main__":
    # Initialiser le chatbot
    # chatbot = EventChatbot()
    
    # # Exemples de requêtes
    # test_queries = [
    #     "Quels concerts sont disponibles en juillet 2026 ?",
    #     "Je cherche des activités pour enfants le premier week-end de juin",
    #     "Je veux découvrir des expositions d'art contemporain",
    #     "Recommande-moi des activités en plein air dans le Vaucluse"
    # ]
    
    # print("\n TEST DES REQUÊTES")
    # print("="*80)
    
    # for query in test_queries:
    #     chatbot.chat(query, k=5, show_sources=False)
    #     print("\n" + "="*80 + "\n")
    
    # Lancer le mode interactif
    # chatbot.interactive_mode()