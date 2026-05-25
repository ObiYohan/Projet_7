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
import unicodedata


load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY", "")

class EventChatbot:
    """Chatbot intelligent pour recommander des événements"""
    
    def __init__(self, index_path=None, chunks_path=None):
        """Initialise le chatbot avec l'index FAISS et les métadonnées"""
        
        project_root = Path(__file__).parent.parent
        
        if index_path is None:
            index_path = project_root / "data/faiss_index.idx"
        if chunks_path is None:
            chunks_path = project_root / "data/chunks_with_embeddings.json"
        
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
        
        self.index = faiss.read_index(str(index_path))
        self.chunks_df = pd.read_json(chunks_path)
        
        date_columns = ['firstdate_begin', 'firstdate_end', 'lastdate_begin', 'lastdate_end']
        for col in date_columns:
            if col in self.chunks_df.columns:
                self.chunks_df[col] = pd.to_datetime(
                    self.chunks_df[col], 
                    unit='ms',
                    utc=True, 
                    errors='coerce'
                )
        
        current_date = pd.Timestamp.now(tz='UTC')
        future_events = self.chunks_df[self.chunks_df['lastdate_end'] >= current_date]
        
        print(f"\n📊 Statistiques des données :")
        print(f"  - Total chunks : {len(self.chunks_df)}")
        print(f"  - Événements futurs : {len(future_events['uid'].unique())} événements uniques")
        print(f"  - Date la plus proche : {self.chunks_df['firstdate_begin'].min()}")
        print(f"  - Date la plus lointaine : {self.chunks_df['lastdate_end'].max()}")

        self.available_cities = self._extract_available_cities()
        self.available_postal_codes = self._extract_available_postal_codes()
        
        print(f"📍 {len(self.available_cities)} villes détectées dans les données")
        print(f"📮 {len(self.available_postal_codes)} codes postaux détectés")
        
        self.mistral = Mistral(api_key=api_key)
        
        self.system_prompt = """Tu es un assistant spécialisé dans la recommandation d'événements culturels.

                                DATE ACTUELLE : {current_date}

                                Ton rôle est de :
                                1. Analyser les préférences de l'utilisateur ET les contraintes temporelles (dates, "ce week-end", "bientôt", etc.)
                                2. Recommander UNIQUEMENT les événements présents dans le contexte fourni
                                3. VÉRIFIER que les événements sont FUTURS par rapport à la date actuelle ({current_date})
                                4. NE JAMAIS recommander un événement dont toutes les dates sont passées ou absentes
                                5. TOUJOURS mentionner les dates EXACTES des événements telles qu'elles apparaissent dans le contexte
                                6. Si une information (date, lieu, horaire) n'est pas dans le contexte, indique clairement "information non disponible"
                                7. Mentionne TOUJOURS la ville et le lieu

                                Contexte des événements disponibles :
                                {context}

                                RÈGLES STRICTES — ANTI-HALLUCINATION :
                                - Utilise UNIQUEMENT les informations présentes dans le contexte ci-dessus
                                - VÉRIFIE que lastdate_end >= {current_date} avant de recommander
                                - Ne recommande JAMAIS un événement passé
                                - Si aucun événement ne correspond aux critères temporels, dis-le clairement
                                - Reproduis les dates et horaires EXACTEMENT comme ils apparaissent dans le contexte — ne les recalcule pas, ne les déduis pas
                                - Chaque événement cité doit correspondre à un bloc === ÉVÉNEMENT === distinct dans le contexte — ne fusionne JAMAIS deux événements en un
                                - Ne transfère JAMAIS un détail (artiste, horaire, tarif) d'un événement à un autre
                                - Si deux événements ont lieu le même jour, présente-les séparément avec leurs informations propres
                                - Pour les listes (groupes, artistes, activités), reproduis EXACTEMENT la liste du contexte — ne l'abrège pas et ne l'invente pas
                                - N'invente JAMAIS de titre pour un événement : utilise le texte exact du contexte

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
        
        months_map = {
            'janvier': 1, 'février': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'août': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'décembre': 12
        }
        
        for month_name, month_num in months_map.items():
            if month_name in query_lower:
                current_year = datetime.now().year
                year_match = re.search(r'\b(20\d{2})\b', query)
                year = int(year_match.group(1)) if year_match else current_year
                
                return {
                    'type': 'month',
                    'month': month_num,
                    'year': year
                }
        
        return None
    
    def _extract_available_cities(self):
        """Extrait toutes les villes uniques présentes dans les données"""
        if 'location_city' not in self.chunks_df.columns:
            return {}
        
        cities = self.chunks_df['location_city'].dropna().unique()
        normalized_cities = {}
        
        for city in cities:
            normalized = unicodedata.normalize('NFD', city.lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            normalized_cities[normalized] = city
        
        return normalized_cities
    
    def _extract_available_postal_codes(self):
        """Extrait tous les codes postaux uniques présents dans les données"""
        if 'location_postalcode' not in self.chunks_df.columns:
            return set()
        
        postal_codes = self.chunks_df['location_postalcode'].dropna().unique()
        return set(str(code) for code in postal_codes)

    def extract_location_filter(self, query):
        """
        Extrait les contraintes géographiques de la requête.

        FIX #5 : Ajout de la reconnaissance par département (ex: "Vaucluse" → 84xxx,
        "Var" → 83xxx). Sans ce mapping, une requête "Vaucluse" ne filtrait pas
        les codes postaux 84xxx et remontait des événements hors département.
        """
        query_normalized = unicodedata.normalize('NFD', query.lower())
        query_normalized = ''.join(c for c in query_normalized if unicodedata.category(c) != 'Mn')

        # Mapping département → préfixe de code postal
        DEPARTMENT_MAP = {
            'vaucluse': '84',
            'bouches-du-rhone': '13',
            'bouches du rhone': '13',
            'var': '83',
            'alpes-de-haute-provence': '04',
            'alpes de haute provence': '04',
            'hautes-alpes': '05',
            'hautes alpes': '05',
            'alpes-maritimes': '06',
            'alpes maritimes': '06',
        }

        # 1. Chercher un département dans la requête
        for dept_normalized, dept_prefix in DEPARTMENT_MAP.items():
            dept_n = unicodedata.normalize('NFD', dept_normalized)
            dept_n = ''.join(c for c in dept_n if unicodedata.category(c) != 'Mn')
            if dept_n in query_normalized:
                print(f"🗺️  Département détecté : {dept_normalized} → codes postaux {dept_prefix}xxx")
                return {'type': 'department_prefix', 'value': dept_prefix}

        # 2. Chercher une ville dans la requête
        for normalized_city, original_city in self.available_cities.items():
            if normalized_city in query_normalized:
                print(f"🏙️  Ville détectée : {original_city}")
                return {'type': 'city', 'value': original_city}

        # 3. Chercher un code postal explicite
        postal_match = re.search(r'\b(\d{5})\b', query)
        if postal_match:
            postal_code = postal_match.group(1)
            if postal_code in self.available_postal_codes:
                print(f"📮 Code postal détecté : {postal_code}")
                return {'type': 'postal', 'value': postal_code}
            else:
                print(f"⚠️  Code postal {postal_code} non trouvé dans les données")

        return None

    def _apply_temporal_filter(self, df, temporal_filter):
        """Applique le filtre temporel sur un DataFrame."""
        month = temporal_filter['month']
        year = temporal_filter['year']
        print(f"🗓️  Filtre temporel : {month}/{year}")
        
        start_of_month = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        end_of_month = (
            pd.Timestamp(year=year + 1, month=1, day=1, tz='UTC')
            if month == 12
            else pd.Timestamp(year=year, month=month + 1, day=1, tz='UTC')
        )
        
        mask = (
            (df['firstdate_begin'] < end_of_month) & 
            (df['lastdate_end'] >= start_of_month)
        )
        return df[mask]

    def _apply_location_filter(self, df, location_filter):
        """
        Applique le filtre géographique sur un DataFrame.

        FIX #5 : Gestion du type 'department_prefix' pour filtrer par préfixe
        de code postal (ex: '84' filtre tous les codes 84000-84999).
        """
        print(f"📍 Filtre géographique : {location_filter['value']}")

        if location_filter['type'] == 'city':
            return df[
                df['location_city'].str.contains(
                    location_filter['value'], case=False, na=False
                )
            ]
        elif location_filter['type'] == 'postal':
            return df[df['location_postalcode'] == location_filter['value']]
        elif location_filter['type'] == 'department_prefix':
            prefix = location_filter['value']
            postal_str = df['location_postalcode'].astype(str).str.zfill(5)
            return df[postal_str.str.startswith(prefix)]

        return df

    def _semantic_rank(self, df, query_embedding, k):
        """
        FIX #1 : Classement sémantique sur un sous-ensemble filtré.
        Au lieu de prendre les k premiers arbitrairement, on reconstruit
        un index FAISS temporaire pour classer par pertinence.
        """
        if df.empty:
            return pd.DataFrame(), np.array([])
        
        filtered_indices = df.index.tolist()
        
        # Construire l'index temporaire
        filtered_embeddings = np.array([
            self.chunks_df.loc[idx, 'embedding'] for idx in filtered_indices
        ]).astype('float32')
        
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
        
        search_k = min(k * 10, len(df))
        distances, local_indices = temp_index.search(query_embedding, search_k)
        
        global_indices = [filtered_indices[i] for i in local_indices[0]]
        result_chunks = self.chunks_df.loc[global_indices].copy()
        result_chunks['distance'] = distances[0]
        
        # Déduplication par UID en gardant le meilleur score
        unique_events = []
        seen_uids = set()
        
        for _, row in result_chunks.sort_values('distance').iterrows():
            uid = row['uid']
            if uid not in seen_uids:
                seen_uids.add(uid)
                event_chunks = result_chunks[result_chunks['uid'] == uid]
                unique_events.append(event_chunks)
                if len(seen_uids) >= k:
                    break
        
        if not unique_events:
            return pd.DataFrame(), np.array([])
        
        result_df = pd.concat(unique_events, ignore_index=True)
        final_distances = result_df['distance'].values
        result_df = result_df.drop('distance', axis=1)
        
        return result_df, final_distances

    def search_relevant_events(self, query, k=10):
        """
        Recherche les événements les plus pertinents.
        
        """
        temporal_filter = self.extract_temporal_filter(query)
        location_filter = self.extract_location_filter(query)
        
        # Partir des événements futurs uniquement
        current_date = pd.Timestamp.now(tz='UTC')
        filtered_df = self.chunks_df[self.chunks_df['lastdate_end'] >= current_date].copy()
        print(f"📊 Événements futurs disponibles : {len(filtered_df['uid'].unique())} événements uniques")
        
        # FIX #1 : Appliquer les deux filtres de façon cumulative (et non exclusive)
        if temporal_filter and temporal_filter['type'] == 'month':
            filtered_df = self._apply_temporal_filter(filtered_df, temporal_filter)
        
        if location_filter:
            filtered_df = self._apply_location_filter(filtered_df, location_filter)
        
        unique_uids = filtered_df['uid'].unique()
        print(f"✓ {len(unique_uids)} événements après filtrage")
        
        if len(unique_uids) == 0:
            print("⚠️  Aucun événement trouvé après filtrage")
            return pd.DataFrame(), np.array([])
        
        # FIX #2 : Toujours classer sémantiquement, même après filtrage
        query_embedding = self.vectorize_query(query)
        return self._semantic_rank(filtered_df, query_embedding, k)
    
    def format_context(self, chunks_df):
        """Formate les chunks en contexte lisible avec métadonnées temporelles"""
        context_parts = []
        
        if chunks_df.empty:
            return "Aucun événement trouvé."
        
        if 'uid' not in chunks_df.columns:
            print(f"⚠️ Colonnes disponibles : {chunks_df.columns.tolist()}")
            return "Erreur : colonne 'uid' manquante dans les données."
        
        grouped = chunks_df.groupby('uid')
        
        for uid, group in grouped:
            event_info = f"=== ÉVÉNEMENT {uid} ==="
            first_row = group.iloc[0]
            
            # FIX #6 : Distinguer événement ponctuel / multi-jours pour éviter
            # que le LLM confonde firstdate_begin et lastdate_end comme une seule date.
            date_info = []
            fdb = pd.to_datetime(first_row.get('firstdate_begin'), utc=True, errors='coerce') if 'firstdate_begin' in first_row else None
            fde = pd.to_datetime(first_row.get('firstdate_end'), utc=True, errors='coerce') if 'firstdate_end' in first_row else None
            ldb = pd.to_datetime(first_row.get('lastdate_begin'), utc=True, errors='coerce') if 'lastdate_begin' in first_row else None
            lde = pd.to_datetime(first_row.get('lastdate_end'), utc=True, errors='coerce') if 'lastdate_end' in first_row else None

            is_multiday = (
                fdb is not None and lde is not None and pd.notna(fdb) and pd.notna(lde)
                and fdb.date() != lde.date()
            )

            if is_multiday:
                # Événement étalé sur plusieurs jours : rendre la structure explicite
                date_info.append(f"⚠️ ÉVÉNEMENT MULTI-JOURS")
                if pd.notna(fdb):
                    date_info.append(f"Jour de début: {fdb.strftime('%d/%m/%Y')} à {fdb.strftime('%H:%M')}")
                if pd.notna(lde):
                    date_info.append(f"Jour de fin: {lde.strftime('%d/%m/%Y')} à {lde.strftime('%H:%M')}")
                date_info.append(f"⚠️ NE PAS CONFONDRE la date de début et la date de fin")
            else:
                if pd.notna(fdb):
                    date_info.append(f"Date début: {fdb.strftime('%d/%m/%Y %H:%M')}")
                if pd.notna(fde):
                    date_info.append(f"Date fin: {fde.strftime('%d/%m/%Y %H:%M')}")
                if ldb is not None and pd.notna(ldb) and ldb != fdb:
                    date_info.append(f"Dernière occurrence début: {ldb.strftime('%d/%m/%Y %H:%M')}")
                if lde is not None and pd.notna(lde) and lde != fde:
                    date_info.append(f"Dernière occurrence fin: {lde.strftime('%d/%m/%Y %H:%M')}")

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
            
            if 'chunk_id' in group.columns:
                full_description = " ".join(group.sort_values('chunk_id')['text'].tolist())
            else:
                full_description = " ".join(group['text'].tolist())
            
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
        current_date = datetime.now().strftime("%d/%m/%Y")
        full_prompt = self.system_prompt.format(
            current_date=current_date,
            context=context
        )
        
        response = self.mistral.chat.complete(
            model="mistral-small-latest",
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
        
        relevant_chunks, distances = self.search_relevant_events(user_query, k=k)
        context = self.format_context(relevant_chunks)
        
        print("\n💭 Génération de la réponse...")
        response = self.generate_response(user_query, context)
        
        print("\n" + "="*80)
        print("RÉPONSE DU CHATBOT")
        print("="*80)
        print(response)
        
        if show_sources:
            print("\n" + "="*80)
            print("SOURCES UTILISÉES")
            print("="*80)
            
            distance_map = dict(zip(relevant_chunks.index, distances)) if len(distances) > 0 else {}
            grouped_sources = relevant_chunks.groupby('uid')
            
            for idx, (uid, group) in enumerate(grouped_sources, 1):
                first_chunk = group.iloc[0]
                
                if distance_map:
                    event_distances = [distance_map.get(i, 0.0) for i in group.index if i in distance_map]
                    avg_distance = np.mean(event_distances) if event_distances else 0.0
                else:
                    avg_distance = 0.0
                
                full_text = " ".join(group.sort_values('chunk_id')['text'].tolist())
                
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