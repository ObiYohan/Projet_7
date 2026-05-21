from mistralai import Mistral
import pandas as pd
from pathlib import Path
import os
import tiktoken
import sys
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ FORCER L'ENCODAGE UTF-8 POUR LA CONSOLE WINDOWS
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ✅ DÉFINIR LES CHEMINS ABSOLUS
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
INPUT_FILE = DATA_DIR / 'new-data.json'
OUTPUT_FILE = DATA_DIR / 'chunks_with_embeddings.json'

# Initialiser l'encodeur
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Calcule le nombre de tokens dans un texte"""
    return len(encoding.encode(text))

def create_smart_batches(texts, max_tokens_per_batch=8000):
    """Crée des batches en fonction du nombre de tokens"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = count_tokens(text)
        
        if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def create_enriched_text(row):
    """Texte enrichi pour embedding de meilleure qualité"""
    parts = [row['text']]
    
    if pd.notna(row.get('location_city')):
        parts.append(f"Ville: {row['location_city']}")
    
    if pd.notna(row.get('location_name')):
        parts.append(f"Lieu: {row['location_name']}")
    
    if pd.notna(row.get('firstdate_begin')):
        try:
            date_obj = pd.to_datetime(row['firstdate_begin'])
            mois_fr = {
                1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril',
                5: 'mai', 6: 'juin', 7: 'juillet', 8: 'août',
                9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
            }
            date_str = f"{mois_fr[date_obj.month]} {date_obj.year}"
            parts.append(f"Date: {date_str}")
        except:
            pass
    
    return " | ".join(parts)

# Configuration du splitter (au niveau module, c'est OK)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def process_description(row):
    """Découpe uniquement si nécessaire ET PRÉSERVE LES MÉTADONNÉES"""
    text = row["longdescription_fr"]
    token_count = row["token_count"]
    uid = row["uid"]
    
    # Métadonnées temporelles
    firstdate_begin = row["firstdate_begin"]
    firstdate_end = row["firstdate_end"]
    lastdate_begin = row["lastdate_begin"]
    lastdate_end = row["lastdate_end"]
    
    # Métadonnées géographiques
    location_name = row.get("location_name", None)
    location_city = row.get("location_city", None)
    location_postalcode = row.get("location_postalcode", None)
    location_phone = row.get("location_phone", None)
    location_website = row.get("location_website", None)
    
    if token_count <= 512:
        return [{
            "uid": uid,
            "chunk_id": 0,
            "text": text,
            "token_count": token_count,
            "firstdate_begin": firstdate_begin,
            "firstdate_end": firstdate_end,
            "lastdate_begin": lastdate_begin,
            "lastdate_end": lastdate_end,
            "location_name": location_name,
            "location_city": location_city,
            "location_postalcode": location_postalcode,
            "location_phone": location_phone,
            "location_website": location_website
        }]
    else:
        chunks = text_splitter.split_text(text)
        return [{
            "uid": uid,
            "chunk_id": i,
            "text": chunk,
            "token_count": count_tokens(chunk),
            "firstdate_begin": firstdate_begin,
            "firstdate_end": firstdate_end,
            "lastdate_begin": lastdate_begin,
            "lastdate_end": lastdate_end,
            "location_name": location_name,
            "location_city": location_city,
            "location_postalcode": location_postalcode,
            "location_phone": location_phone,
            "location_website": location_website
        } for i, chunk in enumerate(chunks)]

def load_and_filter_data(file_path: str) -> pd.DataFrame:
    """
    Charge et filtre les données d'événements
    
    Args:
        file_path: Chemin vers le fichier JSON
        
    Returns:
        DataFrame filtré avec événements futurs uniquement
    """
    print(f"[INFO] Chargement depuis: {file_path}")
    
    # Load file
    source_data = pd.read_json(file_path)
    
    print(f'Nombre de lignes avant filtrage {source_data.shape[0]}')
    
    # Filtrer les lignes avec des descriptions vides
    source_data = source_data[
        source_data["longdescription_fr"].apply(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
    ]
    
    print(f'Nombre de lignes après filtrage {source_data.shape[0]}')
    
    # Filtrer événements futurs
    current_date = pd.Timestamp.now(tz='UTC')
    print(f"Date actuelle de référence : {current_date}")
    
    source_data["firstdate_begin"] = pd.to_datetime(
        source_data["firstdate_begin"], 
        format='ISO8601',
        utc=True
    )
    source_data["firstdate_end"] = pd.to_datetime(
        source_data["firstdate_end"], 
        format='ISO8601',
        utc=True
    )
    source_data["lastdate_begin"] = pd.to_datetime(
        source_data["lastdate_begin"], 
        format='ISO8601',
        utc=True
    )
    source_data["lastdate_end"] = pd.to_datetime(
        source_data["lastdate_end"], 
        format='ISO8601',
        utc=True
    )
    
    # Filtrer les événements dont la date de FIN est postérieure à aujourd'hui
    source_data = source_data[
        source_data["lastdate_end"] >= current_date
    ]
    
    print(f'Nombre de lignes après filtrage des événements futurs : {source_data.shape[0]}')
    
    return source_data

def create_chunks(source_data):
    """Crée les chunks à partir des données"""
    chunks_list = []
    for _, row in source_data.iterrows():
        chunks_list.extend(process_description(row))
    
    chunks_df = pd.DataFrame(chunks_list)
    
    print(f"Nombre de descriptions originales: {len(source_data)}")
    print(f"Nombre total de chunks: {len(chunks_df)}")
    print(f"Descriptions découpées: {len(chunks_df[chunks_df['chunk_id'] > 0]['uid'].unique())}")
    
    # Enrichir le texte pour embedding
    chunks_df['text_for_embedding'] = chunks_df.apply(create_enriched_text, axis=1)
    
    return chunks_df

def generate_embeddings(chunks_df, api_key=None):
    """Génère les embeddings avec l'API Mistral"""
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY", "")
    
    if not api_key or api_key.strip() == "":
        raise ValueError("MISTRAL_API_KEY non définie")
    
    texts_to_embed = chunks_df["text_for_embedding"].tolist()
    batches = create_smart_batches(texts_to_embed, max_tokens_per_batch=7000)
    
    print(f"\n✓ {len(batches)} batches créés")
    
    all_embeddings = []
    
    with Mistral(api_key=api_key) as mistral:
        for i, batch in enumerate(batches):
            print(f"Batch {i+1}/{len(batches)} - {len(batch)} textes")
            
            try:
                res = mistral.embeddings.create(
                    model="mistral-embed", 
                    inputs=batch
                )
                
                all_embeddings.extend([item.embedding for item in res.data])
                print(f"Total embeddings accumulés: {len(all_embeddings)}")
                
                if i < len(batches) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                if "429" in str(e) or "rate_limited" in str(e).lower():
                    print(f"Rate limit atteint, pause de 60s...")
                    time.sleep(60)
                    res = mistral.embeddings.create(
                        model="mistral-embed", 
                        inputs=batch
                    )
                    all_embeddings.extend([item.embedding for item in res.data])
                    print(f"  ✓ Batch traité après retry")
                else:
                    print(f"  ✗ Erreur: {e}")
                    raise
    
    return all_embeddings

def save_chunks_with_embeddings(chunks_df, embeddings, output_file=None):
    """Sauvegarde les chunks avec embeddings"""
    if output_file is None:
        output_file = OUTPUT_FILE
    
    chunks_df["embedding"] = embeddings
    chunks_df_to_save = chunks_df.drop('text_for_embedding', axis=1)
    chunks_df_to_save.to_json(output_file, orient="records", force_ascii=False, date_format='iso')
    
    print(f"\n[OK] Fichier sauvegardé: {output_file}")

# ✅ CODE D'EXÉCUTION UNIQUEMENT SI LANCÉ DIRECTEMENT
if __name__ == "__main__":
    print("\n[INFO] Démarrage du traitement...")
    
    # Charger et filtrer
    source_data = load_and_filter_data()
    
    # Créer les chunks
    chunks_df = create_chunks(source_data)
    
    # Générer les embeddings
    embeddings = generate_embeddings(chunks_df)
    
    # Sauvegarder
    save_chunks_with_embeddings(chunks_df, embeddings)