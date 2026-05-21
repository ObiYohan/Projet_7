from mistralai import Mistral
import os
import pandas as pd 
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import sys
from pathlib import Path

# ✅ FORCER L'ENCODAGE UTF-8 POUR LA CONSOLE WINDOWS
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ✅ DÉFINIR LES CHEMINS ABSOLUS
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
INPUT_FILE = DATA_DIR / 'new-data.json'
OUTPUT_FILE = DATA_DIR / 'chunks_with_embeddings.json'

# Initialiser l'encodeur (utilisez le modèle approprié)
encoding = tiktoken.get_encoding("cl100k_base")

# Calculer les tokens pour une description
def count_tokens(text):
    return len(encoding.encode(text))

# Load file
print(f"[INFO] Chargement depuis: {INPUT_FILE}")
source_data = pd.read_json(INPUT_FILE)

print(f'Nombre de lignes avant filtrage {source_data.shape[0]}')

# Filtrer les lignes de source_data avec des descriptions vide ou avec uniquement des espaces
source_data = source_data[source_data["longdescription_fr"].apply(lambda x: isinstance(x, str) and bool(x.strip()))]

print(f'Nombre de lignes après filtrage {source_data.shape[0]}')

long_desc_list = source_data["longdescription_fr"].tolist()

print(f'Nombre de lignes dans long_desc_list {len(long_desc_list)}')

# Filtrer événements
# Convert both dates to UTC timezone for comparison
current_date = pd.Timestamp.now(tz='UTC')
print(f"Date actuelle de référence : {current_date}")

# Convertir les colonnes de dates en datetime AVANT le filtrage
source_data["firstdate_begin"] = pd.to_datetime(source_data["firstdate_begin"], utc=True)
source_data["firstdate_end"] = pd.to_datetime(source_data["firstdate_end"], utc=True)
source_data["lastdate_begin"] = pd.to_datetime(source_data["lastdate_begin"], utc=True)
source_data["lastdate_end"] = pd.to_datetime(source_data["lastdate_end"], utc=True)

# Filtrer les événements dont la date de FIN est postérieure à aujourd'hui
source_data = source_data[
    source_data["lastdate_end"] >= current_date
]

print(f'Nombre de lignes après filtrage des événements futurs : {source_data.shape[0]}')

# Vérification : afficher quelques dates pour contrôle
print("\n=== Vérification des dates ===")
print(source_data[["uid", "firstdate_begin", "lastdate_end"]].head(10))

print("\n=== VÉRIFICATION DES DATES ===")
current_date = pd.Timestamp.now(tz='UTC')

# Vérifier s'il reste des événements passés
past_events = source_data[
    pd.to_datetime(source_data["lastdate_end"], utc=True) < current_date
]

if len(past_events) > 0:
    print(f"[!] ATTENTION : {len(past_events)} événements passés détectés !")
    print(past_events[["uid", "firstdate_begin", "lastdate_end"]].head())
else:
    print(f"[OK] Tous les {len(source_data)} événements sont futurs")

# Statistiques temporelles
print(f"\nDate la plus proche : {source_data['firstdate_begin'].min()}")
print(f"Date la plus lointaine : {source_data['lastdate_end'].max()}")

# Analyser les tokens
source_data["token_count"] = source_data["longdescription_fr"].apply(count_tokens)

# Vérifier la disponibilité des informations de contact
print("\n=== ANALYSE DES DONNÉES DE CONTACT ===")

contact_columns = ['location_phone', 'location_website']
for col in contact_columns:
    if col in source_data.columns:
        non_null = source_data[col].notna().sum()
        percentage = (non_null / len(source_data)) * 100
        print(f"[OK] {col}: {non_null}/{len(source_data)} ({percentage:.1f}%)")
        
        # Afficher quelques exemples
        if non_null > 0:
            print(f"  Exemples: {source_data[col].dropna().head(3).tolist()}")
    else:
        print(f"[X] {col}: Colonne absente")

print("\n=== STATISTIQUES DE LOCALISATION ===")
location_columns = ['location_name', 'location_address', 'location_city', 'location_postalcode']
for col in location_columns:
    if col in source_data.columns:
        non_null = source_data[col].notna().sum()
        percentage = (non_null / len(source_data)) * 100
        print(f"[OK] {col}: {non_null}/{len(source_data)} ({percentage:.1f}%)")

# Configuration du splitter
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
    
    # ✅ AJOUTER TOUTES LES MÉTADONNÉES NÉCESSAIRES
    firstdate_begin = row["firstdate_begin"]
    firstdate_end = row["firstdate_end"]
    lastdate_begin = row["lastdate_begin"]
    lastdate_end = row["lastdate_end"]
    
    # ✅ NOUVELLES MÉTADONNÉES GÉOGRAPHIQUES
    location_name = row.get("location_name", None)
    location_city = row.get("location_city", None)
    location_postalcode = row.get("location_postalcode", None)
    location_phone = row.get("location_phone", None)
    location_website = row.get("location_website", None)
    
    if token_count <= 512:
        # Pas de chunking nécessaire
        return [{
            "uid": uid,
            "chunk_id": 0,
            "text": text,
            "token_count": token_count,
            "firstdate_begin": firstdate_begin,
            "firstdate_end": firstdate_end,
            "lastdate_begin": lastdate_begin,
            "lastdate_end": lastdate_end,
            # ✅ AJOUTER LES MÉTADONNÉES GÉOGRAPHIQUES
            "location_name": location_name,
            "location_city": location_city,
            "location_postalcode": location_postalcode,
            "location_phone": location_phone,
            "location_website": location_website
        }]
    else:
        # Chunking pour les longues descriptions
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
            # ✅ RÉPÉTER POUR CHAQUE CHUNK
            "location_name": location_name,
            "location_city": location_city,
            "location_postalcode": location_postalcode,
            "location_phone": location_phone,
            "location_website": location_website
        } for i, chunk in enumerate(chunks)]

# Appliquer le traitement
chunks_list = []
for _, row in source_data.iterrows():
    chunks_list.extend(process_description(row))

# Créer un nouveau DataFrame avec les chunks
chunks_df = pd.DataFrame(chunks_list)

print(f"Nombre de descriptions originales: {len(source_data)}")
print(f"Nombre total de chunks: {len(chunks_df)}")
print(f"Descriptions découpées: {len(chunks_df[chunks_df['chunk_id'] > 0]['uid'].unique())}")

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
    
    # Ajouter localisation si disponible
    if pd.notna(row.get('location_city')):
        parts.append(f"Ville: {row['location_city']}")
    
    if pd.notna(row.get('location_name')):
        parts.append(f"Lieu: {row['location_name']}")
    
    # Ajouter date si disponible
    if pd.notna(row.get('firstdate_begin')):
        try:
            date_obj = pd.to_datetime(row['firstdate_begin'])
            # Traduire le mois en français
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

chunks_df['text_for_embedding'] = chunks_df.apply(create_enriched_text, axis=1)

print("\n=== VÉRIFICATION ===")
print(f"✓ Colonne créée : {len(chunks_df['text_for_embedding'])} entrées")
print(f"\nExemple :")
print(chunks_df[['text', 'text_for_embedding']].iloc[0])

texts_to_embed = chunks_df["text_for_embedding"].tolist()
batches = create_smart_batches(texts_to_embed, max_tokens_per_batch=7000)
print(f"\n✓ {len(batches)} batches créés")

batches = create_smart_batches(texts_to_embed, max_tokens_per_batch=7000)
print(f"Nombre de batches créés: {len(batches)}")

all_embeddings = []

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as mistral:
    
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}/{len(batches)} - {len(batch)} textes")
        
        try:
            res = mistral.embeddings.create(
                model="mistral-embed", 
                inputs=batch
            )
            
            all_embeddings.extend([item.embedding for item in res.data])
            print(f"Total embeddings accumulés: {len(all_embeddings)}")
            
            # Pause entre les batches pour respecter le rate limit
            if i < len(batches) - 1:  # Pas de pause après le dernier batch
                wait_time = 2  # Secondes entre chaque batch
                print(f"Pause de {wait_time}s...")
                time.sleep(wait_time)
                
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                print(f"Rate limit atteint, pause de 60s...")
                time.sleep(60)
                # Réessayer le batch
                print(f"Nouvelle tentative pour le batch {i+1}")
                res = mistral.embeddings.create(
                    model="mistral-embed", 
                    inputs=batch
                )
                all_embeddings.extend([item.embedding for item in res.data])
                print(f"  ✓ Batch traité après retry")
            else:
                print(f"  ✗ Erreur: {e}")
                raise

# Vérification avant assignation
print(f"\nVérification:")
print(f"  - Nombre de chunks: {len(chunks_df)}")
print(f"  - Nombre d'embeddings: {len(all_embeddings)}")

# Ajouter les embeddings au DataFrame
chunks_df["embedding"] = all_embeddings

# ✅ Supprimer text_for_embedding avant sauvegarde (on n'en a plus besoin)
chunks_df_to_save = chunks_df.drop('text_for_embedding', axis=1)

# Sauvegarder
chunks_df_to_save.to_json(OUTPUT_FILE, orient="records", force_ascii=False, date_format='iso')

if __name__ == "__main__":
    print("\n[OK] TRAITEMENT TERMINÉ")
    print(f"[INFO] Statistiques finales:")
    print(f"  - Événements traités: {len(source_data)}")
    print(f"  - Chunks créés: {len(chunks_df)}")
    print(f"  - Embeddings générés: {len(all_embeddings)}")
    print(f"  - Fichier sauvegardé: {OUTPUT_FILE}")