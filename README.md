# Chatbot API - Événements Publics OpenAgenda

API de chatbot intelligent pour rechercher et recommander des événements publics dans la région PACA, utilisant RAG (Retrieval-Augmented Generation) avec Mistral AI.

## 🎯 Objectifs du projet

Ce projet vise à créer un assistant conversationnel intelligent capable de :

- Répondre aux questions des utilisateurs sur les événements culturels en région PACA
- Fournir des recommandations personnalisées basées sur les préférences (lieu, date, type d'événement)
- Utiliser la recherche sémantique pour comprendre les intentions des utilisateurs
- Maintenir un contexte conversationnel pour des interactions naturelles
- Filtrer automatiquement les événements passés et présenter uniquement les événements à venir

Le système utilise une architecture RAG (Retrieval-Augmented Generation) combinant :
- **FAISS** pour la recherche vectorielle rapide
- **Mistral AI** pour les embeddings et la génération de réponses
- **FastAPI** pour exposer l'API REST

## ✨ Fonctionnalités

- 🤖 Chatbot conversationnel avec mémoire de contexte
- 🔍 Recherche sémantique d'événements via embeddings Mistral
- 📍 Filtrage géographique (ville, code postal)
- 📅 Filtrage temporel (événements futurs uniquement)
- 🎯 Recommandations personnalisées basées sur RAG
- 📊 Logging détaillé des interactions
- 🏥 Health check intégré
- 🐳 Déploiement Docker simplifié

## 🔧 Prérequis

### Logiciels requis

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Python** >= 3.11 (pour développement local)
- **Git**

### Clés API

- Clé API **Mistral AI** (obtenir sur console.mistral.ai)

## 📦 Installation

### 1. Cloner le repository

    git clone <url-du-repo>
    cd Projet_7

### 2. Créer le fichier d'environnement

    cp .env.example .env

Éditer le fichier .env et ajouter votre clé API :

    MISTRAL_API_KEY=your_mistral_api_key_here

### 3. Préparer les données

Les données d'événements doivent être présentes dans le dossier data/ :

    data/
    ├── evenements-publics-openagenda_26.json
    ├── chunks_with_embeddings.json
    └── faiss_index.bin

**Note** : Si les fichiers chunks_with_embeddings.json et faiss_index.bin n'existent pas, exécutez le notebook de prétraitement :

    jupyter notebook src/data_process.ipynb

Suivez les étapes du notebook pour :
1. Charger les données OpenAgenda
2. Filtrer les événements futurs
3. Créer les chunks de texte
4. Générer les embeddings avec Mistral
5. Construire l'index FAISS

## ⚙️ Configuration

### Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| MISTRAL_API_KEY | Clé API Mistral AI | ✅ Oui |
| PYTHONUNBUFFERED | Logs en temps réel | Non (défaut: 1) |

### Fichiers de configuration

- docker-compose.yml : Configuration des services Docker
- Dockerfile : Image de l'application
- requirements.txt : Dépendances Python

## 🚀 Déploiement

### Déploiement avec Docker (Recommandé)

#### Mode production

Build et démarrage :

    docker compose up -d --build

Vérifier le health check :

    curl http://localhost:8000/health

### Déploiement local (Développement)

Créer un environnement virtuel :

    python -m venv .venv

Activer l'environnement :

Linux/Mac :

    source .venv/bin/activate

Windows :

    .venv\Scripts\activate

Installer les dépendances :

    uv sync

Lancer l'API :

    uvicorn api.main_api:app --host 0.0.0.0 --port 8000 --reload

### Commandes Docker utiles

Arrêter les services :

    docker compose down

Arrêter et supprimer les volumes :

    docker compose down -v

Reconstruire l'image :

    docker compose build --no-cache

Voir les logs en temps réel :

    docker compose logs -f


## 📖 Utilisation

### Endpoints API : Interface Swagger

Documentation interactive disponible sur :

    http://localhost:8000/docs

## 🏗️ Architecture

### Structure du projet

    Projet_7/
    ├── api/
    │   └── main_api.py          # FastAPI endpoints
    ├── src/
    │   ├── chatbot.py           # Logique du chatbot
    │   └── data_process.ipynb   # Prétraitement des données
    ├── data/
    │   ├── evenements-publics-openagenda_26.json
    │   ├── chunks_with_embeddings.json
    │   └── faiss_index.bin
    ├── logs/
    │   └── chatbot.log          # Logs de l'application
    ├── tests/
    │   ├── evaluate_deepeval.py
    │   └── test_queries.json
    ├── docker-compose.yml
    ├── Dockerfile
    ├── requirements.txt
    ├── .env.example
    └── README.md

### Schéma UML
```mermaid
graph TB
    User[Utilisateur] --> API[FastAPI]
    API[Endpoint /ask] --> User[Utilisateur]
    API --> Chatbot[Chatbot]
    LLM --> API
    Chatbot --> FAISS[Base Vectorielle FAISS]
    Chatbot --> LLM[Mistral LLM]
    Data[Données Événements]
    
    subgraph Pipeline["Pipeline de Traitement"]
        Data --> Chunks[Chunking]
        Chunks --> EmbedGen[Génération Embeddings]
        EmbedGen --> FAISS
    end
    
```

### Technologies utilisées

- **FastAPI** : Framework web asynchrone
- **Mistral AI** : Modèle LLM et embeddings
- **FAISS** : Recherche vectorielle
- **Pandas** : Manipulation de données
- **Docker** : Conteneurisation
- **Uvicorn** : Serveur ASGI


### Pipeline de traitement

**1. Prétraitement (data_process.ipynb) :**
- Chargement des données OpenAgenda
- Filtrage des événements futurs
- Chunking conditionnel (> 512 tokens)
- Génération des embeddings Mistral
- Construction de l'index FAISS

**2. Recherche (chatbot.py) :**
- Embedding de la requête utilisateur
- Recherche des k plus proches voisins dans FAISS
- Filtrage par métadonnées (date, lieu)

**3. Génération (chatbot.py) :**
- Construction du prompt avec contexte
- Appel à Mistral AI pour génération
- Formatage de la réponse

## 🧪 Tests

### Tests d'évaluation

Évaluation avec DeepEval :

    python tests/evaluate_deepeval.py


### Mise à jour des données

1. Télécharger les nouvelles données OpenAgenda
2. Placer le fichier dans data/evenements-publics-openagenda_26.json
3. Exécuter le notebook data_process.ipynb
4. Redémarrer le service :

    docker compose restart chatbot-api

### Justification des choix techniques

| Composant | Choix | Justification |
|-----------|-------|---------------|
| **Framework API** | FastAPI | Asynchrone, documentation auto, validation Pydantic |
| **LLM** | Mistral AI | Support français natif, coût optimisé, API simple |
| **Base vectorielle** | FAISS | Rapide sur CPU, pas de serveur externe requis |
| **Chunking** | Conditionnel (400 tokens) | Équilibre contexte/précision, overlap pour continuité |
| **Embedding** | mistral-embed | 1024 dimensions, optimisé pour le français |
| **Conteneurisation** | Docker | Déploiement reproductible, isolation des dépendances |