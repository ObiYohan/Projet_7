FROM python:3.13-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install --no-cache-dir uv

# Copier les fichiers de configuration
COPY pyproject.toml uv.lock* ./

# Créer un environnement virtuel et installer les dépendances
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

# Copier le code source
COPY api/ ./api/
COPY src/ ./src/
COPY data/ ./data/

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["python", "api/main_api.py"]