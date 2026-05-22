import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, Mock, patch
import os
import numpy as np

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_mistral_client(monkeypatch):
    """Mock Mistral API for tests"""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1024)
    ]
    monkeypatch.setenv("MISTRAL_API_KEY", "test_key_123")
    return mock_client



@pytest.fixture(autouse=True)
def mock_mistral_api():
    """Mock automatique de l'API Mistral pour tous les tests"""
    
    # Mock de l'embedding
    mock_embedding_response = Mock()
    mock_embedding_response.data = [
        Mock(embedding=np.random.rand(1024).tolist())
    ]
    
    # Mock de la génération de texte
    mock_chat_response = Mock()
    mock_chat_response.choices = [
        Mock(message=Mock(content="Réponse de test du chatbot"))
    ]
    
    with patch('mistralai.Mistral') as mock_mistral:
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_embedding_response
        mock_client.chat.complete.return_value = mock_chat_response
        mock_mistral.return_value = mock_client
        
        yield mock_client

@pytest.fixture(autouse=True)
def set_test_env_vars():
    """Définir les variables d'environnement pour les tests"""
    os.environ['MISTRAL_API_KEY'] = 'test_mock_key_12345'
    yield
    # Cleanup après le test
    if 'MISTRAL_API_KEY' in os.environ:
        del os.environ['MISTRAL_API_KEY']