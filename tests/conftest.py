import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

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