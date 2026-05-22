import json
import faiss
import pandas as pd
from src.chatbot import EventChatbot


def test_init_with_valid_paths(tmp_path):
    """Test successful initialization with valid index and chunks paths"""
    # Create temporary test files
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    
    # Create a minimal FAISS index
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create sample chunks data with required columns
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event description",
            "embedding": [0.1] * dimension,
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "firstdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Initialize chatbot with explicit paths
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Assertions
    assert chatbot.index is not None
    assert chatbot.index.ntotal == 0  # Empty index
    assert isinstance(chatbot.chunks_df, pd.DataFrame)
    assert len(chatbot.chunks_df) == 1
    assert 'uid' in chatbot.chunks_df.columns
    assert 'text' in chatbot.chunks_df.columns
    assert chatbot.available_cities is not None
    assert chatbot.available_postal_codes is not None
    assert chatbot.mistral is not None

def test_init_raises_error_when_index_path_does_not_exist(tmp_path):
    """Test that FileNotFoundError is raised when index_path does not exist"""
    # Create a non-existent index path
    non_existent_index = tmp_path / "non_existent_index.idx"
    
    # Create a valid chunks file
    chunks_path = tmp_path / "test_chunks.json"
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event",
            "embedding": [0.1] * 1024,
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
        }
    ]
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Attempt to initialize with non-existent index path
    try:
        chatbot = EventChatbot(index_path=str(non_existent_index), chunks_path=str(chunks_path))
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError as e:
        assert "Index FAISS introuvable" in str(e)
        assert str(non_existent_index) in str(e)

def test_init_raises_error_when_chunks_path_does_not_exist(tmp_path):
    """Test that FileNotFoundError is raised when chunks_path does not exist"""
    # Create a valid FAISS index
    index_path = tmp_path / "test_index.idx"
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create a non-existent chunks path
    non_existent_chunks = tmp_path / "non_existent_chunks.json"
    
    # Attempt to initialize with non-existent chunks path
    try:
        chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(non_existent_chunks))
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError as e:
        assert "Fichier chunks introuvable" in str(e)
        assert str(non_existent_chunks) in str(e)


def test_format_context_should_format_multiday_event_with_warning(tmp_path):
    """Test that multi-day events are formatted with explicit warnings about start and end dates"""
    # Create temporary test files
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    
    # Create a minimal FAISS index
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create a multi-day event (different dates for firstdate_begin and lastdate_end)
    now = pd.Timestamp.now(tz='UTC')
    start_date = now + pd.Timedelta(days=5)  # Starts in 5 days
    end_date = now + pd.Timedelta(days=8)    # Ends 3 days later (multi-day event)
    
    chunks_data = [
        {
            "uid": "multiday_event_1",
            "chunk_id": 0,
            "text": "Festival de musique sur plusieurs jours",
            "embedding": [0.1] * dimension,
            "firstdate_begin": start_date.timestamp() * 1000,
            "firstdate_end": (start_date + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": (end_date - pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_end": end_date.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_name": "Parc des expositions"
        }
    ]
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Format context for the multi-day event
    formatted_context = chatbot.format_context(chatbot.chunks_df)
    
    # Assertions
    assert "⚠️ ÉVÉNEMENT MULTI-JOURS" in formatted_context
    assert "Jour de début:" in formatted_context
    assert "Jour de fin:" in formatted_context
    assert "⚠️ NE PAS CONFONDRE la date de début et la date de fin" in formatted_context
    assert start_date.strftime('%d/%m/%Y') in formatted_context
    assert end_date.strftime('%d/%m/%Y') in formatted_context
    assert "=== ÉVÉNEMENT multiday_event_1 ===" in formatted_context
    assert "DATES:" in formatted_context

def test_format_context_should_format_single_day_event(tmp_path):
    """Test that single-day events are formatted without multi-day warnings"""
    # Create temporary test files
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    
    # Create a minimal FAISS index
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create a single-day event (same date for firstdate_begin and firstdate_end)
    now = pd.Timestamp.now(tz='UTC')
    event_date = now + pd.Timedelta(days=5)  # Starts in 5 days
    event_end = event_date + pd.Timedelta(hours=2)  # Ends same day, 2 hours later
    
    chunks_data = [
        {
            "uid": "single_day_event_1",
            "chunk_id": 0,
            "text": "Concert de jazz en soirée",
            "embedding": [0.1] * dimension,
            "firstdate_begin": event_date.timestamp() * 1000,
            "firstdate_end": event_end.timestamp() * 1000,
            "lastdate_begin": event_date.timestamp() * 1000,
            "lastdate_end": event_end.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_name": "Salle Pleyel"
        }
    ]
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Format context for the single-day event
    formatted_context = chatbot.format_context(chatbot.chunks_df)
    
    # Assertions
    assert "⚠️ ÉVÉNEMENT MULTI-JOURS" not in formatted_context
    assert "Jour de début:" not in formatted_context
    assert "Jour de fin:" not in formatted_context
    assert "⚠️ NE PAS CONFONDRE la date de début et la date de fin" not in formatted_context
    assert "Date début:" in formatted_context
    assert "Date fin:" in formatted_context
    assert event_date.strftime('%d/%m/%Y') in formatted_context
    assert "=== ÉVÉNEMENT single_day_event_1 ===" in formatted_context
    assert "DATES:" in formatted_context


def test_interactive_mode_should_print_welcome_message_with_banner_when_started(tmp_path, capsys, monkeypatch):
    """Test that interactive mode prints welcome message with banner when started"""
    # Create temporary test files
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    
    # Create a minimal FAISS index
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create sample chunks data
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event description",
            "embedding": [0.1] * dimension,
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "firstdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Mock user input to immediately quit
    monkeypatch.setattr('builtins.input', lambda _: 'quit')
    
    # Clear any previous output
    capsys.readouterr()
    
    # Run interactive mode
    chatbot.interactive_mode()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Assertions
    assert "="*80 in captured.out
    assert "🎭 CHATBOT D'ÉVÉNEMENTS CULTURELS" in captured.out
    assert "Tapez 'quit' ou 'exit' pour quitter" in captured.out
    assert "👋 Au revoir !" in captured.out


def test_apply_temporal_filter_should_correctly_filter_events_that_start_before_target_month_and_end_during_target_month(tmp_path):
    """Test that events starting before the target month and ending during it are correctly filtered"""
    # Create temporary test files
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    
    # Create a minimal FAISS index
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create test data with an event that starts before March and ends during March
    now = pd.Timestamp.now(tz='UTC')
    
    # Event that starts in February and ends in March
    event_start = pd.Timestamp(year=2025, month=2, day=20, tz='UTC')
    event_end = pd.Timestamp(year=2025, month=3, day=10, tz='UTC')
    
    chunks_data = [
        {
            "uid": "event_spanning_months",
            "chunk_id": 0,
            "text": "Event spanning from February to March",
            "embedding": [0.1] * dimension,
            "firstdate_begin": event_start.timestamp() * 1000,
            "firstdate_end": (event_start + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": (event_end - pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_end": event_end.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Create temporal filter for March 2025
    temporal_filter = {'type': 'month', 'month': 3, 'year': 2025}
    
    # Apply temporal filter
    filtered_df = chatbot._apply_temporal_filter(chatbot.chunks_df, temporal_filter)
    
    # Assertions
    assert len(filtered_df) == 1, "Event starting before March and ending in March should be included"
    assert filtered_df.iloc[0]['uid'] == "event_spanning_months"
    assert filtered_df.iloc[0]['firstdate_begin'] < pd.Timestamp(year=2025, month=3, day=1, tz='UTC')
    assert filtered_df.iloc[0]['lastdate_end'] >= pd.Timestamp(year=2025, month=3, day=1, tz='UTC')