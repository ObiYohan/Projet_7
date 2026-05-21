   
def test_count_tokens_empty_string():
    """Test that count_tokens returns 0 for an empty string"""
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import count_tokens
    
    result = count_tokens("")
    assert result == 0, f"Expected 0 tokens for empty string, got {result}"

def test_process_description_single_chunk_with_metadata():
    """Test that process_description returns a single chunk with all metadata when token_count < 512"""
    import sys
    from pathlib import Path
    import pandas as pd
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import process_description
    
    # Create a test row with token_count < 512
    test_row = pd.Series({
        "longdescription_fr": "This is a short description with few tokens.",
        "token_count": 100,
        "uid": "test-event-123",
        "firstdate_begin": pd.Timestamp("2024-01-15", tz='UTC'),
        "firstdate_end": pd.Timestamp("2024-01-15", tz='UTC'),
        "lastdate_begin": pd.Timestamp("2024-01-15", tz='UTC'),
        "lastdate_end": pd.Timestamp("2024-01-20", tz='UTC'),
        "location_name": "Test Venue",
        "location_city": "Paris",
        "location_postalcode": "75001",
        "location_phone": "0123456789",
        "location_website": "http://test.com"
    })
    
    result = process_description(test_row)
    
    # Verify single chunk returned
    assert len(result) == 1, f"Expected 1 chunk, got {len(result)}"
    
    chunk = result[0]
    
    # Verify all metadata is preserved
    assert chunk["uid"] == "test-event-123"
    assert chunk["chunk_id"] == 0
    assert chunk["text"] == "This is a short description with few tokens."
    assert chunk["token_count"] == 100
    assert chunk["firstdate_begin"] == pd.Timestamp("2024-01-15", tz='UTC')
    assert chunk["firstdate_end"] == pd.Timestamp("2024-01-15", tz='UTC')
    assert chunk["lastdate_begin"] == pd.Timestamp("2024-01-15", tz='UTC')
    assert chunk["lastdate_end"] == pd.Timestamp("2024-01-20", tz='UTC')
    assert chunk["location_name"] == "Test Venue"
    assert chunk["location_city"] == "Paris"
    assert chunk["location_postalcode"] == "75001"
    assert chunk["location_phone"] == "0123456789"
    assert chunk["location_website"] == "http://test.com"

def test_process_description_multiple_chunks_with_metadata():
    """Test that process_description splits text into multiple chunks and preserves all metadata when token_count > 512"""
    import sys
    from pathlib import Path
    import pandas as pd
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import process_description
    
    # Create a test row with token_count > 512 to trigger chunking
    long_text = "a" * 2000  # This will exceed 512 tokens
    test_row = pd.Series({
        "longdescription_fr": long_text,
        "token_count": 2000,
        "uid": "test-event-456",
        "firstdate_begin": pd.Timestamp("2024-02-01", tz='UTC'),
        "firstdate_end": pd.Timestamp("2024-02-01", tz='UTC'),
        "lastdate_begin": pd.Timestamp("2024-02-01", tz='UTC'),
        "lastdate_end": pd.Timestamp("2024-02-15", tz='UTC'),
        "location_name": "Grand Palais",
        "location_city": "Lyon",
        "location_postalcode": "69001",
        "location_phone": "0987654321",
        "location_website": "http://grandpalais.com"
    })
    
    result = process_description(test_row)
    
    # Verify multiple chunks were created
    assert len(result) > 1, f"Expected multiple chunks, got {len(result)}"
    
    # Verify all chunks preserve metadata
    for i, chunk in enumerate(result):
        assert chunk["uid"] == "test-event-456", f"Chunk {i} has wrong uid"
        assert chunk["chunk_id"] == i, f"Chunk {i} has wrong chunk_id"
        assert isinstance(chunk["text"], str), f"Chunk {i} text is not a string"
        assert chunk["token_count"] > 0, f"Chunk {i} has invalid token_count"
        assert chunk["firstdate_begin"] == pd.Timestamp("2024-02-01", tz='UTC'), f"Chunk {i} has wrong firstdate_begin"
        assert chunk["firstdate_end"] == pd.Timestamp("2024-02-01", tz='UTC'), f"Chunk {i} has wrong firstdate_end"
        assert chunk["lastdate_begin"] == pd.Timestamp("2024-02-01", tz='UTC'), f"Chunk {i} has wrong lastdate_begin"
        assert chunk["lastdate_end"] == pd.Timestamp("2024-02-15", tz='UTC'), f"Chunk {i} has wrong lastdate_end"
        assert chunk["location_name"] == "Grand Palais", f"Chunk {i} has wrong location_name"
        assert chunk["location_city"] == "Lyon", f"Chunk {i} has wrong location_city"
        assert chunk["location_postalcode"] == "69001", f"Chunk {i} has wrong location_postalcode"
        assert chunk["location_phone"] == "0987654321", f"Chunk {i} has wrong location_phone"
        assert chunk["location_website"] == "http://grandpalais.com", f"Chunk {i} has wrong location_website"

def test_load_and_filter_data_filters_past_events():
    """Test that load_and_filter_data filters out events where lastdate_end is before the current date"""
    import sys
    from pathlib import Path
    import pandas as pd
    import tempfile
    import json
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import load_and_filter_data
    
    # Create test data with mix of past and future events
    current_date = pd.Timestamp.now(tz='UTC')
    past_date = current_date - pd.Timedelta(days=30)
    near_future_date = current_date + pd.Timedelta(seconds=5)  # ✅ Marge de sécurité
    
    test_data = [
        {
            "uid": "past-event-1",
            "longdescription_fr": "This is a past event",
            "firstdate_begin": past_date.isoformat(),
            "firstdate_end": past_date.isoformat(),
            "lastdate_begin": past_date.isoformat(),
            "lastdate_end": past_date.isoformat()
        },
        {
            "uid": "future-event-1",
            "longdescription_fr": "This is a future event",
            "firstdate_begin": near_future_date.isoformat(),
            "firstdate_end": near_future_date.isoformat(),
            "lastdate_begin": near_future_date.isoformat(),
            "lastdate_end": near_future_date.isoformat()
        },
        {
            "uid": "current-event-1",
            "longdescription_fr": "This event ends today",
            "firstdate_begin": near_future_date.isoformat(),
            "firstdate_end": near_future_date.isoformat(),
            "lastdate_begin": near_future_date.isoformat(),
            "lastdate_end": near_future_date.isoformat()
        }
    ]
    
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        result = load_and_filter_data(temp_file)
        
        # Verify past events are filtered out
        assert "past-event-1" not in result["uid"].values, "Past event should be filtered out"
        
        # Verify future events are kept
        assert "future-event-1" in result["uid"].values, "Future event should be kept"
        
        # Verify current events (ending today) are kept
        assert "current-event-1" in result["uid"].values, "Current event should be kept"
        
        # Verify only 2 events remain
        assert len(result) == 2, f"Expected 2 events after filtering, got {len(result)}"
    finally:
        # Cleanup temporary file
        import os
        os.unlink(temp_file)

def test_load_and_filter_data_converts_dates_to_utc():
    """Test that load_and_filter_data converts all date columns to UTC timezone datetime objects"""
    import sys
    from pathlib import Path
    import pandas as pd
    import tempfile
    import json
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import load_and_filter_data
    
    # Create test data with dates in different formats
    test_data = [
        {
            "uid": "event-1",
            "longdescription_fr": "Test event description",
            "firstdate_begin": "2024-06-01T10:00:00",
            "firstdate_end": "2024-06-01T18:00:00",
            "lastdate_begin": "2024-06-15T10:00:00",
            "lastdate_end": "2024-12-31T23:59:59"
        },
        {
            "uid": "event-2",
            "longdescription_fr": "Another test event",
            "firstdate_begin": "2024-07-01T14:00:00+02:00",
            "firstdate_end": "2024-07-01T20:00:00+02:00",
            "lastdate_begin": "2024-07-20T14:00:00+02:00",
            "lastdate_end": "2025-01-15T23:59:59+02:00"
        }
    ]
    
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        result = load_and_filter_data(temp_file)
        
        # Verify all date columns are datetime objects with UTC timezone
        date_columns = ['firstdate_begin', 'firstdate_end', 'lastdate_begin', 'lastdate_end']
        
        for col in date_columns:
            assert col in result.columns, f"Column {col} not found in result"
            assert pd.api.types.is_datetime64_any_dtype(result[col]), f"Column {col} is not datetime type"
            
            # Verify all dates have UTC timezone
            for idx, value in result[col].items():
                assert value.tz is not None, f"Column {col} at index {idx} has no timezone"
                assert str(value.tz) == 'UTC', f"Column {col} at index {idx} is not UTC timezone, got {value.tz}"
    finally:
        # Cleanup temporary file
        import os
        os.unlink(temp_file)

def test_create_chunks_counts_split_descriptions():
    """Test that create_chunks correctly identifies and counts descriptions that were split into multiple chunks"""
    import sys
    from pathlib import Path
    import pandas as pd
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import create_chunks
    
    # Create test data with mix of short and long descriptions
    short_text = "This is a short description."
    long_text = "a" * 2000  # This will exceed 512 tokens and trigger chunking
    
    test_data = pd.DataFrame([
        {
            "uid": "event-short-1",
            "longdescription_fr": short_text,
            "token_count": 10,
            "firstdate_begin": pd.Timestamp("2024-06-01", tz='UTC'),
            "firstdate_end": pd.Timestamp("2024-06-01", tz='UTC'),
            "lastdate_begin": pd.Timestamp("2024-06-01", tz='UTC'),
            "lastdate_end": pd.Timestamp("2024-12-31", tz='UTC'),
            "location_name": "Venue 1",
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_phone": None,
            "location_website": None
        },
        {
            "uid": "event-long-1",
            "longdescription_fr": long_text,
            "token_count": 2000,
            "firstdate_begin": pd.Timestamp("2024-07-01", tz='UTC'),
            "firstdate_end": pd.Timestamp("2024-07-01", tz='UTC'),
            "lastdate_begin": pd.Timestamp("2024-07-01", tz='UTC'),
            "lastdate_end": pd.Timestamp("2024-12-31", tz='UTC'),
            "location_name": "Venue 2",
            "location_city": "Lyon",
            "location_postalcode": "69001",
            "location_phone": None,
            "location_website": None
        },
        {
            "uid": "event-short-2",
            "longdescription_fr": short_text,
            "token_count": 10,
            "firstdate_begin": pd.Timestamp("2024-08-01", tz='UTC'),
            "firstdate_end": pd.Timestamp("2024-08-01", tz='UTC'),
            "lastdate_begin": pd.Timestamp("2024-08-01", tz='UTC'),
            "lastdate_end": pd.Timestamp("2024-12-31", tz='UTC'),
            "location_name": "Venue 3",
            "location_city": "Marseille",
            "location_postalcode": "13001",
            "location_phone": None,
            "location_website": None
        }
    ])
    
    result = create_chunks(test_data)
    
    # Verify that chunks were created
    assert len(result) > len(test_data), "Expected more chunks than original descriptions"
    
    # Verify that descriptions with chunk_id > 0 exist (i.e., split descriptions)
    split_descriptions = result[result['chunk_id'] > 0]
    assert len(split_descriptions) > 0, "Expected at least one description to be split"
    
    # Verify that only the long description was split
    split_uids = split_descriptions['uid'].unique()
    assert len(split_uids) == 1, f"Expected exactly 1 split description, got {len(split_uids)}"
    assert "event-long-1" in split_uids, "Expected event-long-1 to be split"
    
    # Verify that short descriptions were not split
    short_event_chunks = result[result['uid'].isin(['event-short-1', 'event-short-2'])]
    assert all(short_event_chunks['chunk_id'] == 0), "Short descriptions should not be split"
    
    # Verify that text_for_embedding column was added
    assert 'text_for_embedding' in result.columns, "text_for_embedding column should be present"
    assert result['text_for_embedding'].notna().all(), "All rows should have text_for_embedding"

def test_generate_embeddings_with_valid_api_key():
    """Test that generate_embeddings successfully generates embeddings when a valid api_key is provided as parameter"""
    import sys
    from pathlib import Path
    import pandas as pd
    from unittest.mock import MagicMock, patch
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import generate_embeddings
    
    # Create test dataframe with text_for_embedding column
    test_chunks_df = pd.DataFrame([
        {"text_for_embedding": "Test event in Paris | Ville: Paris | Date: juin 2024"},
        {"text_for_embedding": "Another event in Lyon | Ville: Lyon | Date: juillet 2024"}
    ])
    
    # Mock embeddings response
    mock_embedding_1 = [0.1] * 1024
    mock_embedding_2 = [0.2] * 1024
    
    mock_embedding_item_1 = MagicMock()
    mock_embedding_item_1.embedding = mock_embedding_1
    
    mock_embedding_item_2 = MagicMock()
    mock_embedding_item_2.embedding = mock_embedding_2
    
    mock_response = MagicMock()
    mock_response.data = [mock_embedding_item_1, mock_embedding_item_2]
    
    # Mock Mistral client
    mock_mistral_instance = MagicMock()
    mock_mistral_instance.embeddings.create.return_value = mock_response
    mock_mistral_instance.__enter__ = MagicMock(return_value=mock_mistral_instance)
    mock_mistral_instance.__exit__ = MagicMock(return_value=False)
    
    with patch('load_new_data.Mistral') as mock_mistral_class:
        mock_mistral_class.return_value = mock_mistral_instance
        
        # Call function with explicit API key
        result = generate_embeddings(test_chunks_df, api_key="test-api-key-12345")
        
        # Verify Mistral was called with the provided API key
        mock_mistral_class.assert_called_once_with(api_key="test-api-key-12345")
        
        # Verify embeddings.create was called
        assert mock_mistral_instance.embeddings.create.called, "embeddings.create should be called"
        
        # Verify correct number of embeddings returned
        assert len(result) == 2, f"Expected 2 embeddings, got {len(result)}"
        
        # Verify embeddings content
        assert result[0] == mock_embedding_1, "First embedding should match mock data"
        assert result[1] == mock_embedding_2, "Second embedding should match mock data"

def test_generate_embeddings_raises_valueerror_when_no_api_key():
    """Test that generate_embeddings raises ValueError when api_key is None and MISTRAL_API_KEY environment variable is not set"""
    import sys
    from pathlib import Path
    import pandas as pd
    import os
    from unittest.mock import patch
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import generate_embeddings
    
    # Create test dataframe with text_for_embedding column
    test_chunks_df = pd.DataFrame([
        {"text_for_embedding": "Test event in Paris | Ville: Paris | Date: juin 2024"}
    ])
    
    # Mock os.getenv to return empty string (simulating unset environment variable)
    with patch.dict(os.environ, {}, clear=True):
        try:
            generate_embeddings(test_chunks_df, api_key=None)
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "MISTRAL_API_KEY non définie" in str(e), f"Expected error message about MISTRAL_API_KEY, got {str(e)}"

def test_generate_embeddings_raises_valueerror_when_api_key_is_empty_string():
    """Test that generate_embeddings raises ValueError when api_key is an empty string"""
    import sys
    from pathlib import Path
    import pandas as pd
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import generate_embeddings
    
    # Create test dataframe with text_for_embedding column
    test_chunks_df = pd.DataFrame([
        {"text_for_embedding": "Test event in Paris | Ville: Paris | Date: juin 2024"}
    ])
    
    try:
        generate_embeddings(test_chunks_df, api_key="")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "MISTRAL_API_KEY non définie" in str(e), f"Expected error message about MISTRAL_API_KEY, got {str(e)}"

def test_generate_embeddings_handles_rate_limit_error():
    """Test that generate_embeddings handles rate limit (429) error by waiting 60 seconds and retrying the failed batch"""
    import sys
    from pathlib import Path
    import pandas as pd
    from unittest.mock import MagicMock, patch
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import generate_embeddings
    
    # Create test dataframe with text_for_embedding column
    test_chunks_df = pd.DataFrame([
        {"text_for_embedding": "Test event in Paris | Ville: Paris | Date: juin 2024"},
        {"text_for_embedding": "Another event in Lyon | Ville: Lyon | Date: juillet 2024"}
    ])
    
    # Mock successful embeddings response
    mock_embedding_1 = [0.1] * 1024
    mock_embedding_2 = [0.2] * 1024
    
    mock_embedding_item_1 = MagicMock()
    mock_embedding_item_1.embedding = mock_embedding_1
    
    mock_embedding_item_2 = MagicMock()
    mock_embedding_item_2.embedding = mock_embedding_2
    
    mock_success_response = MagicMock()
    mock_success_response.data = [mock_embedding_item_1, mock_embedding_item_2]
    
    # Mock Mistral client
    mock_mistral_instance = MagicMock()
    
    # First call raises 429 error, second call succeeds
    mock_mistral_instance.embeddings.create.side_effect = [
        Exception("429 Rate limit exceeded"),
        mock_success_response
    ]
    
    mock_mistral_instance.__enter__ = MagicMock(return_value=mock_mistral_instance)
    mock_mistral_instance.__exit__ = MagicMock(return_value=False)
    
    with patch('load_new_data.Mistral') as mock_mistral_class:
        mock_mistral_class.return_value = mock_mistral_instance
        
        with patch('load_new_data.time.sleep') as mock_sleep:
            # Call function
            result = generate_embeddings(test_chunks_df, api_key="test-api-key-12345")
            
            # Verify that sleep was called with 60 seconds after rate limit error
            assert mock_sleep.call_count >= 1, "time.sleep should be called at least once"
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert 60 in sleep_calls, f"Expected 60 second sleep for rate limit, got {sleep_calls}"
            
            # Verify embeddings.create was called twice (first failed, second succeeded)
            assert mock_mistral_instance.embeddings.create.call_count == 2, \
                f"Expected 2 calls to embeddings.create (first fails, second succeeds), got {mock_mistral_instance.embeddings.create.call_count}"
            
            # Verify correct embeddings were returned
            assert len(result) == 2, f"Expected 2 embeddings, got {len(result)}"
            assert result[0] == mock_embedding_1, "First embedding should match mock data"
            assert result[1] == mock_embedding_2, "Second embedding should match mock data"

def test_save_chunks_with_embeddings_custom_output_file():
    """Test that save_chunks_with_embeddings saves chunks DataFrame with embeddings to a custom output file when output_file is specified"""
    import sys
    from pathlib import Path
    import pandas as pd
    import tempfile
    import json
    import os
    
    # Add parent directory to path to import load_new_data
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from load_new_data import save_chunks_with_embeddings
    
    # Create test chunks DataFrame with text_for_embedding column
    test_chunks_df = pd.DataFrame([
        {
            "uid": "event-1",
            "chunk_id": 0,
            "text": "Test event in Paris",
            "token_count": 10,
            "firstdate_begin": pd.Timestamp("2024-06-01", tz='UTC'),
            "firstdate_end": pd.Timestamp("2024-06-01", tz='UTC'),
            "lastdate_begin": pd.Timestamp("2024-06-01", tz='UTC'),
            "lastdate_end": pd.Timestamp("2024-12-31", tz='UTC'),
            "location_name": "Venue 1",
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_phone": None,
            "location_website": None,
            "text_for_embedding": "Test event in Paris | Ville: Paris | Date: juin 2024"
        },
        {
            "uid": "event-2",
            "chunk_id": 0,
            "text": "Another event in Lyon",
            "token_count": 12,
            "firstdate_begin": pd.Timestamp("2024-07-01", tz='UTC'),
            "firstdate_end": pd.Timestamp("2024-07-01", tz='UTC'),
            "lastdate_begin": pd.Timestamp("2024-07-01", tz='UTC'),
            "lastdate_end": pd.Timestamp("2024-12-31", tz='UTC'),
            "location_name": "Venue 2",
            "location_city": "Lyon",
            "location_postalcode": "69001",
            "location_phone": None,
            "location_website": None,
            "text_for_embedding": "Another event in Lyon | Ville: Lyon | Date: juillet 2024"
        }
    ])
    
    # Create test embeddings
    test_embeddings = [[0.1] * 1024, [0.2] * 1024]
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        custom_output_file = f.name
    
    try:
        # Call function with custom output file
        save_chunks_with_embeddings(test_chunks_df, test_embeddings, output_file=custom_output_file)
        
        # Verify file was created
        assert os.path.exists(custom_output_file), f"Output file should exist at {custom_output_file}"
        
        # Load saved data
        with open(custom_output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Verify correct number of records
        assert len(saved_data) == 2, f"Expected 2 records, got {len(saved_data)}"
        
        # Verify embeddings were saved
        assert 'embedding' in saved_data[0], "First record should have embedding"
        assert 'embedding' in saved_data[1], "Second record should have embedding"
        assert saved_data[0]['embedding'] == [0.1] * 1024, "First embedding should match"
        assert saved_data[1]['embedding'] == [0.2] * 1024, "Second embedding should match"
        
        # Verify text_for_embedding was removed
        assert 'text_for_embedding' not in saved_data[0], "text_for_embedding should be removed"
        assert 'text_for_embedding' not in saved_data[1], "text_for_embedding should be removed"
        
        # Verify other fields are preserved
        assert saved_data[0]['uid'] == "event-1", "First record uid should be preserved"
        assert saved_data[0]['chunk_id'] == 0, "First record chunk_id should be preserved"
        assert saved_data[0]['text'] == "Test event in Paris", "First record text should be preserved"
        assert saved_data[1]['uid'] == "event-2", "Second record uid should be preserved"
        
    finally:
        # Cleanup temporary file
        if os.path.exists(custom_output_file):
            os.unlink(custom_output_file)