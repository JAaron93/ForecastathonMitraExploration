"""
Tests for HMAC signature verification in load_joblib function.
"""

import hashlib
import hmac

import pytest

from src.utils.serialization import load_joblib, save_joblib


def test_load_joblib_without_signature(tmp_path):
    """Test that load_joblib works without signature verification (backward compatibility)."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    temp_path = tmp_path / "test.joblib"
    save_joblib(test_data, temp_path)

    # Load without signature verification (should work)
    loaded_data = load_joblib(temp_path)
    assert loaded_data == test_data


def test_load_joblib_with_valid_signature(tmp_path):
    """Test that load_joblib works with valid HMAC signature."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    temp_path = tmp_path / "test_sig.joblib"
    save_joblib(test_data, temp_path)

    # Calculate HMAC signature
    with open(temp_path, "rb") as f:
        file_data = f.read()

    hmac_key = b"test-secret-key"
    signature = hmac.new(hmac_key, file_data, hashlib.sha256).hexdigest()

    # Load with valid signature (should work)
    loaded_data = load_joblib(temp_path, signature=signature, hmac_key=hmac_key)
    assert loaded_data == test_data


def test_load_joblib_with_invalid_signature(tmp_path):
    """Test that load_joblib raises ValueError with invalid HMAC signature."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    temp_path = tmp_path / "test_invalid_sig.joblib"
    save_joblib(test_data, temp_path)

    # Calculate HMAC signature with correct key
    with open(temp_path, "rb") as f:
        file_data = f.read()

    correct_hmac_key = b"correct-secret-key"
    signature = hmac.new(correct_hmac_key, file_data, hashlib.sha256).hexdigest()

    # Load with incorrect key (should raise ValueError)
    with pytest.raises(ValueError, match="HMAC signature verification failed"):
        load_joblib(temp_path, signature=signature, hmac_key=b"wrong-key")


def test_load_joblib_with_tampered_signature(tmp_path):
    """Test that load_joblib raises ValueError with a tampered HMAC signature."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    temp_path = tmp_path / "test_tampered_sig.joblib"
    save_joblib(test_data, temp_path)

    # Use a valid hmac_key
    hmac_key = b"test-secret-key"

    # Provide a tampered signature (e.g., all zeros, should not match)
    tampered_signature = "0" * 64  # SHA256 hex digest is 64 characters long

    # Load with tampered signature (should raise ValueError)
    with pytest.raises(ValueError, match="HMAC signature verification failed"):
        load_joblib(temp_path, signature=tampered_signature, hmac_key=hmac_key)


def test_load_joblib_with_partial_hmac_signature_only(tmp_path):
    """Test load_joblib raises ValueError if only signature is provided."""
    test_data = {"key": "value"}
    temp_path = tmp_path / "test_partial_sig.joblib"
    save_joblib(test_data, temp_path)

    with open(temp_path, "rb") as f:
        file_data = f.read()
    hmac_key = b"test-secret-key"
    signature = hmac.new(hmac_key, file_data, hashlib.sha256).hexdigest()

    with pytest.raises(
        ValueError,
        match="Both 'signature' and 'hmac_key' must be provided for verification",
    ):
        load_joblib(temp_path, signature=signature, hmac_key=None)


def test_load_joblib_with_partial_hmac_key_only(tmp_path):
    """Test load_joblib raises ValueError if only hmac_key is provided."""
    test_data = {"key": "value"}
    temp_path = tmp_path / "test_partial_key.joblib"
    save_joblib(test_data, temp_path)

    with pytest.raises(
        ValueError,
        match="Both 'signature' and 'hmac_key' must be provided for verification",
    ):
        load_joblib(temp_path, signature=None, hmac_key=b"test-secret-key")


def test_load_joblib_nonexistent_file(tmp_path):
    """Test that load_joblib raises FileNotFoundError for nonexistent file."""
    nonexistent_path = tmp_path / "nonexistent.joblib"

    with pytest.raises(FileNotFoundError, match="Joblib file not found"):
        load_joblib(nonexistent_path)
