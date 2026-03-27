#!/usr/bin/env python3
"""
Test script to verify HMAC signature verification in load_joblib function.
"""

import os
import tempfile
import joblib
import hmac
import hashlib
from src.utils.serialization import load_joblib, save_joblib


def test_load_joblib_without_signature():
    """Test that load_joblib works without signature verification (backward compatibility)."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        temp_path = tmp.name

    try:
        # Save the object
        save_joblib(test_data, temp_path)

        # Load without signature verification (should work)
        loaded_data = load_joblib(temp_path)
        assert loaded_data == test_data
        print("✓ Test 1 passed: load_joblib works without signature verification")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_joblib_with_valid_signature():
    """Test that load_joblib works with valid HMAC signature."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        temp_path = tmp.name

    try:
        # Save the object
        save_joblib(test_data, temp_path)

        # Calculate HMAC signature
        with open(temp_path, "rb") as f:
            file_data = f.read()

        hmac_key = b"test-secret-key"
        signature = hmac.new(hmac_key, file_data, hashlib.sha256).hexdigest()

        # Load with valid signature (should work)
        loaded_data = load_joblib(temp_path, signature=signature, hmac_key=hmac_key)
        assert loaded_data == test_data
        print("✓ Test 2 passed: load_joblib works with valid HMAC signature")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_joblib_with_invalid_signature():
    """Test that load_joblib raises ValueError with invalid HMAC signature."""
    # Create a test object
    test_data = {"key": "value", "numbers": [1, 2, 3]}

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        temp_path = tmp.name

    try:
        # Save the object
        save_joblib(test_data, temp_path)

        # Calculate HMAC signature with correct key
        with open(temp_path, "rb") as f:
            file_data = f.read()

        correct_hmac_key = b"correct-secret-key"
        signature = hmac.new(correct_hmac_key, file_data, hashlib.sha256).hexdigest()

        # Load with incorrect key (should raise ValueError)
        try:
            load_joblib(temp_path, signature=signature, hmac_key=b"wrong-key")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "HMAC signature verification failed" in str(e)
            print(
                "✓ Test 3 passed: load_joblib raises ValueError with invalid HMAC signature"
            )

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_joblib_nonexistent_file():
    """Test that load_joblib raises FileNotFoundError for nonexistent file."""
    # Create a temporary file and immediately delete it to guarantee non-existence
    with tempfile.NamedTemporaryFile(delete=True, suffix=".joblib") as tmp:
        nonexistent_path = tmp.name
    # tmp.name is the now-deleted file path, guaranteed unique
    try:
        load_joblib(nonexistent_path)
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError as e:
        assert "Joblib file not found" in str(e)
        print(
            "✓ Test 4 passed: load_joblib raises FileNotFoundError for nonexistent file"
        )


if __name__ == "__main__":
    print("Testing HMAC signature verification in load_joblib...")
    test_load_joblib_without_signature()
    test_load_joblib_with_valid_signature()
    test_load_joblib_with_invalid_signature()
    test_load_joblib_nonexistent_file()
    print("\nAll tests passed! ✓")
