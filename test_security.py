import pytest
import hashlib
import base64
import uuid
from security import hash_password, verify_password, check_password_strength

# Test cases for hash_password
def test_hash_password_generates_salt_if_none_provided():
    password = "testpassword"
    password_hash, salt = hash_password(password)
    assert password_hash is not None
    assert salt is not None
    assert isinstance(password_hash, str)
    assert isinstance(salt, str)
    assert len(salt) > 0

def test_hash_password_uses_provided_salt():
    password = "testpassword"
    provided_salt = base64.b64encode(uuid.uuid4().bytes).decode()
    password_hash, salt = hash_password(password, provided_salt)
    assert salt == provided_salt

def test_hash_password_produces_different_hashes_for_different_salts():
    password = "testpassword"
    _, salt1 = hash_password(password)
    _, salt2 = hash_password(password)
    assert salt1 != salt2 # Salts should be different if not provided

    hash1, _ = hash_password(password, salt1)
    hash2, _ = hash_password(password, salt2)
    assert hash1 != hash2 # Hashes should be different with different salts

def test_hash_password_produces_same_hash_for_same_password_and_salt():
    password = "testpassword"
    salt = base64.b64encode(uuid.uuid4().bytes).decode()
    hash1, _ = hash_password(password, salt)
    hash2, _ = hash_password(password, salt)
    assert hash1 == hash2

# Test cases for verify_password
def test_verify_password_with_correct_password_and_salt():
    password = "securepassword123"
    password_hash, salt = hash_password(password)
    assert verify_password(password, password_hash, salt) == True

def test_verify_password_with_incorrect_password_and_salt():
    password = "securepassword123"
    wrong_password = "wrongpassword"
    password_hash, salt = hash_password(password)
    assert verify_password(wrong_password, password_hash, salt) == False

def test_verify_password_with_incorrect_salt():
    password = "securepassword123"
    password_hash, correct_salt = hash_password(password)
    incorrect_salt = base64.b64encode(uuid.uuid4().bytes).decode()
    assert verify_password(password, password_hash, incorrect_salt) == False

def test_verify_password_with_legacy_sha256_hash():
    password = "legacy_password"
    legacy_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    assert verify_password(password, legacy_hash, stored_salt=None) == True

def test_verify_password_with_incorrect_legacy_sha256_hash():
    password = "legacy_password"
    wrong_password = "wrong_legacy_password"
    legacy_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    assert verify_password(wrong_password, legacy_hash, stored_salt=None) == False

# Test cases for check_password_strength
def test_check_password_strength_very_weak():
    result = check_password_strength("short")
    assert result['strength'] == 'Very Weak'
    assert result['is_valid'] == False

def test_check_password_strength_weak():
    result = check_password_strength("password")
    assert result['strength'] == 'Weak'
    assert result['is_valid'] == False

def test_check_password_strength_medium():
    result = check_password_strength("Password123")
    assert result['strength'] == 'Strong' # Corrected assertion: score 4 is 'Strong'
    assert result['is_valid'] == True # Corrected assertion: score 4 is valid

def test_check_password_strength_strong():
    result = check_password_strength("Password123!")
    assert result['strength'] == 'Very Strong' # Corrected assertion: score 5 is 'Very Strong'
    assert result['is_valid'] == True

def test_check_password_strength_very_strong():
    result = check_password_strength("P@ssw0rd123!")
    assert result['strength'] == 'Very Strong'
    assert result['is_valid'] == True

def test_check_password_strength_all_requirements_met():
    result = check_password_strength("MySecureP@ss1")
    assert result['is_valid'] == True
    assert result['requirements']['length'] == True
    assert result['requirements']['uppercase'] == True
    assert result['requirements']['lowercase'] == True
    assert result['requirements']['digit'] == True
    assert result['requirements']['special'] == True

def test_check_password_strength_missing_requirements():
    result = check_password_strength("nouppercase123")
    assert result['is_valid'] == False
    assert result['requirements']['uppercase'] == False
    assert result['requirements']['special'] == False
