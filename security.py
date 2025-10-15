import hashlib
import uuid
import base64
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

# Database connection will be passed explicitly.

def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Enhanced password hashing with salt"""
    if salt is None:
        salt = base64.b64encode(uuid.uuid4().bytes).decode()

    # Use PBKDF2 for better security
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return base64.b64encode(password_hash).decode(), salt


def verify_password(password: str, password_hash: str, salt: str = None) -> bool:
    """Verify password with enhanced security"""
    if salt:
        computed_hash, _ = hash_password(password, salt)
        return computed_hash == password_hash
    else:
        # Fallback for legacy hashes
        return hashlib.sha256(password.encode()).hexdigest() == password_hash


def check_password_strength(password: str) -> Dict:
    """Check password strength and return requirements"""
    requirements = {
        'length': len(password) >= 8,
        'uppercase': bool(re.search(r'[A-Z]', password)),
        'lowercase': bool(re.search(r'[a-z]', password)),
        'digit': bool(re.search(r'\d', password)),
        'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
    }

    score = sum(requirements.values())
    strength = 'Very Weak' if score < 2 else 'Weak' if score < 3 else 'Medium' if score < 4 else 'Strong' if score < 5 else 'Very Strong'

    return {
        'score': score,
        'strength': strength,
        'requirements': requirements,
        'is_valid': score >= 4
    }


def authenticate_user(conn: Any, username: str, password: str) -> Optional[Dict]:
    """Enhanced authentication with security features"""
    # conn is now passed as an argument

    # Check if account is locked
    user_data = conn.execute("""
        SELECT id, password_hash, login_attempts, account_locked, active, password_expires, full_name, role, email
        FROM users WHERE username = ?
    """, [username]).fetchone()

    if not user_data:
        return None

    user_id, stored_hash, attempts, locked, active, expires, full_name, role, email = user_data

    if locked or not active:
        return None

    # Check password expiration
    if expires and datetime.now() > expires:
        return {'error': 'password_expired'}

    # Verify password (using legacy method for existing data)
    if verify_password(password, stored_hash):
        # Reset login attempts on successful login
        conn.execute("""
            UPDATE users 
            SET login_attempts = 0, last_login = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [user_id])

        return {
            'id': user_id,
            'username': username,
            'role': role,
            'email': email,
            'full_name': full_name,
            'active': active
        }
    else:
        # Increment login attempts
        new_attempts = attempts + 1
        lock_account = new_attempts >= 3

        conn.execute("""
            UPDATE users 
            SET login_attempts = ?, account_locked = ?
            WHERE id = ?
        """, [new_attempts, lock_account, user_id])

        return None


def create_audit_log(conn: Any, table_name: str, record_id: str, action: str, old_values: Dict = None, new_values: Dict = None,
                     reason: str = None, changed_by_user_id: str = None):
    """Create audit log entry for data changes"""
    # conn is now passed as an argument
    # changed_by_user_id is added to explicitly pass the user ID

    log_id = str(uuid.uuid4())
    user_id = changed_by_user_id # Use the explicitly passed user ID

    # The caller must provide changed_by_user_id.
    
    conn.execute("""
        INSERT INTO audit_log (id, table_name, record_id, action, old_values, new_values, changed_by, change_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        log_id, table_name, record_id, action,
        json.dumps(old_values) if old_values else None,
        json.dumps(new_values) if new_values else None,
        user_id, reason
    ])


def generate_alert(conn: Any, alert_type: str, severity: str, title: str, message: str, target_user: str = None,
                   target_role: str = None):
    """Generate system alert"""
    # conn is now passed as an argument

    alert_id = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(days=30)

    conn.execute("""
        INSERT INTO alerts (id, alert_type, severity, title, message, target_user, target_role, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [alert_id, alert_type, severity, title, message, target_user, target_role, expires_at])
