"""
AURA — Database Manager
=======================
SQLite Database — بيحفظ كل بيانات الأهل والأطفال والجلسات

الاستخدام:
    from database import db
    db.init_app(app)
"""

import sqlite3
import os
from datetime import datetime, date
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'aura.db')

# ============================================================
# إنشاء الجداول
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # جدول الأهل
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            first_name TEXT    NOT NULL,
            last_name  TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        )
    ''')

    # جدول الأطفال
    c.execute('''
        CREATE TABLE IF NOT EXISTS children (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            name           TEXT    NOT NULL,
            dob            TEXT    NOT NULL,
            gender         TEXT    NOT NULL,
            jaundice       TEXT    DEFAULT 'no',
            family_history TEXT    DEFAULT 'no',
            created_at     TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # جدول الجلسات
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id     INTEGER NOT NULL,
            user_id      INTEGER NOT NULL,
            date         TEXT    DEFAULT (datetime('now')),
            status       TEXT    DEFAULT 'in_progress',
            FOREIGN KEY (child_id) REFERENCES children(id),
            FOREIGN KEY (user_id)  REFERENCES users(id)
        )
    ''')

    # جدول النتائج
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER NOT NULL,
            model1_prob   REAL,
            model1_gb     REAL,
            model1_tabt   REAL,
            model2_prob   REAL,
            model2_severity TEXT,
            model3_prob   REAL,
            model3_seqs   INTEGER,
            final_prob    REAL,
            risk_level    TEXT,
            q_answers     TEXT,
            created_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    ''')

    conn.commit()
    conn.close()
    print('✅ Database initialized!')

# ============================================================
# Helper
# ============================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # عشان النتايج تبقى dict-like
    return conn

def calc_age(dob_str):
    dob = date.fromisoformat(dob_str)
    return (date.today() - dob).days // 365

# ============================================================
# Users
# ============================================================
def register_user(email, password, first_name, last_name):
    """تسجيل مستخدم جديد — يرجع user_id أو None لو الـ email موجود"""
    try:
        conn = get_conn()
        conn.execute(
            'INSERT INTO users (email, password, first_name, last_name) VALUES (?,?,?,?)',
            (email.lower().strip(),
             generate_password_hash(password),
             first_name.strip(),
             last_name.strip())
        )
        conn.commit()
        user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None  # email موجود

def login_user(email, password):
    """تسجيل الدخول — يرجع user dict أو None"""
    conn = get_conn()
    user = conn.execute(
        'SELECT * FROM users WHERE email = ?', (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if user and check_password_hash(user['password'], password):
        return dict(user)
    return None

def get_user(user_id):
    conn = get_conn()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

# ============================================================
# Children
# ============================================================
def add_child(user_id, name, dob, gender, jaundice='no', family_history='no'):
    """إضافة طفل جديد — يرجع child_id"""
    conn = get_conn()
    conn.execute(
        '''INSERT INTO children (user_id, name, dob, gender, jaundice, family_history)
           VALUES (?,?,?,?,?,?)''',
        (user_id, name.strip(), dob, gender.lower(), jaundice.lower(), family_history.lower())
    )
    conn.commit()
    child_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()
    return child_id

def get_children(user_id):
    """جيب كل أطفال المستخدم"""
    conn = get_conn()
    rows = conn.execute(
        'SELECT * FROM children WHERE user_id = ? ORDER BY created_at DESC', (user_id,)
    ).fetchall()
    conn.close()
    children = []
    for r in rows:
        c = dict(r)
        c['age'] = calc_age(c['dob'])
        c['q_type'] = 'qchat' if c['age'] < 4 else 'aq10'
        children.append(c)
    return children

def get_child(child_id):
    conn = get_conn()
    row  = conn.execute('SELECT * FROM children WHERE id = ?', (child_id,)).fetchone()
    conn.close()
    if row:
        c = dict(row)
        c['age']    = calc_age(c['dob'])
        c['q_type'] = 'qchat' if c['age'] < 4 else 'aq10'
        return c
    return None

# ============================================================
# Sessions
# ============================================================
def create_session(child_id, user_id):
    """إنشاء جلسة جديدة — يرجع session_id"""
    conn = get_conn()
    conn.execute(
        'INSERT INTO sessions (child_id, user_id) VALUES (?,?)',
        (child_id, user_id)
    )
    conn.commit()
    sid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()
    return sid

def get_sessions(user_id, limit=10):
    """جيب آخر جلسات المستخدم"""
    conn = get_conn()
    rows = conn.execute('''
        SELECT s.*, c.name as child_name, r.final_prob, r.risk_level
        FROM sessions s
        JOIN children c ON s.child_id = c.id
        LEFT JOIN results r ON r.session_id = s.id
        WHERE s.user_id = ?
        ORDER BY s.date DESC
        LIMIT ?
    ''', (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ============================================================
# Results
# ============================================================
def save_result(session_id, model1_prob=None, model1_gb=None, model1_tabt=None,
                model2_prob=None, model2_severity=None,
                model3_prob=None, model3_seqs=None,
                final_prob=None, risk_level=None, q_answers=None):
    """حفظ نتيجة جلسة"""
    import json
    conn = get_conn()

    # لو في نتيجة قديمة امسحها
    conn.execute('DELETE FROM results WHERE session_id = ?', (session_id,))

    conn.execute('''
        INSERT INTO results
        (session_id, model1_prob, model1_gb, model1_tabt,
         model2_prob, model2_severity, model3_prob, model3_seqs,
         final_prob, risk_level, q_answers)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', (session_id, model1_prob, model1_gb, model1_tabt,
          model2_prob, model2_severity, model3_prob, model3_seqs,
          final_prob, risk_level,
          json.dumps(q_answers) if q_answers else None))

    # تحديث status الجلسة
    conn.execute(
        "UPDATE sessions SET status = 'completed' WHERE id = ?", (session_id,)
    )
    conn.commit()
    conn.close()

def get_result(session_id):
    """جيب نتيجة جلسة"""
    conn = get_conn()
    row  = conn.execute(
        'SELECT * FROM results WHERE session_id = ?', (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_results(user_id):
    """جيب كل نتايج المستخدم مرتبة بالتاريخ"""
    conn = get_conn()
    rows = conn.execute('''
        SELECT r.*, s.date, c.name as child_name
        FROM results r
        JOIN sessions s ON r.session_id = s.id
        JOIN children c ON s.child_id = c.id
        WHERE s.user_id = ?
        ORDER BY s.date DESC
    ''', (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ============================================================
# Init عند التشغيل
# ============================================================
if __name__ == '__main__':
    init_db()
    print(f'📁 Database location: {DB_PATH}')

    # اختبار سريع
    uid = register_user('test@test.com', '12345678', 'Sarah', 'Ahmed')
    if uid:
        print(f'✅ User created: id={uid}')
        cid = add_child(uid, 'Omar Ahmed', '2021-03-01', 'male', 'no', 'no')
        print(f'✅ Child created: id={cid}')
        sid = create_session(cid, uid)
        print(f'✅ Session created: id={sid}')
        save_result(sid, model1_prob=35.5, model1_gb=34.0, model1_tabt=37.0,
                    model2_prob=40.0, model2_severity='Mild',
                    model3_prob=38.0, model3_seqs=146,
                    final_prob=37.0, risk_level='Moderate')
        print(f'✅ Result saved!')
        print(f'\n📊 All results: {get_all_results(uid)}')
    else:
        print('ℹ️  User already exists — testing login')
        user = login_user('test@test.com', '12345678')
        print(f'✅ Login: {user["first_name"]} {user["last_name"]}')
