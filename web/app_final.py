"""
AURA Web Application — Final
"""
import os, sys
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import date

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))   # = AURA/web/
AURA_ROOT      = os.path.dirname(BASE_DIR)                      # = AURA/
NOTEBOOKS_DIR  = os.path.join(AURA_ROOT, 'F_Notebooks')        # = AURA/F_Notebooks/
VIDEOS_DIR     = os.path.join(AURA_ROOT, 'Videos')             # = AURA/Videos/
TMP_DIR        = os.path.join(BASE_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# ── sys.path: Flask يلاقي model1_predict.py في F_Notebooks ──
sys.path.insert(0, BASE_DIR)       # AURA/web/
sys.path.insert(0, NOTEBOOKS_DIR)  # AURA/F_Notebooks/

app = Flask(__name__)
app.secret_key = 'aura_secret_key_2026'

from database import (
    init_db, register_user, login_user, get_user,
    add_child, get_children, get_child,
    create_session, get_sessions,
    save_result, get_result, get_all_results
)

# ── Models Lazy Load ──────────────────────────────────────
_m1 = _m2 = _m3 = None

def get_model1():
    global _m1
    if _m1 is None:
        from model1_predict import predict_questionnaire
        _m1 = predict_questionnaire
    return _m1

def get_model2():
    global _m2
    if _m2 is None:
        from model2_predict import predict_eye_image
        _m2 = predict_eye_image
    return _m2

def get_model3():
    global _m3
    if _m3 is None:
        from model3_predict import predict_eye_tracking
        _m3 = predict_eye_tracking
    return _m3

# ── Fusion ────────────────────────────────────────────────
def fusion(m1, m2, m3=None):
    final = (0.4*m1 + 0.2*m2 + 0.4*m3) if m3 is not None else (0.5*m1 + 0.5*m2)
    if   final < 30: risk = 'Low'
    elif final < 55: risk = 'Moderate'
    elif final < 75: risk = 'High'
    else:            risk = 'Very High'
    return round(final, 2), risk

# ── Helpers ───────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def calc_age(dob_str):
    return (date.today() - date.fromisoformat(dob_str)).days // 365

def get_current_user():
    if 'user_id' not in session: return None
    u = get_user(session['user_id'])
    if u:
        children      = get_children(u['id'])
        u['children'] = children
        u['child']    = children[0] if children else None
        if u['child']:
            u['name']         = f"{u['first_name']} {u['last_name']}"
            u['child_name']   = u['child']['name']
            u['child_age']    = u['child']['age']
            u['child_gender'] = u['child']['gender']
    return u

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    # لو المستخدم logged in خد روحه على dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = login_user(
            request.form.get('email','').strip(),
            request.form.get('password','').strip()
        )
        if user:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    uid = register_user(
        request.form.get('email','').strip(),
        request.form.get('password','').strip(),
        request.form.get('fname','').strip(),
        request.form.get('lname','').strip()
    )
    if uid is None:
        return render_template('login.html', reg_error='Email already registered')
    add_child(uid,
        request.form.get('child_name','').strip(),
        request.form.get('dob','').strip(),
        request.form.get('gender','').strip(),
        request.form.get('jaundice','no').strip(),
        request.form.get('family_history','no').strip()
    )
    session.clear()
    session['user_id'] = uid
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    if not user: session.clear(); return redirect(url_for('login'))
    all_results = get_all_results(user['id'])
    # Prepare chart data as plain lists (avoid Jinja2 iterator issues)
    chart_labels = [r['date'][:10] for r in reversed(all_results) if r.get('date')]
    chart_data   = [round(r['final_prob'], 1) for r in reversed(all_results) if r.get('final_prob') is not None]
    return render_template('dashboard.html',
        user=user,
        sessions=get_sessions(user['id']),
        results=all_results,
        chart_labels=chart_labels,
        chart_data=chart_data)

@app.route('/questionnaire')
@login_required
def questionnaire():
    user  = get_current_user()
    child = user.get('child')
    if not child: return redirect(url_for('dashboard'))
    sid = create_session(child['id'], user['id'])
    session['current_session_id'] = sid
    session['current_child_id']   = child['id']
    session.modified = True
    return render_template('questionnaire.html', user=user,
                           q_type='qchat' if child['age'] < 4 else 'aq10')

@app.route('/eye-tracking')
@login_required
def eye_tracking():
    return render_template('eye_tracking.html', user=get_current_user())

@app.route('/results')
@login_required
def results():
    user    = get_current_user()
    all_res = get_all_results(user['id'])
    return render_template('results.html', user=user,
                           latest=all_res[0] if all_res else None,
                           all_results=all_res)

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html', user=get_current_user())

# ── API ───────────────────────────────────────────────────
@app.route('/api/predict/questionnaire', methods=['POST'])
@login_required
def api_questionnaire():
    try:
        data  = request.get_json()
        user  = get_current_user()
        child = user.get('child')
        answers = [int(data.get(f'A{i}',0)) for i in range(1,11)]
        result  = get_model1()(
            answers,
            float(child['age']) if child else 5.0,
            'm' if (child and child['gender']=='male') else 'f',
            child.get('jaundice','no') if child else 'no',
            child.get('family_history','no') if child else 'no'
        )
        session['model1_result'] = result
        session['q_answers']     = answers
        session.modified = True
        sid = session.get('current_session_id')
        if sid:
            save_result(sid, model1_prob=result['final_prob'],
                        model1_gb=result.get('gb_prob'),
                        model1_tabt=result.get('tabt_prob'),
                        q_answers=answers)
        return jsonify({'success':True, 'result':result})
    except Exception as e:
        return jsonify({'success':False, 'error':str(e)}), 500

@app.route('/api/predict/eye-image', methods=['POST'])
@login_required
def api_eye_image():
    tmp = os.path.join(TMP_DIR, f"eye_{session.get('user_id','x')}.jpg")
    try:
        if 'image' not in request.files:
            return jsonify({'success':False,'error':'No image'}), 400
        request.files['image'].save(tmp)
        result = get_model2()(tmp)
        result['severity_prob_pct'] = round(result['severity_prob']*100, 2)
        session['model2_result'] = result
        session.modified = True
        sid = session.get('current_session_id')
        if sid:
            save_result(sid, model2_prob=result['severity_prob_pct'],
                        model2_severity=result.get('severity'))
        return jsonify({'success':True, 'result':result})
    except Exception as e:
        return jsonify({'success':False, 'error':str(e)}), 500
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.route('/api/predict/eye-tracking', methods=['POST'])
@login_required
def api_eye_tracking():
    tmp = os.path.join(TMP_DIR, f"csv_{session.get('user_id','x')}.csv")
    try:
        if 'csv' not in request.files:
            return jsonify({'success':False,'error':'No CSV'}), 400
        request.files['csv'].save(tmp)
        result = get_model3()(tmp)
        session['model3_result'] = result
        session.modified = True
        sid = session.get('current_session_id')
        if sid:
            save_result(sid, model3_prob=result.get('asd_prob'),
                        model3_seqs=result.get('num_sequences'))
        return jsonify({'success':True, 'result':result})
    except Exception as e:
        return jsonify({'success':False, 'error':str(e)}), 500
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.route('/api/predict/fusion', methods=['POST'])
@login_required
def api_fusion():
    try:
        m1 = session.get('model1_result',{})
        m2 = session.get('model2_result',{})
        m3 = session.get('model3_result',{})
        data = request.get_json() or {}
        m1p  = float(data.get('model1_prob') or m1.get('final_prob') or 0)
        m2p  = float(data.get('model2_prob') or m2.get('severity_prob_pct') or 0)
        m3r  = data.get('model3_prob') or m3.get('asd_prob')
        m3p  = float(m3r) if m3r is not None else None
        fp, risk = fusion(m1p, m2p, m3p)
        res = {'model1_prob':round(m1p,2),'model2_prob':round(m2p,2),
               'model3_prob':round(m3p,2) if m3p else None,
               'final_prob':fp,'risk_level':risk,'csv_used':m3p is not None}
        sid = session.get('current_session_id')
        if sid:
            save_result(sid, model1_prob=m1p, model1_gb=m1.get('gb_prob'),
                        model1_tabt=m1.get('tabt_prob'), model2_prob=m2p,
                        model2_severity=m2.get('severity'), model3_prob=m3p,
                        model3_seqs=m3.get('num_sequences'),
                        final_prob=fp, risk_level=risk,
                        q_answers=session.get('q_answers'))
        session['final_results'] = res
        session.modified = True
        return jsonify({'success':True,'result':res})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/api/session-info')
@login_required
def session_info():
    user    = get_current_user()
    results = get_all_results(user['id']) if user else []
    return jsonify({
        'user'         : {'name':user['name'],'email':user['email']} if user else None,
        'child'        : user.get('child') if user else None,
        'model1_result': session.get('model1_result'),
        'model2_result': session.get('model2_result'),
        'model3_result': session.get('model3_result'),
        'final_results': session.get('final_results'),
        'all_results'  : results[:5],
    })

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','model1':_m1 is not None,
                    'model2':_m2 is not None,'model3':_m3 is not None})


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """يخدم ملفات الفيديو من AURA/Videos/"""
    from flask import send_from_directory
    import mimetypes
    # Add mp4 mimetype explicitly
    mimetypes.add_type('video/mp4', '.mp4')
    return send_from_directory(VIDEOS_DIR, filename)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)