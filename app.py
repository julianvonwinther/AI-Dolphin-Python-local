import os
import sqlite3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from llama_cpp import Llama
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'md', 'py', 'js', 'html', 'css'}
MODEL_PATH = os.path.join("models", "dolphin-llama3.gguf")
N_GPU_LAYERS = -1
DATABASE_FILE = 'chat_history.db'

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    ''')
    conn.commit()
    conn.close()

# --- Model Loading ---
llm = None
if os.path.exists(MODEL_PATH):
    print("Loading model... This may take a moment.")
    try:
        llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, n_ctx=4096, verbose=False)
        print("✅ Model loaded successfully onto GPU!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}")

# --- API Routes ---
@app.route('/chats', methods=['GET', 'POST'])
def manage_chats():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if request.method == 'GET':
        cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
        chats = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(chats)
    elif request.method == 'POST':
        new_chat_title = request.json.get('title', 'New Chat')
        cursor.execute("INSERT INTO chats (title) VALUES (?)", (new_chat_title,))
        new_chat_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"id": new_chat_id, "title": new_chat_title}), 201

@app.route('/chats/<int:chat_id>', methods=['GET', 'DELETE'])
def manage_single_chat(chat_id):
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if request.method == 'GET':
        cursor.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,))
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(messages)
    elif request.method == 'DELETE':
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        conn.close()
        return jsonify({"message": "Chat deleted successfully"})

@app.route('/chats/<int:chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    if llm is None: return jsonify({"error": "Model not loaded"}), 500
    data = request.json
    user_content = data.get('content')
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, 'user', user_content))
    conn.commit()
    cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,))
    history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    print(f"Generating response for chat {chat_id}...")
    output = llm.create_chat_completion(messages=history, max_tokens=2048)
    ai_response = output['choices'][0]['message']['content'].strip()
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, 'assistant', ai_response))
    conn.commit()
    conn.close()
    print("Response generated and saved.")
    return jsonify({"role": "assistant", "content": ai_response})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    if llm is None: return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files or 'prompt' not in request.form: return jsonify({"error": "File or prompt missing"}), 400

    file = request.files['file']
    prompt = request.form['prompt']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_content = file.read().decode('utf-8')
        analysis_prompt = f"USER: Here is the content of the document '{filename}':\n\n---\n{file_content}\n---\n\nNow, please answer my question: {prompt}\nASSISTANT:"
        output = llm(analysis_prompt, max_tokens=2048, stop=["USER:"], echo=False)
        analysis_result = output['choices'][0]['text'].strip()
        return jsonify({"response": analysis_result})
    return jsonify({"error": "Invalid file type"}), 400

# --- NEW: Route to serve the frontend ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- Run the Server ---
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)