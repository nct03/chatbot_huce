from flask import Flask, jsonify, request
import pyodbc
import os

app = Flask(__name__)

# DATABASE ===============================================================================
# Hàm kết nối DB
def get_db_connection():
    server = ''
    database = 'Chatbot'
    username = ''
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};Trusted_Connection=yes'
    return pyodbc.connect(conn_str)

# Tạo bảng nếu chưa tồn tại
conn = get_db_connection()
cursor = conn.cursor()

create_table_query = '''
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Documents')
BEGIN
    CREATE TABLE Documents (
        id INT IDENTITY(1,1) PRIMARY KEY,
        filename NVARCHAR(255) UNIQUE,
        content NTEXT
    );
END;
'''

create_chunks_table = '''
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Chunks')
BEGIN
    CREATE TABLE Chunks (
    chunk_id INT IDENTITY(1,1) PRIMARY KEY,
    document_id INT,
    chunk_text NVARCHAR(MAX),  -- Đổi từ NTEXT sang NVARCHAR(MAX)
    FOREIGN KEY (document_id) REFERENCES Documents(id) ON DELETE CASCADE
);

END;
'''

cursor.execute(create_table_query)
cursor.execute(create_chunks_table)
conn.commit()
conn.close()

# Cấu hình lưu file ===============================================================================
UPLOAD_FOLDER = 'Database/Data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Tạo thư mục nếu chưa có
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Kiểm tra file đã có trong DB chưa
def is_file_in_database(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT COUNT(*) FROM Documents WHERE filename = ?"
    cursor.execute(query, (filename,))
    result = cursor.fetchone()[0] > 0
    conn.close()
    return result

# Lưu file vào DB (tránh lặp lại)
def save_all_txt_in_folder():
    conn = get_db_connection()
    cursor = conn.cursor()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if is_file_in_database(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        query = "INSERT INTO Documents (filename, content) VALUES (?, ?)"
        cursor.execute(query, (file_path, content))
        conn.commit()
    conn.close()

# Chia nhỏ văn bản (tránh trùng lặp chunks)
def read_and_chunk_files_from_folder(chunk_size=512, overlap=20):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM Documents")
    documents = cursor.fetchall()
    for document_id, content in documents:
        cursor.execute("SELECT COUNT(*) FROM Chunks WHERE document_id = ?", (document_id,))
        if cursor.fetchone()[0] > 0:
            continue  # Bỏ qua nếu đã chia
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size - overlap)]
        for chunk in chunks:
            cursor.execute("INSERT INTO Chunks (document_id, chunk_text) VALUES (?, ?)", (document_id, chunk))
        conn.commit()
    conn.close()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Chỉ hỗ trợ file .txt"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    if os.path.exists(file_path):
        return jsonify({"error": "File đã tồn tại"}), 400
    file.save(file_path)
    save_all_txt_in_folder()
    read_and_chunk_files_from_folder()
    return jsonify({"message": "Tải lên thành công!", "file_path": file_path}), 200

if __name__ == '__main__':
    save_all_txt_in_folder()
    read_and_chunk_files_from_folder()
    app.run(debug=True)
