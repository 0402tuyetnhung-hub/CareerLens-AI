"""
================================================================================
HỆ THỐNG LA BÀN KỸ NĂNG (AGENT-RAG SYSTEM)
Module: backend/app.py
Chức năng: Máy chủ API Backend & Phục vụ Giao diện người dùng Frontend
Đồ án Chuyên đề Công nghệ Thông tin - Trường Đại học Thủy Lợi
GVHD: Ts. Hoàng Văn Quý | SVTH: Nguyễn Thị Tuyết Nhung (Lớp S26-K65-CNTT)
================================================================================
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Thư mục gốc dự án
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
AI_DIR = os.path.join(ROOT_DIR, 'ai_service')
DATASET_DIR = os.path.join(AI_DIR, 'dataset')

# Nạp module AI từ ai_service (Ưu tiên số 1 trên sys.path để tránh nạp nhầm bản sao cũ)
sys.path.insert(0, AI_DIR)
from core.agent_rag import AgentRAGPipeline

print("=================================================================")
print(" Đang khởi tạo Hệ Thống La Bàn Kỹ Năng - Multi-Agent RAG Pipeline...")
pipeline = AgentRAGPipeline(dataset_dir=DATASET_DIR)
print(f" Module AI Core được nạp chính xác từ:")
print(f"   -> {sys.modules['core.agent_rag'].__file__}")
print(" Khởi tạo hoàn tất! Pipeline AI đã sẵn sàng phục vụ.")
print("=================================================================\n")

# Nạp trước các hồ sơ CV mẫu
sample_cvs = {}
for fname, key in [
    ('sample_cv_backend_java.txt', 'backend_java'),
    ('sample_cv_frontend_react.txt', 'frontend_react'),
    ('sample_cv_mobile_flutter.txt', 'mobile_flutter'),
    ('sample_cv_data_ai.txt', 'data_ai')
]:
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            sample_cvs[key] = f.read()

class BackendAPIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def serve_file(self, file_path, content_type):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.send_header('Connection', 'close')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. API Endpoints
        if path == '/api/stats':
            unique_skills_count = len(set(pipeline.profiler.skills_map.values()))
            stats = {
                "jobs_count": len(pipeline.retriever.jobs),
                "skills_count": unique_skills_count,
                "vector_dim": 384,
                "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                "taxonomy_sha256": getattr(pipeline.gap_analyzer, "taxonomy_sha256", "N/A"),
                "taxonomy_source": getattr(pipeline.gap_analyzer, "taxonomy_source", "N/A")
            }
            self.send_json(stats)
            return

        if path == '/api/sample_cvs':
            self.send_json(sample_cvs)
            return

        # 2. Phục vụ Frontend Static Files
        if path == '/' or path == '/index.html':
            self.serve_file(os.path.join(FRONTEND_DIR, 'index.html'), 'text/html; charset=utf-8')
        elif path.startswith('/js/'):
            fname = os.path.basename(path)
            self.serve_file(os.path.join(FRONTEND_DIR, 'js', fname), 'application/javascript; charset=utf-8')
        elif path.startswith('/css/'):
            fname = os.path.basename(path)
            self.serve_file(os.path.join(FRONTEND_DIR, 'css', fname), 'text/css; charset=utf-8')
        elif path.startswith('/webfonts/'):
            fname = os.path.basename(path)
            content_type = 'font/woff2' if fname.endswith('.woff2') else 'font/ttf' if fname.endswith('.ttf') else 'application/octet-stream'
            self.serve_file(os.path.join(FRONTEND_DIR, 'webfonts', fname), content_type)
        # Tương thích với các đường dẫn gốc
        elif path == '/tailwind.min.js':
            self.serve_file(os.path.join(FRONTEND_DIR, 'js', 'tailwind.min.js'), 'application/javascript; charset=utf-8')
        elif path == '/all.min.css':
            self.serve_file(os.path.join(FRONTEND_DIR, 'css', 'all.min.css'), 'text/css; charset=utf-8')
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # 1. API trích xuất file CV
        if parsed.path == '/api/upload_file':
            try:
                import base64
                import io
                import pymupdf
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                req_json = json.loads(post_data.decode('utf-8'))
                b64_data = req_json.get('file_base64', '')
                filename = req_json.get('filename', 'cv.pdf')
                raw_bytes = base64.b64decode(b64_data)
                
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.pdf':
                    doc = pymupdf.open(stream=raw_bytes, filetype='pdf')
                    text = ""
                    for page in doc:
                        text += page.get_text() + "\n"
                    doc.close()
                elif ext == '.docx':
                    from docx import Document
                    d = Document(io.BytesIO(raw_bytes))
                    parts = [p.text for p in d.paragraphs if p.text.strip()]
                    for t in d.tables:
                        for row in t.rows:
                            for cell in row.cells:
                                if cell.text.strip():
                                    parts.append(cell.text.strip())
                    text = "\n".join(parts)
                else:
                    text = raw_bytes.decode('utf-8', errors='ignore')
                    
                self.send_json({"extracted_text": text.strip(), "filename": filename})
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
                
        # 2. API phân tích chuỗi 4 Tác tử AI (đo thời gian thực thi elapsed_ms thật)
        elif parsed.path == '/api/analyze':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                req_json = json.loads(post_data.decode('utf-8'))
                cv_text = req_json.get('cv_text', '')
                
                t0 = time.perf_counter()
                result = pipeline.run_full_pipeline(cv_text, top_k=5)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                result["elapsed_ms"] = elapsed_ms
                
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": f"Lỗi xử lý quy trình: {str(e)}"}, status=500)
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()

def run_server(port=8000):
    import webbrowser
    import threading
    
    server_address = ('127.0.0.1', port)
    try:
        httpd = ThreadingHTTPServer(server_address, BackendAPIHandler)
    except OSError as err:
        print(f"\n[LỖI KHỞI ĐỘNG]: Cổng {port} hiện đang bị chiếm dụng bởi tiến trình khác ({err}).")
        print(f"👉 Vui lòng kiểm tra và tắt bớt ứng dụng đang chạy cổng {port}, hoặc đổi sang cổng khác (ví dụ: run_server(8080)).\n")
        return
        
    httpd.daemon_threads = True
    
    print(f"\n=======================================================")
    print(f" Hệ Thống La Bàn Kỹ Năng - Web Dashboard đang chạy tại:")
    print(f" 👉 Frontend UI:  http://127.0.0.1:{port}")
    print(f" 👉 Backend API:  http://127.0.0.1:{port}/api/stats")
    print(f"=======================================================\n")
    
    def auto_open():
        time.sleep(1)
        webbrowser.open(f"http://127.0.0.1:{port}")
    
    threading.Thread(target=auto_open, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng máy chủ...")
        httpd.server_close()

if __name__ == '__main__':
    run_server(8000)
