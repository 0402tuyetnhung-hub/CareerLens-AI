# LA BÀN KỸ NĂNG (AGENT-RAG SYSTEM)

> **Hệ thống Multi-Agent RAG Phân Tích CV, Đối Sánh Việc Làm và Đánh Giá Khoảng Cách Kỹ Năng Cho Sinh Viên Công Nghệ Thông Tin**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent%20RAG-indigo.svg)](#kiến-trúc-hệ-thống)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#thông-tin-đề-tài)

---

## 📌 THÔNG TIN ĐỀ TÀI

* **Tên đề tài:** Xây dựng hệ thống Agent-RAG phân tích CV, đối sánh việc làm và đánh giá kĩ năng cho sinh viên CNTT
* **Học phần:** Chuyên đề Công nghệ Thông tin
* **Trường:** Trường Đại học Thủy Lợi (Thuyloi University)
* **Sinh viên thực hiện:** Nguyễn Thị Tuyết Nhung (Lớp S26-K65-CNTT)
* **Giảng viên hướng dẫn:** TS. Hoàng Văn Quý

---

## 🏗 KIẾN TRÚC HỆ THỐNG (MULTI-AGENT WORKFLOW)

Hệ thống được thiết kế theo mô hình **Functional Multi-Agent Workflow** với 4 tác tử chuyên biệt, kết hợp quy trình **Two-Stage Retrieval & Re-ranking**:

```
[ Hồ sơ ứng viên CV ] (PDF / DOCX / Text)
         │
         ▼
[ AGENT 1: CV Profiler Agent ]
 ├── Trích xuất Metadata (Họ tên, SĐT, Email)
 ├── Nhận diện 114+ kỹ năng CNTT (Tham chiếu khung ESCO)
 └── Phân loại Chuyên ngành (Argmax Domain Scoring)
         │
         ▼
[ AGENT 2: Job Matcher Agent (Dense Retriever) ]
 ├── Mô hình: SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
 ├── Không gian vector: 384 chiều (Cosine Similarity)
 └── FAISS IndexFlatIP: Quét nhanh Top 20 việc làm từ kho VietJobs
         │
         ▼
[ AGENT 3: Skill Gap Assessor (Hybrid Re-ranking) ]
 ├── Đánh giá ma trận kỹ năng: Matched / Missing / Extra
 ├── Phả hệ tri thức (Concept Taxonomy): Nhận diện quan hệ cha-con (VD: React -> Frontend Framework)
 └── Điểm Phù Hợp Tổng Hợp: Overall Fit = 60% Ngữ nghĩa + 40% Tỷ lệ kỹ năng -> Lọc Top 5
         │
         ▼
[ AGENT 4: Career Advisor Agent (Generator) ]
 ├── Fact-Grounded Rule-Based Synthesis (100% bám sát dữ liệu, loại bỏ hallucination)
 ├── Báo cáo phân tích chuyên môn & Tóm tắt năng lực
 ├── Lộ trình học tập 4 tuần thích ứng theo kỹ năng thiếu hụt
 └── Bộ câu hỏi và gợi ý phỏng vấn kỹ thuật
```

---

## 📊 KẾT QUẢ THỰC NGHIỆM ĐỐI SOÁT KHOA HỌC

Toàn bộ kết quả thực nghiệm được kiểm chứng và đóng gói độc lập trong script `evaluate.py`.

### 1. Khắc phục ca lỗi "Mù ngữ nghĩa" của ATS từ khóa (Jobscan) - Bảng 2
* **Tình huống:** CV ứng viên ghi thành thạo `React`, JD yêu cầu khái niệm chuẩn `Frontend Framework`.
* **Kết quả:**
  * **ATS Từ khóa (Jobscan):** Đánh trượt kỹ năng (`0/1 - MISSING`) do không trùng chuỗi ký tự cứng.
  * **La Bàn Kỹ Năng (Agent-RAG):** Đạt chuẩn (`MATCHED`) qua Cây phả hệ tri thức (Concept Taxonomy). Điểm tương đồng ngữ nghĩa đạt **88.2%** (Cosine = 0.765), Điểm phù hợp tổng hợp (Overall Fit) đạt **92.9%**.

### 2. Độ chính xác truy xuất trên 4 CV Tiêu chuẩn - Bảng 3
Trên 4 bộ hồ sơ ứng viên có từ khóa công nghệ tường minh (Java, React, Flutter, AI):
* **Baseline (TF-IDF):** Precision@1 = 100.0% | Precision@3 = 100.0% | Precision@5 = 100.0%
* **Hệ thống Agent-RAG:** Precision@1 = 100.0% | Precision@3 = 100.0% | Precision@5 = 100.0%

### 3. Đánh giá trên Bộ kiểm thử diễn giải khái niệm (Paraphrase Benchmark - 6 CV) - Bảng 4
Khi ứng viên mô tả năng lực bằng tiếng Việt tự nhiên mà **không dùng từ khóa thương hiệu cứng**, kết quả được đối chiếu song song ở cả hai phiên bản đánh giá:

| Kịch bản đối soát | Baseline Từ khóa (TF-IDF) | Hệ Thống La Bàn Kỹ Năng (Agent-RAG) | Nhận xét đối soát khoa học |
| :--- | :---: | :---: | :--- |
| **Bản 1: Đơn nhãn nghiêm ngặt (Strict)** | Hit@1 = 50.0% (3/6)<br>Hit@3 = 50.0% (3/6)<br>MRR = 0.542 | **Hit@1 = 83.3% (5/6)**<br>**Hit@3 = 100.0% (6/6)**<br>**MRR = 0.889** | Agent-RAG cao hơn (+33.3 điểm phần trăm Hit@1, +50.0 điểm phần trăm Hit@3, +0.347 MRR); TF-IDF trượt 3 CV (CV 1 xuống #4, CV 2 và 5 trượt khỏi Top 5) |
| **Bản 2: Dung thứ liên ngành (Tolerant)** | Hit@1 = 66.7% (4/6)<br>Hit@3 = 66.7% (4/6)<br>MRR = 0.667 | **Hit@1 = 83.3% (5/6)**<br>**Hit@3 = 100.0% (6/6)**<br>**MRR = 0.889** | Dung thứ Fullstack cho CV 1 giúp TF-IDF tăng từ 50% lên 66.7%; Agent-RAG giữ vững 83.3% Hit@1 và 100% Hit@3 |

* **Đối chiếu hai kịch bản:** Khi áp dụng cơ chế dung thứ liên ngành (Bản 2), chỉ số của Baseline TF-IDF tăng từ 50.0% lên Hit@1 = 66.7% và Hit@3 = 66.7% (MRR tăng từ 0.542 lên 0.667), trong khi kết quả của Agent-RAG không thay đổi (Hit@1 = 83.3%, Hit@3 = 100.0%, MRR = 0.889).
* **Phân tích lỗi chi tiết:**
  * TF-IDF trượt hoàn toàn CV 2 (Giao diện web -> Data Engineer) và CV 5 (Đường ống dữ liệu -> Junior Fullstack) khỏi Top 5 ở cả hai bản.
  * Agent-RAG trượt Top 1 tại CV 1 (Dịch vụ máy chủ -> Cloud Infrastructure Engineer do khoảng cách ngữ nghĩa gần giữa ảo hóa hạ tầng và Cloud), nhưng xuất hiện chính xác ở vị trí thứ 3 (Hit@3 = 100.0%).

### 4. Đo lường hiệu năng thực thi (Profiling trên CPU tiêu chuẩn)
* **Cấu hình phần cứng:** CPU 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz, 16 GB RAM, Windows 11 (64-bit), Python 3.12.
* **Độ trễ toàn chuỗi (End-to-End Pipeline - 50 lần lặp độc lập sau 5 lần warm-up):**
  * Thời gian trung bình (Mean): **104.4 ± 11.0 ms**
  * Trung vị (Median / p50): **102.5 ms**
  * Phân vị 95 (p95): **118.6 ms**
  * Khoảng biến thiên (Min — Max): **[83.8 ms — 141.7 ms]**
* **Chi tiết từng tác tử AI chuyên trách:**
  * Agent 1 (CV Profiler - Regex & Domain Scoring): **1.86 ± 0.39 ms**
  * Agent 2 (Job Matcher - Dense Transformer 384D): **48.17 ± 4.97 ms**
  * Agent 2 (FAISS IndexFlatIP Search trên 135 vector): **0.119 ± 0.045 ms**
  * Agent 3 (Skill Gap Assessor): **2.99 ms/job**, tổng thời gian đánh giá 20 việc làm tiềm năng là **≈ 59.9 ms**
  * Agent 4 (Fact-Grounded Rule-Based Synthesis): **0.183 ± 0.063 ms** (< 1ms)
* **Cơ cấu thời gian:** Dense Encode (~48.2 ms) + Skill-Gap ma trận 20 jobs (~59.9 ms) chiếm tỷ trọng chính toàn chuỗi.

---

## 🔒 TÍNH TOÀN VẸN DỮ LIỆU & TÁI LẬP THỰC NGHIỆM

Mã băm SHA-256 của các tệp dữ liệu được công bố minh bạch nhằm bảo đảm tính khách quan:

| Tệp dữ liệu | Đường dẫn | Mã băm SHA-256 |
| :--- | :--- | :--- |
| **Từ điển kỹ năng** | `dataset/it_skills.csv` | `d24f266a0cb51e2d32f4c6c8028527077c3d80f24270c7cc2ce973f3e03c2732` |
| **Tập tin tuyển dụng VietJobs** | `dataset/sample_jobs.json` | `08aff83840c6d74aa5a49d3d5b5ba831ff49b7e3973d292975bd0cc5249bda27` |
| **Phả hệ tri thức (Taxonomy)** | `dataset/concept_taxonomy.json` | `bf307a0edcf8991e2720251874561947306a116d17a512eb0760f7a35c73c1cc` |
| **Bộ kiểm thử diễn giải** | `dataset/paraphrase_benchmark.json` | `9aa6fa6e656157cbb886604953892ee1a8d42d0d4d87cbd1c76c04959414ba25` |

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### 1. Chuẩn bị môi trường
* Python 3.10 trở lên trên hệ điều hành Windows / Linux / macOS.
* Clone mã nguồn:
  ```bash
  git clone https://github.com/0402tuyetnhung-hub/CareerLens-AI.git
  cd CareerLens-AI
  ```

### 2. Thiết lập môi trường ảo và cài đặt thư viện
```bash
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Khởi động ứng dụng Demo (Web Dashboard)
* **Cách 1:** Nhấp đúp chuột vào tệp `01_CHAY_DEMO_LA_BAN_KY_NANG.bat` (Windows).
* **Cách 2:** Khởi chạy bằng dòng lệnh:
  ```bash
  python backend/app.py
  ```
  Truy cập giao diện Web tại địa chỉ: `http://127.0.0.1:8000`

### 4. Tái lập toàn bộ kết quả thực nghiệm khoa học
* **Cách 1:** Nhấp đúp chuột vào tệp `03_CHAY_THUC_NGHIEM_DANH_GIA.bat` (Windows).
* **Cách 2:** Khởi chạy bằng dòng lệnh:
  ```bash
  python evaluate.py
  ```
  Script sẽ tự động đối soát Bảng 2, Bảng 3, Bảng 4 và đo lường thời gian xử lý của toàn bộ 4 tác tử.

---

## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

```
CareerLens-AI/
├── 65CNTT_NguyenThiTuyetNhung_CDCNTT_LaBanKyNang.docx  # Báo cáo chính thức (Chữ đen 100% chuẩn Thủy Lợi)
├── 01_CHAY_DEMO_LA_BAN_KY_NANG.bat                   # Script khởi động giao diện & server API
├── 03_CHAY_THUC_NGHIEM_DANH_GIA.bat                  # Script chạy thực nghiệm đối soát khoa học
├── evaluate.py                                        # Script thực nghiệm đối soát Bảng 2, 3, 4
├── ket_qua_danh_gia.txt                               # Kết quả đo đạc khoa học chi tiết (Bảng 2, 3, 4 & độ trễ)
├── requirements.txt                                   # Danh sách thư viện Python
├── Hinh_KienTruc_AgentRAG.png                         # Sơ đồ kiến trúc 4 Tác tử (Hình 3)
├── Hinh_Dashboard_LaBanKyNang.png                     # Ảnh chụp màn hình giao diện (Hình 4)
├── frontend/                                          # Giao diện người dùng Web Dashboard
│   ├── index.html                                     # Giao diện chính La Bàn Kỹ Năng
│   ├── js/ (app.js, tailwind.min.js)                  # Script điều khiển và render UI
│   ├── css/ (all.min.css)                             # FontAwesome và style giao diện
│   └── webfonts/                                      # Bộ font icon offline 100%
├── backend/                                           # Máy chủ API Backend
│   └── app.py                                         # HTTP API Server phục vụ phân tích & UI
├── dataset/                                           # Tập dữ liệu nghiên cứu chuẩn hóa
│   ├── it_skills.csv                                  # Từ điển 114 kỹ năng tham chiếu ESCO
│   ├── sample_jobs.json                               # 135 tin tuyển dụng VietJobs
│   ├── concept_taxonomy.json                          # Cây phả hệ tri thức công nghệ
│   └── paraphrase_benchmark.json                      # 6 hồ sơ kiểm thử diễn giải khái niệm
└── ai_service/                                        # Lõi tác tử AI (Multi-Agent RAG Core)
    ├── core/agent_rag.py                              # Pipeline điều phối 4 tác tử AI
    └── dataset/ (bản sao lưu dữ liệu cục bộ)
```

---

## 📜 BẢN QUYỀN VÀ TRÍCH DẪN

Đồ án được thực hiện phục vụ mục đích nghiên cứu và bảo vệ học phần Chuyên đề Công nghệ Thông tin tại **Trường Đại học Thủy Lợi**.
Mọi thông tin trích dẫn vui lòng ghi rõ nguồn tác giả: **Nguyễn Thị Tuyết Nhung (Lớp S26-K65-CNTT)**.
