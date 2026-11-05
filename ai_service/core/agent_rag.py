"""
================================================================================
HỆ THỐNG LA BÀN KỸ NĂNG (AGENT-RAG SYSTEM)
Module: core/agent_rag.py
Chức năng: Kiến trúc Multi-Agent Workflow phối hợp 4 tác tử chuyên biệt:
  - Agent 1: CVProfilerAgent (Bóc tách hồ sơ, trích xuất thực thể & kỹ năng tham chiếu ESCO)
  - Agent 2: JobRetrievalAgent (RAG Semantic Retriever dựa trên SentenceTransformer & FAISS)
  - Agent 3: SkillGapAgent (Đánh giá khoảng cách kỹ năng & đối sánh phả hệ tri thức - Concept Taxonomy)
  - Agent 4: CareerAdvisorAgent (Cố vấn nghề nghiệp, sinh lộ trình học tập thích ứng dựa trên bằng chứng dữ liệu)
Đồ án Chuyên đề Công nghệ Thông tin - Trường Đại học Thủy Lợi
GVHD: Ts. Hoàng Văn Quý | SVTH: Nguyễn Thị Tuyết Nhung (Lớp S26-K65-CNTT)
================================================================================
"""

import os
import sys
import json
import re
import csv
import hashlib
from pathlib import Path
from collections import Counter

def calculate_file_sha256(filepath):
    """Tính toán mã băm SHA-256 của file để đối soát tính toàn vẹn khoa học"""
    try:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

# Tối ưu hóa môi trường offline cho Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

try:
    import pymupdf  # modern fitz
except ImportError:
    import fitz as pymupdf

try:
    import docx
except ImportError:
    docx = None


def resolve_dataset_dir(custom_path=None):
    """Tìm kiếm thư mục dataset linh hoạt theo đường dẫn động tương đối"""
    if custom_path and os.path.exists(custom_path):
        return Path(custom_path).resolve()
        
    curr = Path(__file__).resolve().parent
    candidates = [
        curr.parent / "dataset",               # ai_service/dataset
        curr.parent.parent / "dataset",        # <root>/dataset
        curr.parent.parent / "CareerLens-AI" / "dataset"
    ]
    for c in candidates:
        if c.exists() and (c / "sample_jobs.json").exists():
            return c
    return candidates[0]


class CVProfilerAgent:
    """Agent 1: Bóc tách cấu trúc hồ sơ sinh viên và nhận diện kỹ năng tham chiếu theo khung ESCO"""
    def __init__(self, skills_csv_path):
        self.skills_map = {}
        self.skills_category = {}
        self.skills_parent = {}
        self.skills_esco = {}
        
        if skills_csv_path and os.path.exists(skills_csv_path):
            with open(skills_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    norm = row['normalized_name'].lower().strip()
                    display_name = row['skill_name'].lower().strip()
                    self.skills_map[norm] = row['skill_name']
                    self.skills_map[display_name] = row['skill_name']
                    self.skills_category[norm] = row.get('category', 'Technical Skill')
                    self.skills_category[display_name] = row.get('category', 'Technical Skill')
                    self.skills_parent[norm] = row.get('parent_concept', row.get('category', 'Technical Skill'))
                    self.skills_esco[norm] = row.get('esco_id', f'esco:ict-{norm}')

    def parse_document(self, file_path_or_text):
        """Trích xuất văn bản từ file PDF, DOCX, TXT hoặc nhận chuỗi văn bản trực tiếp"""
        if isinstance(file_path_or_text, str) and os.path.exists(file_path_or_text):
            ext = os.path.splitext(file_path_or_text)[1].lower()
            if ext == '.pdf':
                doc = pymupdf.open(file_path_or_text)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text
            elif ext == '.docx':
                if docx is not None:
                    doc = docx.Document(file_path_or_text)
                    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    return text
                else:
                    with open(file_path_or_text, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
            else:
                with open(file_path_or_text, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        return str(file_path_or_text)

    def extract_metadata(self, raw_text):
        """Trích xuất email, số điện thoại và họ tên ứng viên có sàng lọc tiêu đề và chức danh"""
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
        phone_match = re.search(r'(0\d{2,3}[\s\.-]?\d{3}[\s\.-]?\d{3,4})', raw_text)
        
        # 1. Ưu tiên dòng có nhãn tường minh
        label_pattern = re.search(r'^(?:họ\s*(?:và|&)?\s*tên|họ\s*tên|tên\s*ứng\s*viên|full\s*name|name)\s*[:\-]\s*(.+)$', raw_text, re.MULTILINE | re.IGNORECASE)
        if label_pattern:
            cand_name = label_pattern.group(1).strip()
            if 3 <= len(cand_name) <= 35 and not any(ch in cand_name for ch in '@:/\\{}[]0123456789'):
                return {
                    "name": cand_name,
                    "email": email_match.group(0) if email_match else "Chưa cập nhật",
                    "phone": phone_match.group(0) if phone_match else "Chưa cập nhật"
                }
        
        ignore_keywords = [
            'curriculum vitae', 'resume', 'hồ sơ ứng viên', 'hồ sơ xin việc',
            'thông tin cá nhân', 'thông tin ứng viên', 'cv ứng tuyển', 'cv',
            'giới thiệu bản thân', 'mục tiêu nghề nghiệp', 'học vấn', 'kinh nghiệm',
            'bản tự thuật', 'họ và tên', 'tên', 'kỹ năng', 'skills', 'objective', 'education', 'experience'
        ]
        job_title_pattern = re.compile(
            r'\b(?:developer|engineer|lập trình viên|chuyên viên|fresher|junior|senior|intern|thực tập|backend|frontend|fullstack|full\s*stack|mobile|tester|qa|qc|devops|data\s*analyst|data\s*engineer|ai\s*engineer)\b',
            re.IGNORECASE
        )
        tech_words_pattern = re.compile(
            r'(?:^|[\s,.:;!?\(\)\[\]])(?:java|python|react|reactjs|vue|vuejs|spring|flutter|dart|\.net|c\+\+|c#|sql|mysql|postgresql|docker|kubernetes|aws)(?:$|[\s,.:;!?\(\)\[\]])',
            re.IGNORECASE
        )
        
        candidate_name = "Sinh viên CNTT"
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        for line in lines[:8]:
            clean_l = line.lower().strip()
            # Bỏ qua dòng là tiêu đề tài liệu
            if any(kw == clean_l or clean_l.startswith(kw + ':') for kw in ignore_keywords):
                continue
            # Bỏ qua dòng là chức danh công việc
            if job_title_pattern.search(clean_l):
                continue
            # Bỏ qua dòng chứa các từ khóa công nghệ chính
            if tech_words_pattern.search(clean_l):
                continue
                
            sub_name = re.sub(r'^(họ\s*(và|&)?\s*tên|tên\s*ứng\s*viên|name)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
            if 3 <= len(sub_name) <= 35 and not any(ch in sub_name for ch in '@:/\\{}[]0123456789'):
                words = sub_name.split()
                if 2 <= len(words) <= 5:
                    candidate_name = sub_name
                    break

        return {
            "name": candidate_name,
            "email": email_match.group(0) if email_match else "Chưa cập nhật",
            "phone": phone_match.group(0) if phone_match else "Chưa cập nhật"
        }

    def extract_skills(self, text):
        """
        Trích xuất kỹ năng bằng Lookaround Regex biên tự do.
        Xử lý chuẩn xác các ký tự đặc biệt C++, C#, .NET, .NET Core,
        không bỏ sót dấu chấm ngắt câu và tránh nhận diện Java trong JavaScript.
        """
        clean = text.lower()
        found_skills = set()
        
        # Sắp xếp theo chiều dài giảm dần để ưu tiên từ ghép dài trước
        sorted_keys = sorted(self.skills_map.keys(), key=lambda x: len(x), reverse=True)
        
        for k in sorted_keys:
            pattern = r'(?<![\w#\+])' + re.escape(k) + r'(?![\w#\+]|\.[\w])'
            if re.search(pattern, clean):
                found_skills.add(self.skills_map[k])
                # Masking đoạn đã tìm thấy để tránh các từ ngắn con bị trùng
                clean = re.sub(pattern, ' ' * len(k), clean)
                
        return sorted(list(found_skills))

    def analyze(self, file_path_or_text):
        raw_text = self.parse_document(file_path_or_text)
        metadata = self.extract_metadata(raw_text)
        skills = self.extract_skills(raw_text)
        
        # Phân loại chuyên ngành dựa trên tần suất kỹ năng (Argmax Domain Scoring)
        domain_groups = {
            "Backend Development": [
                'spring boot', 'java', '.net core', '.net', 'django', 'fastapi', 'golang', 'nodejs', 
                'expressjs', 'php', 'laravel', 'c#', 'c++', 'sql', 'mysql', 'postgresql', 'mongodb', 
                'redis', 'kafka', 'microservices', 'hibernate', 'jpa'
            ],
            "Frontend Development": [
                'reactjs', 'vuejs', 'angular', 'html5', 'css3', 'tailwind css', 'javascript', 
                'typescript', 'nextjs', 'redux', 'bootstrap'
            ],
            "Mobile Development": [
                'flutter', 'dart', 'android sdk', 'swift', 'react native', 'ios'
            ],
            "AI & Machine Learning": [
                'pytorch', 'tensorflow', 'machine learning', 'deep learning', 'nlp', 'computer vision', 
                'rag', 'llm', 'scikit-learn', 'opencv'
            ],
            "Data Engineering & Analytics": [
                'spark', 'hadoop', 'kafka', 'pandas', 'numpy', 'power bi', 'tableau', 'etl'
            ],
            "DevOps & Cloud": [
                'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'ci/cd', 'terraform', 'linux', 'jenkins'
            ],
            "Software Testing & QA": [
                'selenium', 'postman', 'junit', 'unit testing', 'api testing', 'jmeter'
            ]
        }
        
        skills_lower = [s.lower() for s in skills]
        scores = {d: 0 for d in domain_groups}
        for s in skills_lower:
            for d, group in domain_groups.items():
                if s in group:
                    scores[d] += 1
                    
        best_domain = max(scores, key=scores.get)
        domain = best_domain if scores[best_domain] > 0 else "Công nghệ Thông tin Tổng quát"
            
        return {
            "metadata": metadata,
            "domain": domain,
            "skills": skills,
            "skills_count": len(skills),
            "raw_text": raw_text
        }


class JobRetrievalAgent:
    """Agent 2: RAG Retriever - Truy xuất việc làm ngữ nghĩa qua FAISS Vector Database"""
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', index_path=None, jobs_path=None):
        # Định vị thư mục mô hình SentenceTransformer offline theo thứ tự ưu tiên
        current_dir = Path(__file__).resolve().parent
        model_candidates = [
            current_dir.parent / "models" / "minilm",               # ai_service/models/minilm
            current_dir.parent.parent / "models" / "minilm",        # <root>/models/minilm
            current_dir.parent.parent / "CareerLens-AI" / "models" / "minilm"
        ]
        
        found_model_dir = None
        for candidate in model_candidates:
            if candidate.exists() and (candidate / "config.json").exists():
                found_model_dir = str(candidate.resolve())
                break
                
        if found_model_dir:
            print(f" [AI Agent 2] Nạp thành công mô hình Sentence-Transformers offline từ:")
            print(f"   -> {found_model_dir}")
            self.model = SentenceTransformer(found_model_dir, local_files_only=True)
            self.model_source = found_model_dir
        else:
            err_msg = (
                "\n[LỖI NGHIÊM TRỌNG]: Không tìm thấy thư mục mô hình ngôn ngữ MiniLM offline!\n"
                "Hệ thống đã tìm tại các đường dẫn sau:\n" +
                "\n".join([f"  - {str(c)}" for c in model_candidates]) +
                "\nĐể đảm bảo hoạt động 100% khi không có mạng, vui lòng đặt thư mục mô hình tại: "
                "ai_service/models/minilm (phải chứa config.json và model.safetensors)."
            )
            print(err_msg, file=sys.stderr)
            raise FileNotFoundError(err_msg)
            
        self.index = None
        self.jobs = []
        self.index_path = index_path
        self.jobs_path = jobs_path
        
        if jobs_path and os.path.exists(jobs_path):
            with open(jobs_path, 'r', encoding='utf-8') as f:
                self.jobs = json.load(f)
                
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

    def build_and_save_index(self, jobs_path, index_path):
        with open(jobs_path, 'r', encoding='utf-8') as f:
            self.jobs = json.load(f)
            
        job_texts = []
        for j in self.jobs:
            req = j.get('requirements_text', '')
            skills = ", ".join(j.get('technical_skills', []))
            text = f"{j.get('job_title', '')}. Lĩnh vực: {j.get('category', '')}. Kỹ năng: {skills}. Yêu cầu: {req}"
            job_texts.append(text)
            
        embeddings = self.model.encode(job_texts, convert_to_numpy=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, index_path)
        self.index_path = index_path
        print(f"Xây dựng thành công FAISS Index cho {len(self.jobs)} tin việc làm tại {index_path}!")

    def retrieve(self, query_text, top_k=5):
        if not self.index:
            raise ValueError("Chưa nạp hoặc xây dựng FAISS Index!")
            
        query_vector = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        similarities, indices = self.index.search(query_vector, top_k)
        
        retrieved_jobs = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.jobs):
                continue
            job = dict(self.jobs[idx])
            sim_score = float(similarities[0][i])
            pct = max(0, min(100, round((sim_score + 1) / 2 * 100, 1)))
            job['semantic_similarity'] = sim_score
            job['match_percentage'] = pct
            retrieved_jobs.append(job)
            
        return retrieved_jobs


class SkillGapAgent:
    """
    Agent 3: Phân tích khoảng cách kỹ năng (Skill Gap Analysis) kết hợp phả hệ tri thức (Concept Taxonomy).
    Giải quyết hiệu quả hạn chế so khớp từ khóa của các ATS truyền thống thông qua kiến trúc lai (Hybrid Architecture)
    kết hợp Vector Search và Concept Taxonomy.
    """
    def __init__(self, profiler_agent, taxonomy_path=None):
        self.profiler = profiler_agent
        default_taxonomy = {
            "Frontend Framework": ["React", "ReactJS", "Vue", "VueJS", "Angular", "Next.js", "Nuxt.js", "Svelte"],
            "Backend Framework": ["Spring Boot", "Spring", "Django", "FastAPI", "ExpressJS", "NestJS", ".NET Core", ".NET", "ASP.NET", "Laravel"],
            "Relational Database": ["MySQL", "PostgreSQL", "SQL Server", "Oracle", "SQLite", "SQL"],
            "Database Management": ["MySQL", "PostgreSQL", "SQL Server", "Oracle", "SQLite", "MongoDB", "Redis", "SQL"],
            "NoSQL Database": ["MongoDB", "Redis", "Cassandra", "Elasticsearch"],
            "Mobile Framework": ["Flutter", "React Native", "Android SDK", "Swift", "iOS Development"],
            "Cloud Services": ["AWS", "Google Cloud (GCP)", "Microsoft Azure", "Azure", "Firebase"],
            "Containerization": ["Docker", "Kubernetes", "Podman"],
            "Version Control": ["Git", "GitHub", "GitLab"],
            "Data Science & AI": ["PyTorch", "TensorFlow", "Scikit-Learn", "Machine Learning", "Deep Learning", "Pandas", "NumPy"]
        }
        
        target_path = None
        if taxonomy_path and os.path.exists(taxonomy_path):
            target_path = Path(taxonomy_path).resolve()
        else:
            auto_tax = resolve_dataset_dir() / "concept_taxonomy.json"
            if auto_tax.exists():
                target_path = auto_tax.resolve()

        if target_path and target_path.exists():
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    self.concept_taxonomy = json.load(f)
                self.taxonomy_source = str(target_path)
                self.taxonomy_sha256 = calculate_file_sha256(target_path)
                print(f" [AI Agent 3] Nạp thành công Concept Taxonomy từ:\n   -> {self.taxonomy_source} (SHA-256: {self.taxonomy_sha256})")
            except Exception as e:
                print(f" [AI Agent 3 CẢNH BÁO] Lỗi đọc Concept Taxonomy từ {target_path} ({e}). Dùng default_taxonomy viết cứng.")
                self.concept_taxonomy = default_taxonomy
                self.taxonomy_source = "embedded_fallback"
                self.taxonomy_sha256 = "N/A (fallback)"
        else:
            print(" [AI Agent 3 CẢNH BÁO] Không tìm thấy file concept_taxonomy.json. Dùng default_taxonomy viết cứng.")
            self.concept_taxonomy = default_taxonomy
            self.taxonomy_source = "embedded_fallback"
            self.taxonomy_sha256 = "N/A (fallback)"

    def evaluate_gap(self, candidate_skills, job):
        job_skills = list(job.get('technical_skills', []))
        req_extracted = self.profiler.extract_skills(job.get('requirements_text', ''))
        all_required = sorted(list(set(job_skills + req_extracted)))
        cand_lower_map = {s.lower(): s for s in candidate_skills}
        
        matched = []
        missing = []
        used_candidate_skills = set()
        
        for req in all_required:
            req_l = req.lower()
            if req_l in cand_lower_map:
                real_skill = cand_lower_map[req_l]
                matched.append(real_skill)
                used_candidate_skills.add(real_skill)
            elif req in self.concept_taxonomy:
                sub_list = self.concept_taxonomy[req]
                sub_matched = [sub for sub in sub_list if sub.lower() in cand_lower_map]
                if sub_matched:
                    real_sub = cand_lower_map[sub_matched[0].lower()]
                    matched.append(f"{req} (qua {real_sub})")
                    used_candidate_skills.add(real_sub)
                else:
                    missing.append(req)
            else:
                missing.append(req)
                
        extra = [s for s in candidate_skills if s not in used_candidate_skills]
        
        # Regularized Skill Overlap Score (tránh thiên vị các job có quá ít kỹ năng 1-2 từ)
        if len(all_required) > 0:
            effective_denom = max(len(all_required), 3)
            skill_score = min(100.0, round((len(matched) / effective_denom) * 100, 1))
        else:
            skill_score = 60.0
            
        semantic_score = job.get('match_percentage', 65.0)
        overall_fit = round(semantic_score * 0.6 + skill_score * 0.4, 1)
        
        return {
            "job_id": job.get('job_id'),
            "job_title": job.get('job_title'),
            "company": job.get('company'),
            "category": job.get('category', ''),
            "location": job.get('location'),
            "salary": job.get('salary'),
            "experience": job.get('experience_required'),
            "overall_fit_score": overall_fit,
            "semantic_score": semantic_score,
            "skill_overlap_score": skill_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra,
            "total_required_count": len(all_required)
        }


class CareerAdvisorAgent:
    """
    Agent 4: Cố vấn nghề nghiệp và sinh lộ trình thích ứng (RAG Generator).
    Thực hiện theo cơ chế Fact-Grounded Rule-Based Synthesis: sinh văn bản có cấu trúc
    dựa trên luật chặt chẽ và căn cứ dữ liệu đối sánh RAG. Đảm bảo tốc độ phản hồi tức thì (<50ms),
    vận hành hoàn toàn offline, đảm bảo tính xác định và có căn cứ dữ liệu đối soát minh bạch.
    Đồng thời trang bị sẵn cấu trúc Prompt chuẩn cho kết nối LLM cục bộ (Ollama/Qwen/Gemma) trong giai đoạn tiếp theo.
    """
    def generate_report(self, candidate_profile, top_evaluations):
        name = candidate_profile["metadata"]["name"]
        domain = candidate_profile["domain"]
        cand_skills = candidate_profile["skills"]
        
        best_job = top_evaluations[0] if top_evaluations else {}
        
        all_missing = []
        for ev in top_evaluations[:3]:
            all_missing.extend(ev.get("missing_skills", []))
            
        missing_freq = Counter(all_missing).most_common(5)
        top_critical_missing = [s for s, _ in missing_freq if not s.startswith("esco:")]
        
        # Sắp xếp kỹ năng tiêu biểu theo mức độ liên quan mật thiết nhất với Vị trí Tuyển dụng Top 1
        # (Ưu tiên ngôn ngữ cốt lõi, framework chính và CSDL thay vì cắt theo thứ tự bảng chữ cái)
        title_low = (best_job.get('job_title') or '').lower()
        title_tokens = set(re.findall(r'[a-z0-9+#\.]+', title_low))
        matched_clean = [m.lower().split('(')[0].strip() for m in best_job.get('matched_skills', [])]
        
        def skill_relevance_score(skill_name):
            s_low = skill_name.lower().strip()
            score = 0
            s_tokens = [w for w in re.findall(r'[a-z0-9+#\.]+', s_low) if len(w) > 1 or w in {'c', 'r'}]
            
            # 1. Trùng chính xác toàn bộ tên kỹ năng trong tiêu đề theo ranh giới từ
            if re.search(r'\b' + re.escape(s_low) + r'\b', title_low):
                score += 25
            # 2. Hoặc trùng token có nghĩa (> 1 ký tự), tránh "java" khớp "javascript"
            elif any(w in title_tokens for w in s_tokens if len(w) > 1):
                if s_low == "java" and "javascript" in title_tokens and "java" not in title_tokens:
                    pass
                else:
                    score += 20
            # 3. Kỹ năng 1 ký tự ("c", "r"): chỉ tính nếu xuất hiện độc lập như từ nguyên
            elif s_low in {'c', 'r'} and s_low in title_tokens:
                score += 20

            # Thuộc nhóm công nghệ cốt lõi theo chuyên ngành
            core_techs = ['java', 'spring', 'sql', 'hibernate', 'jpa', 'react', 'vue', 'flutter', 'dart', 'python', 'pytorch']
            if any(re.search(r'\b' + re.escape(k) + r'\b', s_low) for k in core_techs):
                score += 10

            # Nằm trong danh sách kỹ năng đáp ứng yêu cầu của Job
            if s_low in matched_clean:
                score += 5

            # Công cụ hỗ trợ, hạ tầng (Git, Docker, Postman, Linux) xếp sau ngôn ngữ & framework
            support_tools = ['git', 'docker', 'linux', 'postman', 'oop', 'agile']
            if any(re.search(r'\b' + re.escape(k) + r'\b', s_low) for k in support_tools):
                score -= 4

            return score

        sorted_featured = sorted(cand_skills, key=skill_relevance_score, reverse=True)
        preview_skills = ', '.join(sorted_featured[:5]) if sorted_featured else 'N/A'
        
        summary = (
            f"Ứng viên **{name}** có định hướng chuyên môn rõ rệt thuộc lĩnh vực **{domain}**. "
            f"Hồ sơ ghi nhận nền tảng kỹ thuật vững với **{len(cand_skills)} kỹ năng chuyên môn** (tiêu biểu như {preview_skills}). "
            f"Vị trí tuyển dụng đối sánh tối ưu nhất hiện nay là **{best_job.get('job_title')}** tại **{best_job.get('company')}** "
            f"với Điểm Phù Hợp Tổng Hợp đạt **{best_job.get('overall_fit_score')}%** "
            f"(Điểm Tương Đồng Ngữ Nghĩa đạt {best_job.get('semantic_score')}%, Tỷ Lệ Đáp Ứng Kỹ Năng đạt {best_job.get('skill_overlap_score')}%)."
        )
        
        w1_gap = top_critical_missing[0] if len(top_critical_missing) > 0 else "Nâng cao kiến trúc RESTful API & Clean Code"
        w2_gap = top_critical_missing[1] if len(top_critical_missing) > 1 else "Tối ưu hóa Truy vấn Cơ sở Dữ liệu & Caching"
        w3_gap = top_critical_missing[2] if len(top_critical_missing) > 2 else "Đóng gói Container Docker & Quy trình CI/CD"
        
        if len(top_critical_missing) <= 2:
            if "Backend" in domain:
                w3_gap = "Docker Containerization & Kiến trúc Microservices"
            elif "Frontend" in domain:
                w3_gap = "Tối ưu Web Vitals & Quản lý State nâng cao (Redux Toolkit)"
            elif "Mobile" in domain:
                w3_gap = "Quản lý Local State & Tối ưu hiệu năng ứng dụng di động"
            elif "AI" in domain:
                w3_gap = "Đóng gói mô hình Machine Learning & Xây dựng API suy luận"
                
        w4_action = f"Hoàn thiện dự án mẫu áp dụng {w1_gap}, đưa mã nguồn lên GitHub cá nhân, viết tài liệu README chuẩn chỉ và luyện phỏng vấn kỹ thuật cho vị trí {best_job.get('job_title', 'Lập trình viên')}."
        
        roadmap = [
            {
                "week": "Tuần 1",
                "focus": f"Bổ sung lỗ hổng cốt lõi: {w1_gap}",
                "action": f"Nghiên cứu tài liệu chính thức, hoàn thành 2 bài thực hành nhỏ áp dụng trực tiếp {w1_gap}."
            },
            {
                "week": "Tuần 2",
                "focus": f"Mở rộng kỹ năng nâng cao: {w2_gap}",
                "action": f"Áp dụng best practices về kiến trúc, tích hợp {w2_gap} vào hệ thống đã xây dựng từ Tuần 1."
            },
            {
                "week": "Tuần 3",
                "focus": f"Hạ tầng & Chuẩn hóa triển khai: {w3_gap}",
                "action": f"Thực hành đóng gói, cấu hình môi trường chuẩn doanh nghiệp với {w3_gap}."
            },
            {
                "week": "Tuần 4",
                "focus": "Đóng gói Portfolio & Luyện phỏng vấn thực tế",
                "action": w4_action
            }
        ]
        
        interview_tips = [
            f"Giải thích nguyên lý và kiến trúc của các công nghệ nòng cốt trong CV ({', '.join(cand_skills[:3])}).",
            f"Chuẩn bị phương án giải quyết tình huống kỹ thuật khi dự án yêu cầu kỹ năng mới như: {', '.join(top_critical_missing[:2]) if top_critical_missing else 'công nghệ Cloud/DevOps'}.",
            "Trình bày cấu trúc dự án thực tế theo mô hình STAR (Situation, Task, Action, Result) để chứng minh năng lực thực chiến với hội đồng và nhà tuyển dụng."
        ]
        
        return {
            "executive_summary": summary,
            "critical_missing_skills": top_critical_missing,
            "learning_roadmap": roadmap,
            "interview_tips": interview_tips,
            "generation_method": "Fact-Grounded Rule-Based Synthesis (Offline Deterministic Mode)"
        }

    def format_llm_prompt(self, candidate_profile, top_evaluations):
        """Tạo prompt RAG tiêu chuẩn sẵn sàng tích hợp với LLM cục bộ (Ollama/Qwen/Gemma)"""
        best_job = top_evaluations[0] if top_evaluations else {}
        prompt = f"""[HỆ THỐNG CỐ VẤN NGHỀ NGHIỆP CNTT - AGENT-RAG]
Bối cảnh truy xuất:
- Ứng viên: {candidate_profile['metadata']['name']} (Lĩnh vực: {candidate_profile['domain']})
- Kỹ năng hiện có: {', '.join(candidate_profile['skills'])}
- Việc làm phù hợp nhất: {best_job.get('job_title')} tại {best_job.get('company')}
- Kỹ năng đáp ứng: {', '.join(best_job.get('matched_skills', []))}
- Kỹ năng còn thiếu: {', '.join(best_job.get('missing_skills', []))}

Nhiệm vụ: Hãy sinh bản nhận xét chuyên môn và đề xuất lộ trình 4 tuần giúp ứng viên hoàn thiện khoảng cách năng lực trên.
"""
        return prompt


class AgentRAGPipeline:
    """
    Điều phối luồng công việc đa tác tử (Multi-Agent Workflow Pipeline)
    Áp dụng quy trình Two-Stage Retrieval & Re-ranking:
      - Stage 1: Quét tập ứng viên rộng (Top 20) qua FAISS để tối đa độ phủ ngữ nghĩa.
      - Stage 2: Đánh giá khoảng cách kỹ năng chi tiết và Tái xếp hạng đa mục tiêu (Multi-Objective Re-ranking).
    """
    def __init__(self, dataset_dir=None):
        data_dir = resolve_dataset_dir(dataset_dir)
        skills_csv = data_dir / "it_skills.csv"
        jobs_json = data_dir / "sample_jobs.json"
        faiss_idx = data_dir / "jobs_faiss.index"
        taxonomy_json = data_dir / "concept_taxonomy.json"
        
        self.dataset_dir = data_dir
        self.profiler = CVProfilerAgent(str(skills_csv))
        self.retriever = JobRetrievalAgent(
            index_path=str(faiss_idx) if faiss_idx.exists() else None, 
            jobs_path=str(jobs_json)
        )
        
        if not faiss_idx.exists() and jobs_json.exists():
            self.retriever.build_and_save_index(str(jobs_json), str(faiss_idx))
            
        self.gap_analyzer = SkillGapAgent(
            self.profiler,
            taxonomy_path=str(taxonomy_json) if taxonomy_json.exists() else None
        )
        self.advisor = CareerAdvisorAgent()

    def run_full_pipeline(self, cv_file_or_text, top_k=5):
        # Bước 1: Agent 1 - Bóc tách CV và nhận diện thực thể/kỹ năng
        profile = self.profiler.analyze(cv_file_or_text)
        
        # Bước 2: Agent 2 - RAG Semantic Retriever
        # Stage 1: Quét Top 20 việc làm trong không gian vector FAISS
        query = f"Vị trí: {profile['domain']}. Kỹ năng: {', '.join(profile['skills'])}. {profile['raw_text'][:400]}"
        candidate_pool_size = max(20, top_k * 4)
        retrieved_jobs = self.retriever.retrieve(query, top_k=candidate_pool_size)
        
        # Bước 3: Agent 3 - Đánh giá khoảng cách kỹ năng & đối chiếu phả hệ tri thức cho toàn bộ Top 20
        evaluations = []
        for job in retrieved_jobs:
            ev = self.gap_analyzer.evaluate_gap(profile['skills'], job)
            evaluations.append(ev)
            
        # Stage 2: Tái xếp hạng đa mục tiêu (Multi-Objective Re-ranking)
        # Sắp xếp lại danh sách theo Overall Fit Score (60% Ngữ nghĩa + 40% Kỹ năng)
        evaluations.sort(key=lambda x: x['overall_fit_score'], reverse=True)
        final_top_jobs = evaluations[:top_k]
        
        # Bước 4: Agent 4 - Sinh nhận xét cố vấn và lộ trình học tập thích ứng
        advice = self.advisor.generate_report(profile, final_top_jobs)
        
        return {
            "candidate_profile": profile,
            "top_job_evaluations": final_top_jobs,
            "career_advice": advice,
            "pipeline_meta": {
                "initial_candidates_retrieved": len(retrieved_jobs),
                "final_selected": len(final_top_jobs)
            }
        }


if __name__ == "__main__":
    print("Testing AgentRAGPipeline with enhanced features...")
    pipeline = AgentRAGPipeline()
    sample_cv = resolve_dataset_dir() / "sample_cv_backend_java.txt"
    res = pipeline.run_full_pipeline(str(sample_cv), top_k=3)
    
    print("\n--- AGENT 1: PROFILER ---")
    print("Candidate:", res['candidate_profile']['metadata'])
    print("Domain:", res['candidate_profile']['domain'])
    print("Skills:", res['candidate_profile']['skills'])
    
    print("\n--- AGENT 2 & 3: TWO-STAGE RETRIEVER & SKILL GAP RE-RANKING ---")
    for ev in res['top_job_evaluations']:
        print(f">> {ev['job_title']} ({ev['company']}) - Overall Fit: {ev['overall_fit_score']}% (Matched: {len(ev['matched_skills'])}, Missing: {len(ev['missing_skills'])})")
        
    print("\n--- AGENT 4: ADVISOR ---")
    print(res['career_advice']['executive_summary'])
