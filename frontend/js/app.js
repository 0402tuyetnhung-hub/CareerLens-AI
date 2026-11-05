// Client-side Controller for La Ban Ky Nang Dashboard
let sampleData = {};

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Fetch sample CVs from backend API on startup
async function loadSampleCVs() {
    try {
        const resp = await fetch('/api/sample_cvs');
        if (resp.ok) {
            sampleData = await resp.json();
            selectSample('backend_java');
        }
    } catch (err) {
        console.warn('Cannot load sample CVs dynamically:', err);
    }
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const label = document.getElementById('fileNameLabel');
    label.innerHTML = '<span class="text-cyan-400 font-bold"><i class="fa-solid fa-spinner fa-spin"></i> Đang đọc file ' + escapeHtml(file.name) + '...</span>';

    const reader = new FileReader();
    reader.onload = async function(e) {
        const base64 = e.target.result.split(',')[1];
        try {
            const resp = await fetch('/api/upload_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_base64: base64, filename: file.name })
            });
            const data = await resp.json();
            if (data.error) {
                alert('Lỗi đọc file: ' + data.error);
                label.innerText = 'Nhấn vào đây để chọn file khác';
                return;
            }
            document.getElementById('cvInput').value = data.extracted_text;
            label.innerHTML = '<span class="text-emerald-400 font-bold"><i class="fa-solid fa-circle-check"></i> Đã nạp thành công: ' + escapeHtml(file.name) + '</span>';
            
            // Reset to waiting state
            const placeholder = document.getElementById('waitingPlaceholder');
            const resContainer = document.getElementById('resultsContainer');
            if (placeholder && resContainer) {
                placeholder.classList.remove('hidden');
                resContainer.classList.add('hidden');
            }
            setStepStatus('step1', 'waiting', '');
            setStepStatus('step2', 'waiting', '');
            setStepStatus('step3', 'waiting', '');
            setStepStatus('step4', 'waiting', '');
        } catch (err) {
            alert('Lỗi tải file: ' + err.message);
            label.innerText = 'Nhấn vào đây để chọn lại';
        }
    };
    reader.readAsDataURL(file);
}

function selectSample(key) {
    document.querySelectorAll('.sample-btn').forEach(b => {
        b.classList.remove('border-indigo-500', 'bg-indigo-950/70', 'ring-1', 'ring-indigo-500');
        b.classList.add('border-slate-700', 'bg-slate-800/80');
    });
    const activeBtn = document.getElementById('btn_' + key);
    if (activeBtn) {
        activeBtn.classList.remove('border-slate-700', 'bg-slate-800/80');
        activeBtn.classList.add('border-indigo-500', 'bg-indigo-950/70', 'ring-1', 'ring-indigo-500');
    }
    if (sampleData[key]) {
        document.getElementById('cvInput').value = sampleData[key];
    }
    // Reset step tracker to waiting
    setStepStatus('step1', 'waiting', '');
    setStepStatus('step2', 'waiting', '');
    setStepStatus('step3', 'waiting', '');
    setStepStatus('step4', 'waiting', '');
    
    // Switch back to waiting placeholder until user clicks Activate
    const placeholder = document.getElementById('waitingPlaceholder');
    const resContainer = document.getElementById('resultsContainer');
    if (placeholder && resContainer) {
        placeholder.classList.remove('hidden');
        resContainer.classList.add('hidden');
    }
}

function setStepStatus(stepId, state, text) {
    const el = document.getElementById(stepId);
    if (!el) return;
    const badge = el.querySelector('.status-badge');
    if (state === 'running') {
        badge.className = 'status-badge text-xs text-cyan-400 animate-pulse font-extrabold';
        badge.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ' + escapeHtml(text);
        el.classList.add('border-cyan-500/60', 'bg-cyan-950/40');
    } else if (state === 'done') {
        badge.className = 'status-badge text-xs text-emerald-400 font-extrabold';
        badge.innerHTML = '<i class="fa-solid fa-check"></i> Hoàn thành';
        el.classList.remove('border-cyan-500/60', 'bg-cyan-950/40');
        el.classList.add('border-emerald-500/40');
    } else {
        badge.className = 'status-badge text-xs font-bold text-slate-400';
        badge.innerHTML = 'Chờ';
        el.className = 'flex items-center justify-between p-3.5 rounded-xl bg-slate-900/70 border border-slate-800 transition';
    }
}

async function runAnalysis() {
    const text = document.getElementById('cvInput').value.trim();
    if (!text) {
        alert('Vui lòng nhập hoặc chọn CV mẫu!');
        return;
    }

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang Phân Tích...';

    // Mô phỏng trực quan trạng thái kích hoạt từng tác tử
    setStepStatus('step1', 'running', 'Đang bóc tách thực thể...');
    setStepStatus('step2', 'waiting', '');
    setStepStatus('step3', 'waiting', '');
    setStepStatus('step4', 'waiting', '');

    try {
        await new Promise(r => setTimeout(r, 200));
        setStepStatus('step1', 'done');
        setStepStatus('step2', 'running', 'Đang quét FAISS 384D...');

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cv_text: text })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || ('Lỗi máy chủ HTTP ' + response.status));
        }

        setStepStatus('step2', 'done');
        setStepStatus('step3', 'running', 'Đo độ vênh & phả hệ tri thức...');
        await new Promise(r => setTimeout(r, 150));
        setStepStatus('step3', 'done');
        setStepStatus('step4', 'running', 'Sinh nhận xét & lộ trình...');

        const data = await response.json();
        setStepStatus('step4', 'done');

        renderResults(data);
    } catch (err) {
        alert('Lỗi phân tích: ' + err.message);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fa-solid fa-play"></i> KÍCH HOẠT QUY TRÌNH AGENT-RAG';
    }
}

function renderResults(data) {
    const placeholder = document.getElementById('waitingPlaceholder');
    const resContainer = document.getElementById('resultsContainer');
    if (placeholder && resContainer) {
        placeholder.classList.add('hidden');
        resContainer.classList.remove('hidden');
    }

    const p = data.candidate_profile;
    document.getElementById('resName').textContent = p.metadata.name || 'Ứng viên CNTT';
    document.getElementById('resContact').textContent = (p.metadata.email || 'Chưa cập nhật') + ' • ' + (p.metadata.phone || 'Chưa cập nhật');
    document.getElementById('resDomain').textContent = p.domain || 'Công nghệ Thông tin';
    document.getElementById('resSkillCount').textContent = p.skills.length;
    
    // Hiển thị thời gian thực thi thật được Backend đo đạc
    const elapsedEl = document.getElementById('resElapsedMs');
    if (elapsedEl) {
        elapsedEl.textContent = data.elapsed_ms != null ? data.elapsed_ms : '--';
    }

    const tagsContainer = document.getElementById('resSkillsTags');
    tagsContainer.innerHTML = '';
    p.skills.forEach(s => {
        const tag = document.createElement('span');
        tag.className = 'text-sm bg-indigo-950/90 border border-indigo-500/50 text-indigo-100 px-4 py-1.5 rounded-lg font-bold shadow-sm';
        tag.textContent = s;
        tagsContainer.appendChild(tag);
    });

    // Advisor: làm sạch và escape chuỗi trước khi in đậm markdown
    const adv = data.career_advice;
    const safeSummary = escapeHtml(adv.executive_summary || '');
    document.getElementById('advSummary').innerHTML = safeSummary.replace(/\*\*(.*?)\*\*/g, '<b class="text-cyan-300 font-extrabold">$1</b>');

    const roadmapContainer = document.getElementById('advRoadmap');
    roadmapContainer.innerHTML = '';
    adv.learning_roadmap.forEach(r => {
        const card = document.createElement('div');
        card.className = 'bg-slate-900/90 p-4 rounded-xl border border-slate-800 shadow-sm';
        card.innerHTML = `<span class="text-xs font-black text-indigo-300 bg-indigo-950/80 border border-indigo-500/40 px-2.5 py-1 rounded">${escapeHtml(r.week)}</span>
                          <p class="font-bold text-white mt-2 text-base">${escapeHtml(r.focus)}</p>
                          <p class="text-sm text-slate-200 mt-1 leading-relaxed">${escapeHtml(r.action)}</p>`;
        roadmapContainer.appendChild(card);
    });

    // Jobs
    const jobsContainer = document.getElementById('jobsList');
    jobsContainer.innerHTML = '';
    data.top_job_evaluations.forEach((job, idx) => {
        const jobCard = document.createElement('div');
        jobCard.className = 'bg-slate-900/80 p-5 rounded-2xl border border-slate-800 hover:border-slate-700 transition shadow-md';
        
        const matchedTags = job.matched_skills.map(s => {
            const isConcept = s.includes('(qua ') || s.includes('(thông qua ');
            const cls = isConcept ? 'bg-indigo-950/90 text-cyan-300 border-cyan-500/60 shadow-sm' : 'bg-emerald-950/90 text-emerald-300 border-emerald-500/50';
            const icon = isConcept ? 'fa-solid fa-brain text-cyan-400' : 'fa-solid fa-check';
            const badge = isConcept ? ' <span class="text-xs bg-cyan-500/20 text-cyan-200 px-2 py-0.5 rounded border border-cyan-500/30 ml-1">Phả hệ tri thức</span>' : '';
            return `<span class="text-sm ${cls} border px-3.5 py-1.5 rounded-md font-bold"><i class="${icon} text-sm"></i> ${escapeHtml(s)}${badge}</span>`;
        }).join(' ');
        
        const missingTags = job.missing_skills.map(s => `<span class="text-sm bg-rose-950/90 text-rose-300 border border-rose-500/50 px-3.5 py-1.5 rounded-md font-bold"><i class="fa-solid fa-circle-exclamation text-sm"></i> ${escapeHtml(s)}</span>`).join(' ');

        jobCard.innerHTML = `
            <div class="flex justify-between items-start gap-4">
                <div>
                    <span class="text-xs text-slate-400 font-mono font-bold tracking-wider">HẠNG #${idx + 1} • ${escapeHtml(job.job_id)}</span>
                    <h4 class="text-xl font-bold text-white hover:text-indigo-400 transition mt-0.5">${escapeHtml(job.job_title)}</h4>
                    <p class="text-sm text-slate-200 font-medium mt-0.5">${escapeHtml(job.company)} • <span class="text-slate-400">${escapeHtml(job.location)}</span></p>
                    <p class="text-sm text-emerald-400 font-bold mt-1.5"><i class="fa-solid fa-money-bill-wave"></i> ${escapeHtml(job.salary)} • Kinh nghiệm: ${escapeHtml(job.experience)}</p>
                </div>
                <div class="text-right flex-shrink-0 bg-slate-950/90 px-4 py-3 rounded-2xl border border-slate-800">
                    <span class="text-xs text-slate-400 uppercase font-extrabold tracking-wider">Độ Phù Hợp</span>
                    <div class="text-3xl font-black text-cyan-300 mt-0.5">${job.overall_fit_score}%</div>
                    <span class="text-xs text-slate-300 font-medium">(Vector: ${job.semantic_score}%)</span>
                </div>
            </div>
            <div class="mt-4 pt-4 border-t border-slate-800 space-y-2.5">
                <div>
                    <span class="text-xs text-emerald-400 font-extrabold uppercase tracking-wider">Kỹ năng đáp ứng (${job.matched_skills.length}):</span>
                    <div class="flex flex-wrap gap-2 mt-1.5">${matchedTags || '<span class="text-sm text-slate-400">Chưa ghi nhận</span>'}</div>
                </div>
                <div>
                    <span class="text-xs text-rose-400 font-extrabold uppercase tracking-wider">Kỹ năng cần bổ sung (${job.missing_skills.length}):</span>
                    <div class="flex flex-wrap gap-2 mt-1.5">${missingTags || '<span class="text-sm text-emerald-400 font-bold">Đáp ứng trọn vẹn 100%</span>'}</div>
                </div>
            </div>
        `;
        jobsContainer.appendChild(jobCard);
    });
}

// Auto-run on load
window.onload = function() {
    loadSampleCVs();
};
