@echo off
chcp 65001 > nul
title La Ban Ky Nang - He Thong Agent-RAG

echo ======================================================================
echo       HE THONG LA BAN KY NANG - AGENT-RAG PHAN TICH CV VA IT SKILL GAP
echo       Do an Chuyen de Cong nghe Thong tin - Truong DH Thuy Loi
echo       GVHD: Ts. Hoang Van Quy - SVTH: Nguyen Thi Tuyet Nhung
echo ======================================================================
echo.

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

if exist ".\venv\Scripts\python.exe" (
    set "PY_EXE=.\venv\Scripts\python.exe"
) else if exist ".\CareerLens-AI\ai-service\venv\Scripts\python.exe" (
    set "PY_EXE=.\CareerLens-AI\ai-service\venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo [1/2] Dang khoi dong Backend API va Frontend Dashboard tai http://127.0.0.1:8000 ...
echo       Trinh thong dich Python: %PY_EXE%
echo       Trinh duyet se tu dong mo len sau khi nap xong pipeline.
echo.
echo * Luu y: Giu nguyen cua so nay trong suot buoi van dap.
echo * Nhan Ctrl+C de tat ung dung khi bao cao xong.
echo.

"%PY_EXE%" "backend\app.py"

pause
