@echo off
chcp 65001 > nul
title KET QUA THUC NGHIEM DOI SOAT KHOA HOC - LA BAN KY NANG

echo ======================================================================
echo    THUC NGHIEM DOI SOAT KHOA HOC - HE THONG LA BAN KY NANG (AGENT-RAG)
echo    Do an Chuyen de CNTT - SV: Nguyen Thi Tuyet Nhung (K65-CNTT)
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

echo Dang chay kich ban thuc nghiem doi soat khoa hoc bang evaluate.py...
echo Trinh thong dich Python: %PY_EXE%
echo.

"%PY_EXE%" "evaluate.py"

echo.
echo ======================================================================
echo Thuc nghiem hoan tat. Nhan phim bat ky de thoat.
echo ======================================================================
pause
