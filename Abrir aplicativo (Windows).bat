@echo off
REM Abre o aplicativo de valor de shopping centers no navegador
cd /d "%~dp0"
if not exist ".venv" (
  echo Primeira execucao: criando ambiente Python e instalando bibliotecas - alguns minutos...
  python -m venv .venv
  if errorlevel 1 (
    echo Python nao encontrado. Instale em https://www.python.org/downloads/ marcando Add python.exe to PATH
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -r requirements.txt
)
echo Abrindo o aplicativo em http://localhost:8501 - feche esta janela para encerrar
.venv\Scripts\python -m streamlit run app.py
pause
