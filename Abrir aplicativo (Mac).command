#!/bin/bash
# Abre o aplicativo de valor de shopping centers no navegador
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Primeira execucao: criando ambiente Python e instalando bibliotecas (alguns minutos)..."
  python3 -m venv .venv || { echo "Python 3 nao encontrado. Instale em https://www.python.org/downloads/"; read -p "Tecle Enter para sair"; exit 1; }
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
fi
echo "Abrindo o aplicativo em http://localhost:8501 ... (feche esta janela para encerrar)"
.venv/bin/python -m streamlit run app.py
