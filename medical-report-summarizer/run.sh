#!/bin/sh
# One-step setup and launch: ./run.sh   then open http://127.0.0.1:5000
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python app.py
