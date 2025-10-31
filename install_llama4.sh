#!/bin/bash

echo "======================================"
echo "🚀 INSTALLAZIONE LLAMA 4 SCOUT"
echo "======================================"
echo ""

# Controlla Python
echo "📦 Controllo versione Python..."
python3 --version

# Aggiorna pip
echo ""
echo "⬆️  Aggiornamento pip..."
pip install --upgrade pip

# Installa dipendenze base
echo ""
echo "📚 Installazione librerie necessarie..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installa Transformers (versione più recente)
echo ""
echo "🤗 Installazione Transformers..."
pip install transformers>=4.50.0

# Installa accelerate per gestione GPU
echo ""
echo "⚡ Installazione Accelerate..."
pip install accelerate

# Installa altre dipendenze utili
echo ""
echo "🛠️  Installazione dipendenze aggiuntive..."
pip install sentencepiece protobuf pillow requests

# Installa hf_xet per download più veloci (opzionale)
echo ""
echo "🚄 Installazione hf_xet per download veloci..."
pip install 'transformers[hf_xet]'

echo ""
echo "======================================"
echo "✅ INSTALLAZIONE COMPLETATA!"
echo "======================================"
echo ""
echo "📝 Prossimi passi:"
echo "1. Effettua il login su Hugging Face:"
echo "   huggingface-cli login"
echo ""
echo "2. Vai su https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct"
echo "   e accetta la licenza del modello"
echo ""
echo "3. Esegui lo script di test:"
echo "   python3 llama4_scout_setup.py"
echo ""
