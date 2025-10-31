![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-Llama%204-green.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

# 🚀 Llama 4 Scout - Setup Completo

## 📦 File Inclusi

```
📁 Progetto Llama 4 Scout
├── 📄 README.md                    ← Sei qui!
├── 📄 GUIDA_LLAMA4_SCOUT.md       ← Guida completa dettagliata
├── 🔧 install_llama4.sh           ← Script installazione dipendenze
├── 🐍 llama4_scout_setup.py       ← Script completo con 4 modalità
└── 🐍 test_veloce.py              ← Test rapido per verificare funzionamento
```

---

## ⚡ Quick Start (3 passi)

### 1️⃣ Installa le dipendenze
```bash
chmod +x install_llama4.sh
./install_llama4.sh
```

### 2️⃣ Configura Hugging Face
```bash
# Login (ti verrà chiesto il token)
huggingface-cli login

# Poi vai su questa pagina e accetta la licenza:
# https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
```

### 3️⃣ Esegui il test
```bash
python3 test_veloce.py
```

✅ Se tutto funziona, sei pronto!

---

## 📚 Modalità d'Uso

### 🎯 Modalità 1: Test Veloce (Consigliato per iniziare)
```bash
python3 test_veloce.py
```
- Carica il modello automaticamente
- Esegue 3 test di esempio
- Perfetto per verificare che tutto funzioni

### 🎯 Modalità 2: Script Completo
```bash
python3 llama4_scout_setup.py
```
Poi scegli:
- **Opzione 1:** Pipeline semplice (solo testo)
- **Opzione 2:** AutoModel avanzato (più controllo)
- **Opzione 3:** Multimodale (con immagini!)
- **Opzione 4:** Chat interattivo

### 🎯 Modalità 3: Chat Interattivo
```bash
python3 llama4_scout_setup.py
# Scegli opzione 4
```
Chatta con il modello in tempo reale!

---

## 🖥️ Requisiti Minimi

### Hardware
- **RAM:** 32 GB (64 GB consigliato)
- **GPU:** NVIDIA con 24 GB VRAM
  - ✅ RTX 4090
  - ✅ A100 (40GB/80GB)
  - ✅ RTX 6000 Ada
  - ⚠️ RTX 3090 (al limite)

### Software
- **Python:** 3.8+
- **CUDA:** 11.8 o 12.1+
- **Account Hugging Face** (gratuito)

---

## 🔑 Setup Hugging Face (Dettagliato)

### Passo 1: Crea account
1. Vai su https://huggingface.co
2. Registrati gratuitamente

### Passo 2: Genera token
1. Vai su https://huggingface.co/settings/tokens
2. Clicca "New token"
3. Nome: `llama4-access`
4. Tipo: **Read**
5. Copia il token

### Passo 3: Login
```bash
pip install huggingface-hub
huggingface-cli login
# Incolla il token quando richiesto
```

### Passo 4: Accetta licenza
1. Vai su https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
2. Clicca "Agree and access repository"
3. Attendi approvazione (da 5 minuti a 2 ore)

---

## 🎓 Esempi di Codice

### Esempio Base
```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

messages = [{"role": "user", "content": "Ciao!"}]
output = pipe(messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

### Con Immagine
```python
from transformers import AutoProcessor, Llama4ForConditionalGeneration

processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
model = Llama4ForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "https://example.com/foto.jpg"},
        {"type": "text", "text": "Cosa vedi?"}
    ]
}]

inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
```

---

## 🔧 Risoluzione Problemi Comuni

### ❌ "OutOfMemoryError"
**Problema:** GPU non ha abbastanza memoria

**Soluzione 1:** Usa quantizzazione 8-bit
```python
model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map="auto"
)
```

**Soluzione 2:** Usa quantizzazione 4-bit (ancora più leggera)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)

model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### ❌ "Repository not found" o "401 Unauthorized"
**Problema:** Non hai accesso al modello

**Soluzione:**
1. Verifica di aver accettato la licenza su Hugging Face
2. Riprova il login: `huggingface-cli login`
3. Attendi l'approvazione (può richiedere tempo)

### ❌ Modello molto lento
**Problema:** Sta usando CPU

**Soluzione:**
```python
# Verifica GPU
import torch
print(torch.cuda.is_available())  # Deve essere True

# Forza GPU
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device=0,  # Usa GPU 0
    torch_dtype=torch.bfloat16
)
```

### ❌ "Some weights not initialized"
**Non è un errore!** È normale per modelli grandi. Ignora.

---

## 📖 Documentazione Completa

Per la guida dettagliata con tutte le funzionalità avanzate:
```bash
cat GUIDA_LLAMA4_SCOUT.md
# oppure aprila con un editor di testo
```

La guida include:
- Quantizzazione avanzata
- Streaming delle risposte
- Ottimizzazione memoria
- Tuning dei parametri
- Esempi multimodali completi
- Best practices per prompt
- Benchmark e performance

---

## 🎯 Caratteristiche Llama 4 Scout

### ✨ Punti di Forza
- **Multimodale:** Testo + Immagini nativo
- **12 Lingue:** Include italiano perfettamente
- **17B parametri attivi** su 109B totali (MoE)
- **Context:** Fino a 128K token
- **Efficiente:** Gira su GPU singola
- **Veloce:** ~20-30 token/secondo

### 📊 Performance
- **First token:** 2-3 secondi
- **Throughput:** 20-30 token/s
- **Qualità:** Eccellente in italiano
- **Comprensione immagini:** Stato dell'arte

---

## 🆚 Alternative

Se hai problemi con Llama 4 Scout, considera:

| Modello | Parametri | VRAM | Velocità | Italiano |
|---------|-----------|------|----------|----------|
| **Llama 4 Scout** | 17B (109B) | 24GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| Llama 3.3 70B | 70B | 48GB+ | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| Llama 3 8B | 8B | 16GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| Mistral 7B | 7B | 14GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |

---

## 💡 Tips Utili

### Per Risparmiare Memoria
```python
# Libera cache tra generazioni
torch.cuda.empty_cache()
```

### Per Risposte Migliori
```python
# Più deterministico
output = pipe(messages, temperature=0.3)

# Più creativo
output = pipe(messages, temperature=0.9)

# Bilanciato
output = pipe(messages, temperature=0.7, top_p=0.9)
```

### Per Conversazioni Lunghe
```python
# Usa system prompt
messages = [
    {"role": "system", "content": "Sei un assistente esperto."},
    {"role": "user", "content": "Domanda..."}
]
```

---

## 🤝 Supporto

### Problemi con questo setup?
1. Controlla [Risoluzione Problemi](#-risoluzione-problemi-comuni)
2. Leggi `GUIDA_LLAMA4_SCOUT.md`
3. Verifica i log di errore

### Problemi con il modello?
- [Hugging Face Forum](https://discuss.huggingface.co)
- [Discord Hugging Face](https://hf.co/join/discord)
- [Model Card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

---

## 📈 Roadmap

Prossimi miglioramenti a questi script:
- [ ] Supporto per quantizzazione GPTQ/AWQ
- [ ] Esempio con RAG (Retrieval Augmented Generation)
- [ ] Fine-tuning guidato
- [ ] Deployment con FastAPI
- [ ] Integrazione con LangChain

---

## ⭐ Feedback

Questo setup ti è stato utile? Hai suggerimenti?
Fammi sapere come posso migliorarlo!

---

## 📄 Licenza

Questi script sono forniti "as-is" per uso personale.

Il modello Llama 4 Scout è soggetto alla licenza Llama 4 Community License Agreement di Meta.
Leggi i termini su: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

---

**🎉 Buon divertimento con Llama 4 Scout!**
