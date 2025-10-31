# 🚀 Guida Completa: Llama-4-Scout-17B-16E-Instruct

## 📋 Indice
1. [Requisiti di Sistema](#requisiti)
2. [Installazione](#installazione)
3. [Configurazione Hugging Face](#configurazione)
4. [Utilizzo Base](#utilizzo-base)
5. [Funzionalità Avanzate](#avanzate)
6. [Risoluzione Problemi](#troubleshooting)

---

## 🖥️ Requisiti di Sistema {#requisiti}

### Hardware Minimo
- **RAM:** 32 GB (consigliato 64 GB)
- **GPU:** NVIDIA con 24 GB VRAM minimo
  - ✅ RTX 4090 (24 GB)
  - ✅ A100 (40/80 GB)
  - ✅ RTX 6000 Ada (48 GB)
  - ⚠️ RTX 3090 (24 GB) - limite stretto

### Software
- **Python:** 3.8 o superiore
- **CUDA:** 11.8 o 12.1+
- **PyTorch:** 2.0+
- **Transformers:** 4.50.0+

### Memoria Modello
- **Parametri totali:** 109B
- **Parametri attivi:** 17B
- **Formato BF16:** ~54 GB
- **Formato FP8:** ~27 GB
- **Formato 4-bit:** ~13.6 GB

---

## 📥 Installazione {#installazione}

### Passo 1: Clone/Download dei file
```bash
# Scarica i file che ho creato
# - llama4_scout_setup.py
# - install_llama4.sh
```

### Passo 2: Installa le dipendenze
```bash
# Rendi eseguibile lo script
chmod +x install_llama4.sh

# Esegui l'installazione
./install_llama4.sh
```

### Installazione Manuale (alternativa)
```bash
# Installa PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installa Transformers (versione recente)
pip install transformers>=4.50.0

# Installa dipendenze aggiuntive
pip install accelerate sentencepiece protobuf pillow

# Opzionale: per download più veloci
pip install 'transformers[hf_xet]'
```

---

## 🔑 Configurazione Hugging Face {#configurazione}

### Passo 1: Crea un account
1. Vai su https://huggingface.co
2. Crea un account gratuito

### Passo 2: Genera un token di accesso
1. Vai su https://huggingface.co/settings/tokens
2. Clicca "New token"
3. Nome: `llama4-access`
4. Tipo: **Read**
5. Copia il token generato

### Passo 3: Login dal terminale
```bash
# Installa CLI (se non installato)
pip install huggingface-hub

# Effettua il login
huggingface-cli login

# Incolla il token quando richiesto
```

### Passo 4: Accetta la licenza del modello
1. Vai su https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
2. Clicca su "Agree and access repository"
3. Leggi e accetta i termini della licenza Llama 4

⏱️ **Nota:** L'approvazione può richiedere da pochi minuti a qualche ora.

---

## 🎯 Utilizzo Base {#utilizzo-base}

### Test Rapido
```bash
python3 llama4_scout_setup.py
```

Scegli l'opzione **1** per il test più semplice.

### Esempio Base in Python
```python
from transformers import pipeline
import torch

# Carica il modello
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Conversazione
messages = [
    {"role": "user", "content": "Ciao! Raccontami una storia breve."}
]

# Genera risposta
output = pipe(messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

### Conversazione Multi-turno
```python
conversazione = [
    {"role": "user", "content": "Come si fa la pizza?"},
]

output = pipe(conversazione, max_new_tokens=300)
risposta = output[0]["generated_text"][-1]["content"]

# Continua la conversazione
conversazione.append({"role": "assistant", "content": risposta})
conversazione.append({"role": "user", "content": "E per l'impasto?"})

output = pipe(conversazione, max_new_tokens=200)
```

---

## 🚀 Funzionalità Avanzate {#avanzate}

### 1. Analisi di Immagini (Multimodale)
```python
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
model = Llama4ForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Analizza un'immagine
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "https://example.com/foto.jpg"},
        {"type": "text", "text": "Cosa vedi in questa immagine?"}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
risposta = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(risposta)
```

### 2. Controllo della Temperatura
```python
# Risposta più creativa
output = pipe(
    messages, 
    max_new_tokens=200,
    temperature=1.0,  # 0.0 = deterministica, 2.0 = molto creativa
    top_p=0.95
)

# Risposta più coerente
output = pipe(
    messages,
    max_new_tokens=200,
    temperature=0.3,
    top_p=0.9
)
```

### 3. Quantizzazione per Risparmiare Memoria
```python
from transformers import BitsAndBytesConfig

# Configurazione 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Carica modello quantizzato
model = Llama4ForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 4. Streaming delle Risposte
```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Genera in un thread separato
generation_kwargs = dict(
    inputs.to(model.device),
    streamer=streamer,
    max_new_tokens=200
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stampa man mano che arriva
for new_text in streamer:
    print(new_text, end="", flush=True)
```

---

## 🔧 Risoluzione Problemi {#troubleshooting}

### ❌ Errore: "OutOfMemoryError"
**Causa:** GPU non ha abbastanza memoria

**Soluzioni:**
```python
# 1. Usa quantizzazione 8-bit
model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,
    device_map="auto"
)

# 2. Usa quantizzazione 4-bit (ancora più leggera)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Riduci batch size
output = pipe(messages, max_new_tokens=100)  # Invece di 500
```

### ❌ Errore: "Repository not found"
**Causa:** Non hai accettato la licenza

**Soluzione:**
1. Vai su https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
2. Clicca "Agree and access repository"
3. Attendi l'approvazione (può richiedere qualche ora)

### ❌ Errore: "Invalid token"
**Causa:** Token Hugging Face non valido

**Soluzione:**
```bash
# Re-login con un nuovo token
huggingface-cli logout
huggingface-cli login
```

### ❌ Modello molto lento
**Causa:** Sta usando CPU invece di GPU

**Soluzione:**
```python
# Verifica CUDA
import torch
print(torch.cuda.is_available())  # Deve essere True

# Forza l'uso della GPU
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device=0,  # Specifica GPU 0
    torch_dtype=torch.bfloat16
)
```

### ⚠️ Warning: "Some weights were not initialized"
**Causa:** Normale per modelli grandi, non è un errore

**Soluzione:** Ignora, è normale e non influenza il funzionamento.

---

## 📊 Benchmark e Prestazioni

### Velocità (su A100 80GB)
- **First Token:** ~2-3 secondi
- **Generazione:** ~20-30 token/secondo
- **Risposta 200 token:** ~8-12 secondi

### Qualità
- **Lingue supportate:** 12 (incluso italiano!)
- **Context length:** fino a 128K token
- **Comprensione italiana:** Eccellente
- **Multimodalità:** Supporto nativo immagini

---

## 🎓 Risorse Aggiuntive

### Documentazione
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [Llama 4 Model Card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [Hugging Face Hub](https://huggingface.co/docs/hub)

### Community
- [Hugging Face Forum](https://discuss.huggingface.co)
- [Discord Hugging Face](https://hf.co/join/discord)

### Alternative
Se hai problemi con Llama 4 Scout:
- **Llama 3.3 70B:** Più stabile ma richiede più memoria
- **Llama 3 8B:** Più leggero, ottimo per GPU consumer
- **Mistral 7B:** Eccellente alternativa open-source

---

## ✅ Checklist Finale

Prima di iniziare, assicurati di aver:
- [ ] Installato Python 3.8+
- [ ] Installato CUDA e PyTorch con GPU
- [ ] Creato account Hugging Face
- [ ] Generato token di accesso
- [ ] Fatto login con `huggingface-cli login`
- [ ] Accettato la licenza del modello
- [ ] Verificato che `torch.cuda.is_available()` sia True
- [ ] Almeno 24 GB di VRAM disponibili

---

## 💡 Tips & Tricks

### Ottimizzazione Memoria
```python
# Libera cache GPU tra le generazioni
torch.cuda.empty_cache()

# Usa gradient checkpointing
model.gradient_checkpointing_enable()
```

### Migliori Prompt
```python
# ✅ Buono
"Spiega in modo dettagliato come funziona la fotosintesi."

# ❌ Troppo vago
"Fotosintesi?"

# ✅ Ottimo (con contesto)
"Sei un insegnante di biologia. Spiega la fotosintesi a uno studente di 15 anni usando analogie semplici."
```

### System Prompt
```python
messages = [
    {"role": "system", "content": "Sei un assistente esperto in cucina italiana."},
    {"role": "user", "content": "Come si fa la carbonara?"}
]
```

---

## 🎉 Buon Divertimento!

Ora hai tutto quello che ti serve per usare Llama 4 Scout!

Per domande o problemi, consulta la sezione [Risoluzione Problemi](#troubleshooting) o chiedi nella community di Hugging Face.

**Buona sperimentazione! 🚀**
