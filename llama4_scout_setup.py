#!/usr/bin/env python3
"""
🚀 Setup e Test per Llama-4-Scout-17B-16E-Instruct
Configurazione completa per conversazioni naturali in italiano
"""

import torch
from transformers import pipeline, AutoTokenizer, Llama4ForConditionalGeneration, AutoProcessor
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAZIONE
# ============================================
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

print("=" * 60)
print("🚀 LLAMA 4 SCOUT - Setup e Test")
print("=" * 60)
print()

# ============================================
# METODO 1: Pipeline (PIÙ SEMPLICE) - Solo Testo
# ============================================
def metodo_pipeline_semplice():
    """
    Metodo più semplice usando pipeline di Hugging Face
    Ottimo per iniziare e testare rapidamente
    """
    print("📝 METODO 1: Pipeline Semplice (Solo Testo)")
    print("-" * 60)
    
    try:
        # Carica il modello con pipeline
        pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            device_map="auto",  # Usa GPU automaticamente se disponibile
            torch_dtype=torch.bfloat16  # Usa precisione ridotta per risparmiare memoria
        )
        
        # Messaggio di test in italiano
        messages = [
            {"role": "user", "content": "Ciao! Chi sei e cosa puoi fare per me?"},
        ]
        
        print("🤖 Generazione risposta...")
        output = pipe(
            messages, 
            do_sample=False,  # Risposta deterministica
            max_new_tokens=200  # Lunghezza massima risposta
        )
        
        # Estrai e mostra la risposta
        risposta = output[0]["generated_text"][-1]["content"]
        print(f"\n✅ Risposta del modello:\n{risposta}\n")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("\nℹ️  Possibili soluzioni:")
        print("   1. Assicurati di aver accettato la licenza su Hugging Face")
        print("   2. Login con: huggingface-cli login")
        print("   3. Verifica di avere abbastanza memoria GPU/RAM")
        return None


# ============================================
# METODO 2: AutoModel (PIÙ CONTROLLO) - Solo Testo
# ============================================
def metodo_automodel_avanzato():
    """
    Metodo più avanzato con maggior controllo
    Utile per personalizzazioni avanzate
    """
    print("\n📝 METODO 2: AutoModel Avanzato (Solo Testo)")
    print("-" * 60)
    
    try:
        # Carica tokenizer e modello separatamente
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = Llama4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Messaggio di test
        messages = [
            {"role": "user", "content": "Spiegami in modo semplice cosa significa intelligenza artificiale."},
        ]
        
        # Prepara gli input
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            return_dict=True
        )
        
        print("🤖 Generazione risposta...")
        # Genera la risposta
        outputs = model.generate(
            **inputs.to(model.device), 
            max_new_tokens=200,
            temperature=0.7,  # Creatività della risposta
            top_p=0.9  # Nucleus sampling
        )
        
        # Decodifica solo la parte generata
        risposta = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        print(f"\n✅ Risposta del modello:\n{risposta[0]}\n")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None, None


# ============================================
# METODO 3: Multimodale (TESTO + IMMAGINI)
# ============================================
def metodo_multimodale():
    """
    Metodo per usare il modello con immagini
    Llama 4 Scout supporta nativamente le immagini!
    """
    print("\n📝 METODO 3: Multimodale (Testo + Immagini)")
    print("-" * 60)
    
    try:
        # Carica processor e modello
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Llama4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # URL di un'immagine di esempio
        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        
        # Messaggio con immagine
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": "Descrivi questa immagine in italiano in due frasi."},
                ]
            },
        ]
        
        print("🤖 Analisi immagine e generazione risposta...")
        # Prepara gli input
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Genera la risposta
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )
        
        # Decodifica la risposta
        risposta = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:]
        )[0]
        print(f"\n✅ Risposta del modello:\n{risposta}\n")
        
        return processor, model
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None, None


# ============================================
# FUNZIONE INTERATTIVA
# ============================================
def chat_interattivo(pipe):
    """
    Modalità chat interattiva
    """
    print("\n💬 MODALITÀ CHAT INTERATTIVA")
    print("-" * 60)
    print("Digita 'exit' o 'quit' per uscire\n")
    
    conversazione = []
    
    while True:
        # Input utente
        user_input = input("Tu: ")
        
        if user_input.lower() in ['exit', 'quit', 'esci']:
            print("\n👋 Arrivederci!")
            break
        
        # Aggiungi messaggio alla conversazione
        conversazione.append({"role": "user", "content": user_input})
        
        # Genera risposta
        output = pipe(
            conversazione, 
            do_sample=True,
            temperature=0.7,
            max_new_tokens=300
        )
        
        # Estrai risposta
        risposta = output[0]["generated_text"][-1]["content"]
        
        # Aggiungi risposta alla conversazione
        conversazione.append({"role": "assistant", "content": risposta})
        
        print(f"\n🤖 Llama 4: {risposta}\n")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n🔍 Controllo sistema...")
    print(f"PyTorch versione: {torch.__version__}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Scegli quale metodo usare
    print("Quale metodo vuoi testare?")
    print("1. Pipeline Semplice (consigliato per iniziare)")
    print("2. AutoModel Avanzato")
    print("3. Multimodale (con immagini)")
    print("4. Chat Interattivo")
    
    scelta = input("\nScegli (1-4): ").strip()
    
    if scelta == "1":
        pipe = metodo_pipeline_semplice()
        
    elif scelta == "2":
        tokenizer, model = metodo_automodel_avanzato()
        
    elif scelta == "3":
        processor, model = metodo_multimodale()
        
    elif scelta == "4":
        print("\n⚙️  Caricamento modello per chat interattivo...")
        pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        chat_interattivo(pipe)
    
    else:
        print("❌ Scelta non valida")
    
    print("\n" + "=" * 60)
    print("✅ Test completato!")
    print("=" * 60)
