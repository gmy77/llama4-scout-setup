#!/usr/bin/env python3
"""
🚀 Test Veloce Llama 4 Scout
Versione semplificata per test rapido
"""

print("=" * 60)
print("🤖 LLAMA 4 SCOUT - Test Veloce")
print("=" * 60)

# 1. Import librerie
print("\n📚 Caricamento librerie...")
try:
    from transformers import pipeline
    import torch
    print("✅ Librerie caricate con successo!")
except ImportError as e:
    print(f"❌ Errore: {e}")
    print("\n💡 Esegui prima: pip install transformers torch accelerate")
    exit(1)

# 2. Verifica GPU
print("\n🔍 Controllo GPU...")
if torch.cuda.is_available():
    print(f"✅ GPU disponibile: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  GPU non disponibile, userò CPU (molto lento!)")

# 3. Carica il modello
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

print(f"\n⬇️  Caricamento modello...")
print(f"   {MODEL_ID}")
print("   (Questo può richiedere diversi minuti al primo avvio...)")

try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("✅ Modello caricato con successo!")
    
except Exception as e:
    print(f"\n❌ Errore nel caricamento del modello:")
    print(f"   {e}")
    print("\n💡 Possibili soluzioni:")
    print("   1. Assicurati di aver fatto login: huggingface-cli login")
    print("   2. Accetta la licenza su:")
    print("      https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct")
    print("   3. Verifica di avere abbastanza memoria GPU (24GB minimo)")
    exit(1)

# 4. Test 1: Domanda semplice
print("\n" + "=" * 60)
print("📝 TEST 1: Domanda Semplice in Italiano")
print("=" * 60)

messages = [
    {"role": "user", "content": "Ciao! Chi sei?"}
]

print("\n🤖 Generazione risposta...")
output = pipe(messages, max_new_tokens=150, do_sample=False)
risposta = output[0]["generated_text"][-1]["content"]

print(f"\n✅ RISPOSTA:")
print(f"{risposta}")

# 5. Test 2: Domanda più complessa
print("\n" + "=" * 60)
print("📝 TEST 2: Domanda Complessa")
print("=" * 60)

messages = [
    {"role": "user", "content": "Spiegami in 3 punti cos'è l'intelligenza artificiale."}
]

print("\n🤖 Generazione risposta...")
output = pipe(messages, max_new_tokens=300, temperature=0.7)
risposta = output[0]["generated_text"][-1]["content"]

print(f"\n✅ RISPOSTA:")
print(f"{risposta}")

# 6. Test 3: Conversazione multi-turno
print("\n" + "=" * 60)
print("📝 TEST 3: Conversazione Multi-turno")
print("=" * 60)

conversazione = [
    {"role": "user", "content": "Mi consigli una ricetta italiana facile?"}
]

print("\n👤 Tu: Mi consigli una ricetta italiana facile?")
output = pipe(conversazione, max_new_tokens=200)
risposta1 = output[0]["generated_text"][-1]["content"]
print(f"\n🤖 Llama: {risposta1}")

# Continua la conversazione
conversazione.append({"role": "assistant", "content": risposta1})
conversazione.append({"role": "user", "content": "Quanto tempo ci vuole?"})

print(f"\n👤 Tu: Quanto tempo ci vuole?")
output = pipe(conversazione, max_new_tokens=150)
risposta2 = output[0]["generated_text"][-1]["content"]
print(f"\n🤖 Llama: {risposta2}")

# Fine
print("\n" + "=" * 60)
print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("=" * 60)
print("\n💡 Prossimi passi:")
print("   - Esegui 'python3 llama4_scout_setup.py' per più opzioni")
print("   - Scegli opzione 4 per chat interattiva")
print("   - Leggi GUIDA_LLAMA4_SCOUT.md per funzionalità avanzate")
print()
