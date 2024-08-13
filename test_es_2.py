import json
import time
import pandas as pd
import torch
import numpy as np
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

start_time = time.time()

# Token given after approval
with open('./file.json') as f: #save the token inside a file
    config_data = json.load(f)
HF_TOKEN = config_data["HF_TOKEN"]

# 1. CSV curated
all_data = pd.read_csv('df_curated_2023.csv', encoding='latin-1')
print("Punto de control 1: Datos cargados")

# 2. Preprocessing
relevant_columns = ['Client Name', 'Project Name', 'TECH-Mgmt', 'Tech Score', 'RAIDD']
missing_columns = [col for col in relevant_columns if col not in all_data.columns]
if missing_columns:
    print(f"Error: Faltan las siguientes columnas relevantes: {', '.join(missing_columns)}")
else:
    all_data['combined_info'] = all_data[relevant_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# 3. Embeddings
final_dict = {}
chunk_size = 100
sentence_model = SentenceTransformer("all-mpnet-base-v2")
for i in np.arange(0, all_data.shape[0], chunk_size):
    sample = all_data.iloc[i:i + chunk_size]
    embeddings = sentence_model.encode(sample['combined_info'].tolist())
    for j, emb in enumerate(embeddings):
        final_dict[i + j] = {'embedding': emb, 'text': sample['combined_info'].iloc[j]}

with open('embeddings_dict.pkl', 'wb') as f:
    pickle.dump(final_dict, f)

print("Punto de control 2: Embeddings creados y guardados")

# 4. Query
def get_context(query, top_k=3):
    query_embedding = sentence_model.encode(query)
    distances = []
    for key in final_dict:
        distances.append(torch.cdist(torch.tensor(query_embedding).reshape(1, -1), 
                                     torch.tensor(final_dict[key]['embedding']).reshape(1, -1)).flatten().item())
    
    top_indices = torch.topk(torch.tensor(distances), k=top_k, largest=False).indices
    return [final_dict[idx.item()]['text'] for idx in top_indices]

query = "Qué riesgos (RISKS) identificas para Embol y Embonor?" # Modificar Query
context = get_context(query)
print("Punto de control 3: Contexto obtenido")

# Llama 2 model 
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=HF_TOKEN, 
    trust_remote_code=True,
    #low_cpu_mem_usage=True, 
    device_map='auto')
print("Punto de control 4: Modelo cargado... ahora, paciencia")

rag_prompt = f"""Contexto:\n{context}\n\nPregunta: {query}\nRespuesta (en español):"""

input_ids = tokenizer(rag_prompt, return_tensors="pt").to('cuda') 
with torch.no_grad():
    output = model.generate(**input_ids, max_new_tokens=256)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)

# 5. Time Tracking
end_time = time.time()
print(f"Tiempo total transcurrido: {end_time - start_time:.2f} segundos")
