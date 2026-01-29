from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import json
import time

app = FastAPI()

# --- Load Model (Optimized for CPU) ---
# n_threads=2 matches the HF Free Tier (2 vCPU)
print("⏳ Loading Model...")
llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,          # Context window
    n_threads=2,         # STRICTLY 2 for free tier
    n_batch=512,         # Process prompts in chunks
    verbose=False
)
print("✅ Model Loaded!")

class ArticleRequest(BaseModel):
    id: str
    content: str

@app.get("/")
def home():
    return {"status": "active", "model": "Gemma-3-1B-News-Analyzer"}

@app.post("/analyze")
def analyze_article(request: ArticleRequest):
    start_time = time.time()
    
    # 1. Format Prompt (Strictly matching your training)
    prompt = f"<ARTICLE>\n{request.content}\n</ARTICLE>"
    
    # 2. Run Inference
    # max_tokens=200 prevents it from rambling if it gets confused
    output = llm(
        prompt,
        max_tokens=256, 
        stop=["<eos>", "<end_of_turn>"], 
        echo=False
    )
    
    raw_text = output['choices'][0]['text'].strip()
    
    # 3. Safe Parse
    try:
        # Sometimes models add extra text; find the JSON object
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = raw_text[json_start:json_end]
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON found")
            
        return {
            "id": request.id,
            "analysis": parsed,
            "time_taken": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        print(f"❌ Parse Error: {raw_text}")
        return {
            "id": request.id,
            "error": "Parsing Failed", 
            "raw_output": raw_text
        }