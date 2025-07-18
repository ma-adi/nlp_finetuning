# api_server.py
import os
from llama_cpp import Llama
from fastapi import FastAPI
from pydantic import BaseModel

# Define the path to your GGUF model file
MODEL_PATH = "/Users/maadi5/nlp_finetuning/llama_ft_merged_no_hint/Llama_Ft_Merged_No_Hint-3.2B-F16.gguf"

# Initialize FastAPI app
app = FastAPI()

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  
        n_gpu_layers=0, 
        verbose=True,
    )
    print(f"Successfully loaded GGUF model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None # Set llm to None so routes can handle it

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = []

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    if llm is None:
        return {"error": "Model not loaded. Check server logs for details."}, 500

    try:
        output = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            echo=False, # Set to True to echo the prompt in the output
        )
        return {"generated_text": output["choices"][0]["text"]}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)