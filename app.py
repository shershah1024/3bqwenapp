from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
import zipfile
import requests
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

# --- Download and extract LoRA adapter ---
def download_and_extract(url, dest="lora_adapter"):
    zip_path = "/tmp/lora_adapter.zip"
    if not os.path.exists(dest):
        print("[INFO] Downloading LoRA adapter...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest)
        print("[INFO] LoRA adapter extracted.")

# Adapter URL
adapter_url = "https://mbjkvwatoiryvmskgewn.supabase.co/storage/v1/object/public/site_files//4.zip"
download_and_extract(adapter_url)

# --- Load base model + processor from Hugging Face ---
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --- Apply LoRA adapter ---
model = PeftModel.from_pretrained(base_model, "lora_adapter")
model.eval()

# --- Inference endpoint ---
@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs)
        prediction = processor.batch_decode(output, skip_special_tokens=True)[0]
        return JSONResponse(content={"output": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})