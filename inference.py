import numpy as np
import random
from diffusers import DiffusionPipeline
import torch
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_KEY")

if openai_key == "<YOUR_OPENAI_KEY>":
    openai_key = ""

if openai_key == "":
    sys.exit("Please Provide Your OpenAI API Key")

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
else: 
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
    pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max

def infer_stable_diffusion(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_steps, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    return image

def infer_dall_e(text, model, quality, size):
    try:
        client = OpenAI(api_key=openai_key)

        response = client.images.generate(
            prompt=text,
            model=model,
            quality=quality,
            size=size,
            n=1,
        )
    except Exception as error:
        print(str(error))
        raise gr.Error("An error occurred while generating image.")

    return response.data[0].url