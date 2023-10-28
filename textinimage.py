pip install git+https://github.com/huggingface/diffusers
pip install transformers accelerate safetensors
from diffusers import StableDiffusionXLPipeline
import torch
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
prompt = "An astronaut riding a green horse" 
neg_prompt = "ugly, blurry, poor quality" 
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
