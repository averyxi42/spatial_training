import torch
import time
import argparse
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- Constants ---
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_WIDTH = 640//5
IMAGE_HEIGHT = 480//5
# Approximate token count for 640x480. 
# Qwen2-VL uses patch_size=14x14. (640/14) * (480/14) ≈ 45 * 34 ≈ 1500 tokens.
EXPECTED_TOKENS_PER_IMG = 1600 

class BenchmarkContext:
    def __init__(self, model_id=MODEL_ID):
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        ).eval()
        self.device = self.model.device
        
        # Warmup the CUDA allocator
        torch.cuda.empty_cache()

    def get_vram_gb(self):
        return torch.cuda.memory_allocated() / 1024**3

class DataGenerator:
    """Generates synthetic turn data."""
    def __init__(self, width, height, processor):
        self.width = width
        self.height = height
        self.processor = processor

    def create_synthetic_image(self):
        arr = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def prepare_turn_inputs(self, step_idx):
        """
        Creates inputs for a SINGLE turn (Image + Text).
        We do not build the full conversation history in the prompt.
        We rely on the KV cache for history.
        """
        image = self.create_synthetic_image()
        
        # Construct a standalone prompt for this step
        # We simulate the user asking for a move
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Step {step_idx}: Next move?"}
                ]
            },
            # We add the Assistant start token to force the model to predict the response immediately
            {"role": "assistant", "content": "move forward or wtv"} 
        ]
        
        # Process ONLY this turn's data
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(
            text=[text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt"
        )
        return inputs

def run_step(ctx: BenchmarkContext, inputs, past_key_values):
    """
    Executes a single forward pass (Prefill of the new turn).
    Returns: Latency (ms), new_past_key_values
    """
    # Move inputs to GPU
    inputs = {k: v.to(ctx.device) for k, v in inputs.items()}
    
    # Qwen2-VL specific: Ensure we pass the past_key_values
    # Note: 'use_cache=True' is default, but we make it explicit.
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.inference_mode():
        outputs = ctx.model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True
        )
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    return (t1 - t0) * 1000, outputs.past_key_values

def run_benchmark(args):
    ctx = BenchmarkContext()
    generator = DataGenerator(IMAGE_WIDTH, IMAGE_HEIGHT, ctx.processor)
    
    past_key_values = None
    latencies = []
    
    print(f"\nStarting Benchmark: {args.turns} Turns")
    print(f"Image Size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"{'Step':<6} | {'Input Toks':<10} | {'Latency (ms)':<12} | {'VRAM (GB)':<10}")
    print("-" * 50)

    for step in range(args.turns):
        # 1. Prepare Inputs (CPU heavy, strictly separated from GPU measurement)
        inputs = generator.prepare_turn_inputs(step)
        
        # Calculate input size for logging
        # Qwen2-VL flattens images, so input_ids length includes image tokens
        seq_len = inputs.input_ids.shape[1]
        
        # 2. Run Forward Pass
        # We manually feed the output KV cache of step N-1 into step N
        latency, past_key_values = run_step(ctx, inputs, past_key_values)
        
        latencies.append(latency)
        vram = ctx.get_vram_gb()
        
        print(f"{step+1:<6} | {seq_len:<10} | {latency:<12.2f} | {vram:<10.2f}")

    print("-" * 50)
    print(f"Avg Latency: {np.mean(latencies):.2f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Total KV Cache accumulation not tracked (handled implicitly by HF backend)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=10, help="Number of steps to simulate")
    args = parser.parse_args()
    
    run_benchmark(args)