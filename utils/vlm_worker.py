# import os
import numpy as np
# from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half

class VLMWorker:

    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct",attn_implementation='sdpa',dtype='float16', prefix = '<|im_start|>assistant\n**',postfix = '**<|im_end|>',vocab=["stop","forward","left","right","up","down"],cache_outputs=False,load_model=True,offload_cache=False):
        import transformers.modeling_flash_attention_utils as fa_utils
        def patched(position_ids, batch_size):
            return False
        fa_utils._is_packed_sequence = patched
        import torch
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vocab = vocab
        self.vocab_ids = self._vocab_to_ids(vocab)
        self.cache_outputs = cache_outputs
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        self.model=None
        self.prefix_ids = self.processor.tokenizer.encode(prefix)
        self.postfix_ids = self.processor.tokenizer.encode(postfix)
        self.offload_cache = offload_cache
        # Warmup the CUDA allocator
        if load_model:
            self.load_model()
        torch.cuda.empty_cache()
        self.reset()
        
    def reset(self):
        from transformers import DynamicCache
        import torch
        self.past_key_values=DynamicCache(config=self.model.config, offloading=self.offload_cache)
        self.cumulative_attention_mask = None
        self.offset = 0
        self.output_list = []
        torch.cuda.empty_cache()

    def load_model(self):
        from transformers import AutoModelForImageTextToText
        print(f"Loading {self.model_id}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,#"sdpa",
            device_map="cuda",
        ).eval()
        self.device = self.model.device

    def tokenize_inputs(self,messages,images):
                # Process ONLY this turn's data
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        inputs = self.processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt"
        )
        return inputs
    
    def _get_sandwich_indices(self, input_ids,find_last=True):
        import torch
        """
        Locates the indices of the logits that predict the sandwiched tokens.
        
        Returns:
            logit_indices (torch.Tensor): Indices relative to 'input_ids' to pass to logits_to_keep.
            target_ids (torch.Tensor): The ground truth tokens to calculate logprobs for.
        """
        # 1. Convert to NumPy for fast, robust search
        seq = input_ids[0].cpu().numpy()
        
        # Helper: NumPy sliding window search
        def search_sequence_numpy(arr, sub, find_last=True):
            window_size = len(sub)
            if len(arr) < window_size:
                return -1
            # Create strided view for O(1) comparison
            shape = (arr.shape[0] - window_size + 1, window_size)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            
            # Find all matches
            matches = np.all(windows == sub, axis=1)
            indices = np.where(matches)[0]
            
            if indices.size > 0:
                return indices[-1] if find_last else indices[0]
            return -1

        # 2. Find Prefix End
        prefix_np = np.array(self.prefix_ids)
        prefix_start = search_sequence_numpy(seq, prefix_np,find_last=find_last)
        if prefix_start == -1:
            return None, None
        prefix_end = prefix_start + len(prefix_np)

        # 3. Find Postfix Start (Search after prefix)
        postfix_np = np.array(self.postfix_ids)
        seq_suffix = seq[prefix_end:] 
        postfix_relative_start = search_sequence_numpy(seq_suffix, postfix_np,find_last=False)
        if postfix_relative_start == -1:
            return None, None
        postfix_start = prefix_end + postfix_relative_start

        # 4. Calculate Indices
        # Target tokens are at: input_ids[prefix_end : postfix_start]
        # The hidden state at index 'i' predicts the token at 'i+1'.
        # So we need hidden states at: [prefix_end - 1 : postfix_start - 1]
        
        logit_start = prefix_end - 1
        logit_end = postfix_start - 1

        # Create the indices tensor to pass to the model
        logit_indices = torch.arange(logit_start, logit_end, device='cpu', dtype=torch.long)
        mask_indices = torch.arange(prefix_start,postfix_start+len(postfix_np), device='cpu', dtype=torch.long) # mask for the entire sequence with postfix and prefix
        return logit_indices, mask_indices

    def _vocab_to_ids(self,vocab):
        ids = []
        for word in vocab:
            ids +=self.processor.tokenizer.encode(word)
        if len(ids)!=len(vocab):
            raise("input vocabulary is not valid token list!")
        return ids
    
    def infer_step(self,messages,images,full_logprobs=False,temperature=1.2,check_probs=True,crop_after_prefix=True):
        import torch
        if self.model is None:
            self.load_model()
            self.reset()
        inputs = self.tokenize_inputs(messages,images)
        logit_indices,mask_indices = self._get_sandwich_indices(inputs['input_ids'])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        image_grid_thw = inputs['image_grid_thw']
        attention_mask = inputs['attention_mask']
        # 1. Ask Qwen to calculate the 3D layout for this chunk
        # This returns positions starting at T=0, H=0, W=0 relative to this chunk
        position_ids, deltas = self.model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw, 
            video_grid_thw=None,
            attention_mask=attention_mask
        )
        inputs['position_ids'] = position_ids+self.offset
        chunk_attention_mask = inputs['attention_mask']

        # Update the cumulative mask
        if self.cumulative_attention_mask is None:
            self.cumulative_attention_mask = chunk_attention_mask
        else:
            # Concatenate past mask with current mask
            self.cumulative_attention_mask = torch.cat(
                [self.cumulative_attention_mask, chunk_attention_mask], 
                dim=1
            )

        inputs['attention_mask'] = self.cumulative_attention_mask # torch.ones(1,seql,device=self.device)
        previous_history_len = self.past_key_values.get_seq_length() if self.past_key_values.get_seq_length() is not None else 0
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                past_key_values=self.past_key_values,
                use_cache=True,
                logits_to_keep = logit_indices.to(self.model.device)
            )
            self.past_key_values = outputs.past_key_values
            if self.cache_outputs:
                self.output_list.append(outputs)
                # Compute logprobs directly (1-to-1 mapping)
            relevant_logits = outputs.logits[0].float()
            all_logprobs = torch.log_softmax(relevant_logits/temperature, dim=-1)

        if crop_after_prefix:
            absolute_crop_index = previous_history_len + mask_indices[0].item()
            # 2. Crop KV Cache (In-place)
            self.past_key_values.crop(absolute_crop_index)
            # 3. Crop the Cumulative Attention Mask
            # We must discard the corresponding columns in the mask
            if self.cumulative_attention_mask is not None:
                self.cumulative_attention_mask = self.cumulative_attention_mask[:, :absolute_crop_index]
            # 4. Reset Offset
            # The next generation should start exactly where we cropped
            self.offset += mask_indices[0].item()
        else:
            # Normal behavior: advance offset by the full input length
            self.offset += len(inputs['input_ids'][0])
        self.offset+=deltas.item() # this is ok because assistant generations don't contain images so no effect on deltas

        if check_probs:
            try:
                assert(torch.argmax(all_logprobs,dim=-1).item() in self.vocab_ids)
            except:
                print("WARNING: prediction not in provided vocab")
        # print("inference done!")
        if full_logprobs:
            return all_logprobs.cpu().float().numpy()
        else:
            return all_logprobs[:,self.vocab_ids].cpu().float().numpy()

    def infer_probs(self,messages,images,temperature=1.2,check_probs=False):
        logprobs = self.infer_step(messages,images,temperature=temperature, check_probs=check_probs)
        assert(len(logprobs)==1) #ensure there is a unique token position for decision making
        logprobs = logprobs[0]
        probs = np.exp(logprobs)
        probs /= np.sum(probs)
        return probs


if __name__ == "__main__":
    from vlm_worker import VLMWorker
    import torch
    import time
    import argparse
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print("running inference test")
    # --- Constants ---
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

    class DataGenerator:
        """Generates synthetic turn data."""
        def __init__(self, width, height, processor):
            self.width = width
            self.height = height
            self.processor = processor

        def create_synthetic_image(self):
            arr = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
            return Image.fromarray(arr)

        def _prepare_turn_inputs(self, step_idx):
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
                {"role": "assistant", "content": "**forward**"} 
            ]
            return messages, [image]

    worker = VLMWorker()
    generator = DataGenerator(IMAGE_WIDTH, IMAGE_HEIGHT, worker.processor)
    worker.reset()
    from tqdm import tqdm
    for i in tqdm(range(100)):
        messages,images = generator._prepare_turn_inputs(i)
        action = worker.infer_probs(messages,images)