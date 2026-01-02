import torch 
from transformers import Cache, Qwen3VLForConditionalGeneration, Qwen3VLTextModel,Qwen3VLModel, Qwen3VLVisionModel
from transformers.utils.generic import check_model_inputs
from typing import Any, Callable, Optional, TypedDict, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
import torch.nn as nn
from transformers import AutoConfig

def load_sparse_model(model_path, **kwargs):
    # 1. Load the Config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Instantiate our Custom Sparse Class
    # This creates the model with random weights but the correct sparse architecture
    # model = Qwen3VLSparseForConditionalGeneration(config)
    
    # 3. Load Pretrained Weights
    # We use from_pretrained on our class, pointing to the original model directory.
    # Because our attribute names (self.model, self.text_model) match the original,
    # the weights map 1:1.
    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        model_path, 
        config=config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation = "flash_attention_2",
        **kwargs
    )
    
    return model


def extract_contiguous_segments(tensor, mask):
    """
    Args:
        tensor: Shape (batch, seq, hidden)
        mask: Boolean tensor of shape (batch, seq)
    
    Returns:
        List[List[Tensor]]: A nested list where results[b] contains 
                            tensors of contiguous true regions for batch b.
    """
    results = []
    batch_size = tensor.shape[0]

    for b in range(batch_size):
        # Slice the current batch
        curr_tensor = tensor[b]
        curr_mask = mask[b]
        
        # 1. Pad the mask with False (0) on both sides.
        #    This ensures we catch segments starting at index 0 or ending at the last index.
        #    Example: [T, T, F] -> [0, 1, 1, 0, 0]
        padded_mask = torch.cat([
            torch.tensor([0], device=mask.device, dtype=torch.int), 
            curr_mask.int(), 
            torch.tensor([0], device=mask.device, dtype=torch.int)
        ])
        
        # 2. compute diff: 
        #    +1 indicates a transition from False to True (Start)
        #    -1 indicates a transition from True to False (End)
        diff = padded_mask[1:] - padded_mask[:-1]
        
        # 3. Get indices where transitions occur
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        
        # 4. Slice the tensor and collect segments
        batch_segments = []
        for s, e in zip(starts, ends):
            batch_segments.append(curr_tensor[s:e])
            
        results.append(batch_segments)

    return results

def filter_embeds(image_embeds, limit=None, threshold=0.8):
    import torch
    import torch.nn.functional as F

    device = image_embeds[0].device
    # Normalize and Flatten
    image_embeds = [F.normalize(emb, p=2, dim=1) for emb in image_embeds]
    image_embeds_flat = torch.cat(image_embeds, dim=0)
    
    total_patches = image_embeds_flat.shape[0]
    
    # Pre-allocate mask (1 byte per patch, negligible memory)
    keep_mask = torch.zeros(total_patches, dtype=torch.bool, device=device)
    
    # Initialize cursor and set first image as kept
    cursor = image_embeds[0].shape[0]
    keep_mask[:cursor] = True

    for embeds in image_embeds[1:]:
        # Select currently kept embeddings using the mask slice
        # Note: This limits the gather operation to only valid history up to 'cursor'
        db = image_embeds_flat[:cursor][keep_mask[:cursor]]

        # Similarity check
        sim = (db @ embeds.T).amax(dim=0)
        
        # Update mask directly (No concatenation overhead)
        local_unique = sim < threshold
        keep_mask[cursor : cursor + embeds.shape[0]] = local_unique
        
        cursor += embeds.shape[0]

    # Convert mask to indices
    embeds_to_keep_idx = torch.nonzero(keep_mask).squeeze()

    # Handle Limit
    if limit is not None and embeds_to_keep_idx.numel() > limit:
        subset = torch.randperm(embeds_to_keep_idx.numel(), device=device)[:limit]
        embeds_to_keep_idx = embeds_to_keep_idx[subset]
        embeds_to_keep_idx, _ = embeds_to_keep_idx.sort()

    return embeds_to_keep_idx

# def filter_embeds(image_embeds, limit=None, threshold=0.8):
#     import torch
#     import torch.nn.functional as F

#     # 1. Normalize upfront (In-place if possible to save memory, though F.normalize creates copy)
#     # We process them one by one to avoid creating one giant 'image_embeds_flat' initially if possible,
#     # but since we need to return global indices later, we need to track offsets.
    
#     device = image_embeds[0].device
    
#     # Store kept segments to avoid re-allocating the 'db' tensor every loop
#     # kept_history_chunks[i] = tensor of shape (N_kept_in_image_i, Dim)
#     kept_history_chunks = []
    
#     # We still need to track which *original* indices we kept. 
#     # Let's just store the integer indices directly.
#     kept_global_indices = []
    
#     current_offset = 0
    
#     # Handle first image (Always keep all)
#     first_img = F.normalize(image_embeds[0], p=2, dim=1)
#     kept_history_chunks.append(first_img)
    
#     # Create indices for first image [0, 1, ... N-1]
#     n_first = first_img.shape[0]
#     kept_global_indices.append(torch.arange(n_first, device=device))
#     current_offset += n_first

#     # Iterate through subsequent images
#     for i in range(1, len(image_embeds)):
#         # Current candidate image
#         curr_img_raw = image_embeds[i]
#         curr_img = F.normalize(curr_img_raw, p=2, dim=1)
#         n_curr = curr_img.shape[0]
        
#         # We need to find max similarity of each token in curr_img 
#         # against ALL tokens in kept_history_chunks.
        
#         # Initialize max_sim with -1 (or very small)
#         # Shape: (N_curr,)
#         global_max_sim = torch.full((n_curr,), -1.0, device=device, dtype=curr_img.dtype)
        
#         # Compare against history chunks (Streaming)
#         for chunk in kept_history_chunks:
#             # chunk: (N_hist, D)
#             # curr_img: (N_curr, D)
#             # sim_chunk: (N_hist, N_curr) -> This is much smaller than (Total_Hist, N_curr)
            
#             # We want max over history dimension (dim 0)
#             # Optimization: We don't need the full matrix, just the max values.
#             # But torch.matmul is fastest way.
            
#             sim_chunk = torch.matmul(chunk, curr_img.T)
#             max_sim_chunk = sim_chunk.amax(dim=0) # Shape (N_curr,)
            
#             # Update global max
#             global_max_sim = torch.maximum(global_max_sim, max_sim_chunk)
            
#         # Determine which tokens to keep in current image
#         local_keep_mask = global_max_sim < threshold
        
#         # 1. Add kept tokens to history for FUTURE images
#         # Only slice and store what we need. 
#         if local_keep_mask.any():
#             kept_tokens = curr_img[local_keep_mask]
#             kept_history_chunks.append(kept_tokens)
            
#             # 2. Record global indices
#             # Local indices of kept tokens
#             local_indices = torch.nonzero(local_keep_mask).squeeze(-1)
#             # Convert to global scope
#             global_indices = local_indices + current_offset
#             kept_global_indices.append(global_indices)
        
#         current_offset += n_curr

#     # Concatenate all indices at the very end
#     if not kept_global_indices:
#         return torch.tensor([], device=device, dtype=torch.long)
        
#     embeds_to_keep_idx = torch.cat(kept_global_indices)

#     # Handle Limit (Post-process)
#     if limit is not None and embeds_to_keep_idx.numel() > limit:
#         # Random subset
#         subset = torch.randperm(embeds_to_keep_idx.numel(), device=device)[:limit]
#         embeds_to_keep_idx = embeds_to_keep_idx[subset]
#         embeds_to_keep_idx, _ = embeds_to_keep_idx.sort()

#     return embeds_to_keep_idx

class Qwen3VLSparseTextModel(Qwen3VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwarg,
    ):
        # print(input_ids.shape)
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        self.seq_keep_mask = None
        self.vis_keep_mask = None
        # 1. Determine Sequence Length
        if inputs_embeds is not None:
            B, S, H = inputs_embeds.shape
            device = inputs_embeds.device
        elif input_ids is not None:
            B, S = input_ids.shape
            device = input_ids.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # 2. Compute Sparse Mask
        # Default: Keep everything (True)
        final_keep_mask = torch.ones((S,), device=device, dtype=torch.bool,requires_grad=False)

        if inputs_embeds is not None and visual_pos_masks is not None:
            with torch.no_grad():
                # A. Initialize a batch-level mask
                # Shape: (B, S). Init to True. 
                batch_keep_mask = torch.ones((B, S), device=device, dtype=torch.bool)
                ds_keep_mask = torch.zeros(deepstack_visual_embeds[0].shape[0],dtype=torch.bool)
                # B. Mark ALL visual positions as False initially (we will only add back the keepers)
                batch_keep_mask[visual_pos_masks] = False
                
                # C. Extract and Filter
                image_embeds_list = extract_contiguous_segments(inputs_embeds, visual_pos_masks)
                
                for b, image_embeds in enumerate(image_embeds_list):
                    if not image_embeds: # Handle cases with no images
                        continue
                        
                    # Get indices of embeddings to KEEP (relative to the visual segments)
                    embeds_to_keep_rel_idx = filter_embeds(image_embeds,limit=27000,threshold=0.94) #TODO: eliminate magic numbers
                    
                    # MAP RELATIVE INDICES -> GLOBAL INDICES
                    # Get global indices where this batch has visual tokens
                    global_visual_indices = torch.nonzero(visual_pos_masks[b]).squeeze()
                    
                    # Select the global indices corresponding to the kept relative indices
                    global_indices_to_keep = global_visual_indices[embeds_to_keep_rel_idx]
                    
                    # Set these specific visual tokens back to True
                    batch_keep_mask[b, global_indices_to_keep] = True
                    ds_keep_mask[embeds_to_keep_rel_idx] = True
                # D. Union over batch (Keep token if ANY sequence in the batch needs it)
                # This maintains a rectangular tensor shape required by standard attention
                final_keep_mask = batch_keep_mask.any(dim=0) 

                # E. Sparsify Deepstack args if present
                if deepstack_visual_embeds is not None:
                    deepstack_visual_embeds = [
                        v[ds_keep_mask, :] for v in deepstack_visual_embeds
                    ]
                print(f"keeping {torch.sum(ds_keep_mask)}/{len(ds_keep_mask)}")
        # 3. Apply Mask to Inputs
        # Helper to slice only if tensor is not None
        def apply_mask(t):
            if t is None: return None
            # Handle tensors that might be (B, S, ...) or (B, 1, S, S)
            if t.shape[1] == S:
                return t[:, final_keep_mask]
            return t # Fallback for shapes that don't match S

        input_ids = apply_mask(input_ids)
        attention_mask = apply_mask(attention_mask)#.detach()
        position_ids = position_ids[:,:,final_keep_mask]#.detach()
        input_embeds = apply_mask(inputs_embeds)#.detach()
        # input_embeds.requires_grad_()
        visual_pos_masks = apply_mask(visual_pos_masks)#.detach()

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks, 
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwarg,
        )
        self.seq_keep_mask = final_keep_mask.cpu()
        self.vis_keep_mask = ds_keep_mask.cpu()
        return outputs
        # print("assigning hidden states")
        # last_hidden_state = torch.zeros(B,S,outputs.last_hidden_state.shape[-1],dtype=self.dtype,device=outputs.last_hidden_state.device)
        # print("initialized empty hidden")
        # last_hidden_state[:,final_keep_mask]=outputs.last_hidden_state
        # return BaseModelOutputWithPast(
        #     last_hidden_state=last_hidden_state,
        #     past_key_values=outputs.past_key_values,
        # )

        # return outputs

# --- 2. The Middle Man (The Composite Model) ---
class Qwen3VLSparseModel(Qwen3VLModel):
    """
    Overrides the main model to inject the Sparse Text Backbone.
    Assumes Qwen3VLModel initializes a self.text_model.
    """
    def __init__(self, config):
        super(Qwen3VLModel,self).__init__(config) #grandparent
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLSparseTextModel(config.text_config)
        self.rope_deltas = None
        # Re-run post_init to ensure weights (if initialized randomly) are correct,
        # though usually we load pretrained immediately after.
        self.post_init()

# --- 3. The Top-Level Conditional Generation Model ---
class Qwen3VLSparseForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    The user-facing class. 
    It replaces the internal model with Qwen3VLSparseModel.
    """
    def __init__(self, config):
        # We do NOT call super().__init__(config) because that would instantiate
        # the heavy standard Qwen3VLModel, which we would immediately throw away.
        # Instead, we init the Grandparent (PreTrainedModel) and build manually.
        
        # Init Qwen3VLPreTrainedModel
        super(Qwen3VLForConditionalGeneration, self).__init__(config) 
        
        # Instantiate OUR Sparse Model
        self.model = Qwen3VLSparseModel(config)
        
        # Standard LM Head (same as original)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            labels = labels[:,self.model.language_model.seq_keep_mask]
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
    
        return ModelOutput({
            "loss":loss,
            "logits": logits,
            "past_key_values": outputs.past_key_values,
            "rope_deltas": outputs.rope_deltas,
            "seq_keep_mask": self.model.language_model.seq_keep_mask
        })
