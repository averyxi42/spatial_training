import torch
import numpy as np

class TensorPacker:
    @staticmethod
    def pack(obj):
        """
        Recursively converts torch.Tensors to numpy arrays.
        Returns:
            packed_obj: The original structure with tensors replaced by numpy arrays.
            metadata: A parallel structure containing dtype/device info for reconstruction.
        """
        if isinstance(obj, torch.Tensor):
            # 1. Capture Metadata
            meta = {
                'dtype': str(obj.dtype).split('.')[-1], # e.g. "float32", "bfloat16"
            }
            
            # 2. Convert to Numpy
            # NumPy has no bfloat16 scalar type. Preserve the raw 16-bit payload
            # instead of doubling every cached visual embedding to float32 in Ray.
            if obj.dtype == torch.bfloat16:
                data = obj.detach().cpu().contiguous().view(torch.uint16).numpy()
                meta['storage_dtype'] = 'uint16_bits'
            else:
                data = obj.detach().cpu().numpy()
            
            return data, meta
            
        elif isinstance(obj, dict):
            packed = {}
            meta = {}
            for k, v in obj.items():
                p, m = TensorPacker.pack(v)
                packed[k] = p
                meta[k] = m
            return packed, meta
            
        elif isinstance(obj, list):
            packed = []
            meta = []
            for v in obj:
                p, m = TensorPacker.pack(v)
                packed.append(p)
                meta.append(m)
            return packed, meta
            
        elif isinstance(obj, tuple):
            # Tuples are immutable, so we reconstruct them
            unzipped = [TensorPacker.pack(v) for v in obj]
            packed = tuple(u[0] for u in unzipped)
            meta = tuple(u[1] for u in unzipped)
            return packed, meta
            
        # Base case (int, float, str, None)
        return obj, None

    @staticmethod
    def unpack(packed_obj, metadata, device=None):
        """
        Reconstructs torch tensors from numpy arrays using the provided metadata.
        Args:
            device: you know what this is.
        """
        if isinstance(packed_obj, np.ndarray) and isinstance(metadata, dict):
            # 1. Retrieve Metadata
            target_dtype = getattr(torch, metadata['dtype'])
            
            # 2. Convert back to Tensor
            tensor = torch.from_numpy(packed_obj)
            # 3. Cast and Move
            # Note: from_numpy always creates CPU tensor. Reinterpret BF16 bitwise;
            # numeric uint16->bfloat16 conversion would corrupt every embedding.
            if metadata.get('storage_dtype') == 'uint16_bits':
                if target_dtype is not torch.bfloat16:
                    raise ValueError(
                        "uint16_bits storage is only valid for bfloat16 tensors"
                    )
                return tensor.view(torch.bfloat16)
            return tensor.to(dtype=target_dtype)
            
        elif isinstance(packed_obj, dict):
            return {k: TensorPacker.unpack(v, metadata[k], device) for k, v in packed_obj.items()}
            
        elif isinstance(packed_obj, list):
            return [TensorPacker.unpack(v, m, device) for v, m in zip(packed_obj, metadata)]
            
        elif isinstance(packed_obj, tuple):
            return tuple(TensorPacker.unpack(v, m, device) for v, m in zip(packed_obj, metadata))
            
        return packed_obj
    
