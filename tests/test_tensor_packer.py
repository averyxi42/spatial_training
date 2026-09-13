import numpy as np
import pytest
import torch

from longnav.utils.tensor_utils import TensorPacker


def test_bfloat16_pack_preserves_bits_without_float32_expansion():
    source = torch.tensor(
        [[-3.5, -0.0, 0.125], [1.0, 17.25, float("nan")]],
        dtype=torch.bfloat16,
    )

    packed, metadata = TensorPacker.pack(source)
    restored = TensorPacker.unpack(packed, metadata)

    assert packed.dtype == np.uint16
    assert packed.nbytes == source.numel() * 2
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored.view(torch.uint16), source.view(torch.uint16))


def test_uint16_bit_storage_rejects_non_bfloat16_metadata():
    with pytest.raises(ValueError, match="only valid for bfloat16"):
        TensorPacker.unpack(
            np.asarray([0], dtype=np.uint16),
            {"dtype": "float32", "storage_dtype": "uint16_bits"},
        )
