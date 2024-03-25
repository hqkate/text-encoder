import mindspore as ms
from mindspore import context
from transformers import AutoTokenizer
from t5_encoder.t5 import get_t5_encoder
from mindspore.amp import auto_mixed_precision


context.set_context(
    mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=1
)


textencoder = get_t5_encoder(pretrained="models--DeepFloyd--t5-v1_1-xxl/ms_t5_xxl.ckpt")
tokenizer = AutoTokenizer.from_pretrained("models--DeepFloyd--t5-v1_1-xxl/", TOKENIZERS_PARALLELISM=False)

auto_mixed_precision(textencoder, "O0")

tokens = tokenizer(["an example to test the functionality of t5 model!"])
tokens = {k: ms.Tensor(v, ms.int32) for k, v in tokens.items()  }
# tokens = {
#     'input_ids': ms.Tensor([[  46,  677,   12,  794,    8, 6730,   13,    3,   17,  755,  825,   55,1]], ms.int32),
#     'attention_mask': ms.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], ms.int32)
# }
res = textencoder(**tokens)

# import pdb; pdb.set_trace()
print(res)
