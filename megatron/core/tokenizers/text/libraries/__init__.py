# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib

_TOKENIZER_MODULES = {
    "ByteLevelTokenizer": "megatron.core.tokenizers.text.libraries.bytelevel_tokenizer",
    "HuggingFaceTokenizer": "megatron.core.tokenizers.text.libraries.huggingface_tokenizer",
    "MegatronHFTokenizer": "megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer",
    "NullTokenizer": "megatron.core.tokenizers.text.libraries.null_tokenizer",
    "SentencePieceTokenizer": "megatron.core.tokenizers.text.libraries.sentencepiece_tokenizer",
    "SFTTokenizer": "megatron.core.tokenizers.text.libraries.sft_tokenizer",
    "TikTokenTokenizer": "megatron.core.tokenizers.text.libraries.tiktoken_tokenizer",
}

__all__ = sorted(_TOKENIZER_MODULES)


def __getattr__(name):
    if name not in _TOKENIZER_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_TOKENIZER_MODULES[name])
    tokenizer_cls = getattr(module, name)
    globals()[name] = tokenizer_cls
    return tokenizer_cls
