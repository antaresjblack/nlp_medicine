# MacBERT config and tokenizer

This directory contains the small `hfl/chinese-macbert-base` configuration and
tokenizer files needed for checkpoint inference:

- `config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `vocab.txt`

It intentionally does **not** contain `pytorch_model.bin` or any other
pretrained model weights. The fine-tuned MacBERT checkpoint is downloaded and
verified separately by `src/reproduce.py`.

File SHA-256 values:

```text
648c8a2b51c12551f94bae19a66c21a59c9bd8aa0ebb99f2a0af7236cdb99daa  config.json
303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3  special_tokens_map.json
61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874  tokenizer_config.json
53ff61207898738bbdc000f38abebef01041c8d23b6270c11855fc692d0a3ad6  tokenizer.json
45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c  vocab.txt
```

