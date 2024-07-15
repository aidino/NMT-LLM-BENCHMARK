
## Run vllm OpenAI Compatible server

```bash
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2-1.5B-Instruct \
--dtype auto \
--api-key token-abc123 
```