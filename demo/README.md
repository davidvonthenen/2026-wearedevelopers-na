# Setup

## OS Prep

```bash
sudo apt-get update
sudo apt-get install -y zip unzip wget
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  python3-dev \
  git \
  git-lfs \
  curl \
  jq \
  rsync \
  tmux \
  haproxy
```

# Download Our Model

```bash
mkdir -p ~/model
cd ~/model

curl -L -o config.json  https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/config.json?download=true
curl -L -o generation_config.json https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/generation_config.json?download=true
curl -L -o merges.txt https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/merges.txt?download=true
curl -L -o model-00001-of-00002.safetensors https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/model-00001-of-00002.safetensors?download=true
curl -L -o model-00002-of-00002.safetensors https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/model-00002-of-00002.safetensors?download=true
curl -L -o model.safetensors.index.json https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/model.safetensors.index.json?download=true
curl -L -o tokenizer.json https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/tokenizer.json?download=true
curl -L -o tokenizer_config.json https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/tokenizer_config.json?download=true
curl -L -o vocab.json https://huggingface.co/Qwen/Qwen3-8B-FP8/resolve/main/vocab.json?download=true
```

## Install

Install `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

IMPORTANT: Download the model to the OS https://huggingface.co/Qwen/Qwen3-8B-FP8/tree/main
IMPORTANT: Validate the CUDA version, run: ./toolkit-version.sh

Create a uv virtual environment!

```bash
uv pip install -r requirements.txt
uv pip install lmcache==0.5.4

# ARM
uv pip install vllm==0.27.1 --torch-backend="cu128"

# H100
pip install vllm --extra-index-url https://vllm.ai

# OLDER COMMAND
# uv pip install vllm --torch-backend="cu128"
```

IMPORTANT: Validaet the environment, run: python validate-env.py

## Configuration

### LMCache

```bash
lmcache server \
  --host 127.0.0.1 \
  --port 5555 \
  --http-port 8080 \
  --l1-size-gb 8 \
  --eviction-policy LRU \
  --chunk-size 256 \
  2>&1 \
  | tee "$HOME/kv-demo/logs/lmcache.log"
```

### Instance A

```bash
export KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "127.0.0.1",
    "lmcache.mp.port": 5555
  }
}'

CUDA_VISIBLE_DEVICES=0 \
vllm serve /home/ubuntu/wearedevs-na-2026-1/Qwen3-8B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-8B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.40 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 \
  | tee "$HOME/kv-demo/logs/vllm-a.log"
```

If fails:

```bash
--gpu-memory-utilization 0.40
```

Then:

```bash
--max-model-len 12288
```


### Instance B

```bash
export KV_TRANSFER_CONFIG='{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "127.0.0.1",
    "lmcache.mp.port": 5555
  }
}'

CUDA_VISIBLE_DEVICES=0 \
vllm serve /home/ubuntu/wearedevs-na-2026-1/Qwen3-8B-FP8 \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name Qwen3-8B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.40 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  2>&1 \
  | tee "$HOME/kv-demo/logs/vllm-b.log"
```

If fails:

```bash
--gpu-memory-utilization 0.40
```

Then:

```bash
--max-model-len 12288
```

## Service Checks

```bash
curl -sf \
  http://127.0.0.1:8080/metrics \
  >/dev/null \
  && echo "LMCache HTTP endpoint: ready"

curl -s \
  http://127.0.0.1:8000/v1/models \
  | jq

curl -s \
  http://127.0.0.1:8001/v1/models \
  | jq
```

## Test Driver

```bash
python cross_instance_lmcache_demo.py \
  --instance-a http://127.0.0.1:8000 \
  --instance-b http://127.0.0.1:8001 \
  --lmcache-url http://127.0.0.1:8080 \
  --model Qwen3-8B-FP8 \
  --tokenizer /home/ubuntu/wearedevs-na-2026-1/Qwen3-8B-FP8 \
  --prefix-tokens 12288 \
  --max-output-tokens 8 \
  --output-json "$HOME/kv-demo/results/cross-instance-fp8.json"
```

## Mutli-Round QA

### HA Proxy

#### HA Proxy Config

```bash
cat > "$HOME/kv-demo/haproxy.cfg" <<'EOF'
global
  maxconn 256

defaults
  mode http
  timeout connect 5s
  timeout client 300s
  timeout server 300s

frontend qwen_frontend
  bind 127.0.0.1:7999
  default_backend qwen_pool

backend qwen_pool
  balance roundrobin
  server instance_a 127.0.0.1:8000
  server instance_b 127.0.0.1:8001
EOF
```

#### HA Proxy

```bash
haproxy \
  -db \
  -f "$HOME/kv-demo/haproxy.cfg"
```

#### Test Service

```bash
curl -s \
  http://127.0.0.1:7999/v1/models \
  | jq
```

## Run the Multi-Round QA Demo

### Clear Cache

```bash
rm -rf "$HOME/kv-demo/results"
mkdir -p "$HOME/kv-demo/results"

curl -s \
  -X POST \
  http://127.0.0.1:8080/cache/clear

curl -s \
  -X POST \
  http://127.0.0.1:8080/metrics/reset
```

### Run Demo

IMPORTANT: multi-round-qa.py is found in LMCache Github: https://github.com/LMCache/LMCache/tree/dev/benchmarks/multi_round_qa

```bash
python3 multi-round-qa.py \
  --num-users 4 \
  --num-rounds 6 \
  --qps 0.5 \
  --shared-system-prompt 8192 \
  --user-history-prompt 1024 \
  --answer-len 32 \
  --time 120 \
  --model Qwen3-8B-FP8 \
  --base-url http://127.0.0.1:7999/v1 \
  --output "$HOME/kv-demo/results/lmcache-mp.csv"
```
