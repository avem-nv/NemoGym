# NemoGym Parse — Setup & Usage Guide

## Prerequisites

1. Complete the environment setup described in the [main README](README.md).
2. No `policy_*` configuration is needed — this workflow uses a vLLM server for inference.

---

## 1. Launch the vLLM Server

Set your `NGC_API_KEY` environment variable, then start the NIM container:

```bash
docker run -it -d \
  --gpus "device=0" \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -u 0:0 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8992:8000 \
  nvcr.io/nim/nvidia/nemotron-parse:latest
```

Verify the server is ready:

```bash
curl http://localhost:8992/v1/health/ready
# Expected: {"object":"health.response","message":"Service is ready."}
```

---

## 2. Configure the Server Address

Ensure the vLLM server address in `env.yaml` matches the port used above (`8992`).

---

## 3. Data Setup

Use `generate_inference.py` to produce an `inference.jsonl` file from your image data:

```
Data path: /ads_ds3/data/nemotron_parse/inference.jsonl
```

Then collect rollouts:

```bash
ng_collect_rollouts \
  +agent_name=nemotron_parse_agent \
  +input_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/inference.jsonl \
  +output_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/verif_rewards1/nemo_gym_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 64000, temperature: 0.0}"
```

The inference output will be written to the path specified by `output_jsonl_fpath`.

---

## 4. Rewards

The reward function is defined in [`resources_servers/nemotron_parse/app.py`](resources_servers/nemotron_parse/app.py).

Running the rollout collection step above will automatically produce a reward score JSON file in the same output directory.
