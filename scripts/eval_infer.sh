#!/bin/bash

ip_path="/ads_ds3/data/nemotron_parse/nemo_gym_data/inference.jsonl"
op_path="/ads_ds3/data/nemotron_parse/nemo_gym_data/vllm_verif_rewards_6k"
mkdir -p $op_path

OP_FILE="${op_path}/nemo_gym_rollouts.jsonl"
LOG_FILE="${op_path}/rollouts.log"
ng_collect_rollouts \
  +agent_name=nemotron_parse_agent \
  +input_jsonl_fpath=$ip_path \
  +output_jsonl_fpath=$OP_FILE \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 6000, temperature: 0.0}"

python /raid/home/avem/nemogym/data/nemo_gym_data/strip_input.py --input $OP_FILE

exit 0
### End of example commands ###
# Example commands:
# Start servers
#  ng_run "+config_paths=[resources_servers/nemotron_parse/configs/nemotron_parse.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

  # Smoke test with example data (5 entries)
#  ng_collect_rollouts \
#    +agent_name=doc_eval_agent \
#    +input_jsonl_fpath=resources_servers/doc_eval/data/example.jsonl \
#    +output_jsonl_fpath=results/rollouts.jsonl \
#    +num_repeats=5 \
#    "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"


#/ads_ds3/data/nemotron_parse/inference.jsonl #All 37 images for full inference
#/ads_ds3/data/nemotron_parse/nemo_gym_rollouts.jsonl #Results for 37 bounding box parse outputs
#/ads_ds3/data/nemotron_parse/generate_inference_jsonl.py #Script to regenerate the JSONL from PNG files

#To start the ng server:

#RAY_TMPDIR=/tmp ng_run "+config_paths=[resources_servers/nemotron_parse/configs/nemotron_parse.yaml]"

#To collect rollouts:
#ng_collect_rollouts \
#  +agent_name=nemotron_parse_agent \
#  +input_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/inference.jsonl \
#  +output_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/verif_rewards_16k/nemo_gym_rollouts.jsonl \
#  +num_repeats=1 \
#  "+responses_create_params={max_output_tokens: 16000, temperature: 0.0}"

#test example data
#  ng_collect_rollouts +agent_name=nemotron_parse_agent +input_jsonl_fpath=resources_server/nemotron_parse/data/example.jsonl +output_jsonl_fpath=/tmp/nemotron_parse_test.jsonl +num_repeats=1 "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"