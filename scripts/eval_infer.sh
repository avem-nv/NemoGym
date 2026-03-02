# Start servers
  ng_run "+config_paths=[resources_servers/doc_eval/configs/doc_eval.yaml,responses_api_models/vllm
  _model/configs/vllm_model.yaml]"

  # Smoke test with example data (5 entries)
  ng_collect_rollouts \
    +agent_name=doc_eval_agent \
    +input_jsonl_fpath=resources_servers/doc_eval/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +num_repeats=5 \
    "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"

  # Profile per-task scores
  ng_reward_profile \
    +input_jsonl_fpath=resources_servers/doc_eval/data/example.jsonl \
    +rollouts_jsonl_fpath=results/rollouts.jsonl \
    +output_jsonl_fpath=results/profiled.jsonl \
    +pass_threshold=1.0

  # Aggregate metrics (pass@1 from avg_reward, pass@k from max_reward)
  python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl


"""
/ads_ds3/data/nemotron_parse/inference.jsonl #All 37 images for full inference
/ads_ds3/data/nemotron_parse/nemo_gym_rollouts.jsonl #Results for 37 bounding box parse outputs
/ads_ds3/data/nemotron_parse/generate_inference_jsonl.py #Script to regenerate the JSONL from PNG files

#To start the ng server:

RAY_TMPDIR=/tmp ng_run "+config_paths=[resources_servers/nemotron_parse/configs/nemotron_parse.yaml]"

#To collect rollouts:
ng_collect_rollouts \
  +agent_name=nemotron_parse_agent \
  +input_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/inference.jsonl \
  +output_jsonl_fpath=/ads_ds3/data/nemotron_parse/nemo_gym_data/verif_rewards/nemo_gym_rollouts.jsonl \
  +num_repeats=2 \
  "+responses_create_params={max_output_tokens: 64000, temperature: 0.1}"

#test example data
  ng_collect_rollouts +agent_name=nemotron_parse_agent +input_jsonl_fpath=resources_server/nemotron_parse/data/example.jsonl +output_jsonl_fpath=/tmp/nemotron_parse_test.jsonl +num_repeats=1 "+responses_create_params={max_output_tokens: 4096, temperature: 0.0}"
"""