gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
port=$((8000 + ${GPULIST[0]}))

vllm serve --port $port Qwen/Qwen2-VL-7B-Instruct