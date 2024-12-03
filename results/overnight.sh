for file in "prompt_context_length_1000_depth_percent0" "prompt_context_length_1000_depth_percent10" "prompt_context_length_1000_depth_percent100" "prompt_context_length_1000_depth_percent20" "prompt_context_length_1000_depth_percent30" "prompt_context_length_1000_depth_percent40" "prompt_context_length_1000_depth_percent50" "prompt_context_length_1000_depth_percent60" "prompt_context_length_1000_depth_percent70" "prompt_context_length_1000_depth_percent80" "prompt_context_length_1000_depth_percent90" "prompt_context_length_2000_depth_percent0" "prompt_context_length_2000_depth_percent10" "prompt_context_length_2000_depth_percent100" "prompt_context_length_2000_depth_percent20" "prompt_context_length_2000_depth_percent30" "prompt_context_length_2000_depth_percent40" "prompt_context_length_2000_depth_percent50" "prompt_context_length_2000_depth_percent60" "prompt_context_length_2000_depth_percent70" "prompt_context_length_2000_depth_percent80" "prompt_context_length_2000_depth_percent90" "prompt_context_length_3000_depth_percent0" "prompt_context_length_3000_depth_percent10" "prompt_context_length_3000_depth_percent100" "prompt_context_length_3000_depth_percent20" "prompt_context_length_3000_depth_percent30" "prompt_context_length_3000_depth_percent40" "prompt_context_length_3000_depth_percent50" "prompt_context_length_3000_depth_percent60" "prompt_context_length_3000_depth_percent70" "prompt_context_length_3000_depth_percent80" "prompt_context_length_3000_depth_percent90" "prompt_context_length_4000_depth_percent0" "prompt_context_length_4000_depth_percent10" "prompt_context_length_4000_depth_percent100" "prompt_context_length_4000_depth_percent20" "prompt_context_length_4000_depth_percent30" "prompt_context_length_4000_depth_percent40" "prompt_context_length_4000_depth_percent50" "prompt_context_length_4000_depth_percent60" "prompt_context_length_4000_depth_percent70" "prompt_context_length_4000_depth_percent80" "prompt_context_length_4000_depth_percent90" "prompt_context_length_5000_depth_percent0" "prompt_context_length_5000_depth_percent10" "prompt_context_length_5000_depth_percent100" "prompt_context_length_5000_depth_percent20" "prompt_context_length_5000_depth_percent30" "prompt_context_length_5000_depth_percent40" "prompt_context_length_5000_depth_percent50" "prompt_context_length_5000_depth_percent60" "prompt_context_length_5000_depth_percent70" "prompt_context_length_5000_depth_percent80" "prompt_context_length_5000_depth_percent90" "prompt_context_length_6000_depth_percent0" "prompt_context_length_6000_depth_percent10" "prompt_context_length_6000_depth_percent100" "prompt_context_length_6000_depth_percent20" "prompt_context_length_6000_depth_percent30" "prompt_context_length_6000_depth_percent40" "prompt_context_length_6000_depth_percent50" "prompt_context_length_6000_depth_percent60" "prompt_context_length_6000_depth_percent70" "prompt_context_length_6000_depth_percent80" "prompt_context_length_6000_depth_percent90" "prompt_context_length_7000_depth_percent0" "prompt_context_length_7000_depth_percent10" "prompt_context_length_7000_depth_percent100" "prompt_context_length_7000_depth_percent20" "prompt_context_length_7000_depth_percent30" "prompt_context_length_7000_depth_percent40" "prompt_context_length_7000_depth_percent50" "prompt_context_length_7000_depth_percent60" "prompt_context_length_7000_depth_percent70" "prompt_context_length_7000_depth_percent80" "prompt_context_length_7000_depth_percent90" "prompt_context_length_8000_depth_percent0" "prompt_context_length_8000_depth_percent10" "prompt_context_length_8000_depth_percent100" "prompt_context_length_8000_depth_percent20" "prompt_context_length_8000_depth_percent30" "prompt_context_length_8000_depth_percent40" "prompt_context_length_8000_depth_percent50" "prompt_context_length_8000_depth_percent60" "prompt_context_length_8000_depth_percent70" "prompt_context_length_8000_depth_percent80" "prompt_context_length_8000_depth_percent90" "prompt_context_length_9000_depth_percent0" "prompt_context_length_9000_depth_percent10" "prompt_context_length_9000_depth_percent100" "prompt_context_length_9000_depth_percent20" "prompt_context_length_9000_depth_percent30" "prompt_context_length_9000_depth_percent40" "prompt_context_length_9000_depth_percent50" "prompt_context_length_9000_depth_percent60" "prompt_context_length_9000_depth_percent70" "prompt_context_length_9000_depth_percent80" "prompt_context_length_9000_depth_percent90" "prompt_context_length_10000_depth_percent0" "prompt_context_length_10000_depth_percent10" "prompt_context_length_10000_depth_percent100" "prompt_context_length_10000_depth_percent20" "prompt_context_length_10000_depth_percent30" "prompt_context_length_10000_depth_percent40" "prompt_context_length_10000_depth_percent50" "prompt_context_length_10000_depth_percent60" "prompt_context_length_10000_depth_percent70" "prompt_context_length_10000_depth_percent80" "prompt_context_length_10000_depth_percent90"; do
	python examples/run_streaming_llama.py  --enable_streaming --model_name_or_path meta-llama/Llama-2-7b-chat-hf --enable_retriever --debug_retriever --chunk_size 100 --recent_size 4096 --file_path "${file}.json" > "results/llama2_chunk_100_size_4096_w_retriever/${file}.txt"
done

for file in "prompt_context_length_1000_depth_percent0" "prompt_context_length_1000_depth_percent10" "prompt_context_length_1000_depth_percent100" "prompt_context_length_1000_depth_percent20" "prompt_context_length_1000_depth_percent30" "prompt_context_length_1000_depth_percent40" "prompt_context_length_1000_depth_percent50" "prompt_context_length_1000_depth_percent60" "prompt_context_length_1000_depth_percent70" "prompt_context_length_1000_depth_percent80" "prompt_context_length_1000_depth_percent90" "prompt_context_length_2000_depth_percent0" "prompt_context_length_2000_depth_percent10" "prompt_context_length_2000_depth_percent100" "prompt_context_length_2000_depth_percent20" "prompt_context_length_2000_depth_percent30" "prompt_context_length_2000_depth_percent40" "prompt_context_length_2000_depth_percent50" "prompt_context_length_2000_depth_percent60" "prompt_context_length_2000_depth_percent70" "prompt_context_length_2000_depth_percent80" "prompt_context_length_2000_depth_percent90" "prompt_context_length_3000_depth_percent0" "prompt_context_length_3000_depth_percent10" "prompt_context_length_3000_depth_percent100" "prompt_context_length_3000_depth_percent20" "prompt_context_length_3000_depth_percent30" "prompt_context_length_3000_depth_percent40" "prompt_context_length_3000_depth_percent50" "prompt_context_length_3000_depth_percent60" "prompt_context_length_3000_depth_percent70" "prompt_context_length_3000_depth_percent80" "prompt_context_length_3000_depth_percent90" "prompt_context_length_4000_depth_percent0" "prompt_context_length_4000_depth_percent10" "prompt_context_length_4000_depth_percent100" "prompt_context_length_4000_depth_percent20" "prompt_context_length_4000_depth_percent30" "prompt_context_length_4000_depth_percent40" "prompt_context_length_4000_depth_percent50" "prompt_context_length_4000_depth_percent60" "prompt_context_length_4000_depth_percent70" "prompt_context_length_4000_depth_percent80" "prompt_context_length_4000_depth_percent90" "prompt_context_length_5000_depth_percent0" "prompt_context_length_5000_depth_percent10" "prompt_context_length_5000_depth_percent100" "prompt_context_length_5000_depth_percent20" "prompt_context_length_5000_depth_percent30" "prompt_context_length_5000_depth_percent40" "prompt_context_length_5000_depth_percent50" "prompt_context_length_5000_depth_percent60" "prompt_context_length_5000_depth_percent70" "prompt_context_length_5000_depth_percent80" "prompt_context_length_5000_depth_percent90" "prompt_context_length_6000_depth_percent0" "prompt_context_length_6000_depth_percent10" "prompt_context_length_6000_depth_percent100" "prompt_context_length_6000_depth_percent20" "prompt_context_length_6000_depth_percent30" "prompt_context_length_6000_depth_percent40" "prompt_context_length_6000_depth_percent50" "prompt_context_length_6000_depth_percent60" "prompt_context_length_6000_depth_percent70" "prompt_context_length_6000_depth_percent80" "prompt_context_length_6000_depth_percent90" "prompt_context_length_7000_depth_percent0" "prompt_context_length_7000_depth_percent10" "prompt_context_length_7000_depth_percent100" "prompt_context_length_7000_depth_percent20" "prompt_context_length_7000_depth_percent30" "prompt_context_length_7000_depth_percent40" "prompt_context_length_7000_depth_percent50" "prompt_context_length_7000_depth_percent60" "prompt_context_length_7000_depth_percent70" "prompt_context_length_7000_depth_percent80" "prompt_context_length_7000_depth_percent90" "prompt_context_length_8000_depth_percent0" "prompt_context_length_8000_depth_percent10" "prompt_context_length_8000_depth_percent100" "prompt_context_length_8000_depth_percent20" "prompt_context_length_8000_depth_percent30" "prompt_context_length_8000_depth_percent40" "prompt_context_length_8000_depth_percent50" "prompt_context_length_8000_depth_percent60" "prompt_context_length_8000_depth_percent70" "prompt_context_length_8000_depth_percent80" "prompt_context_length_8000_depth_percent90" "prompt_context_length_9000_depth_percent0" "prompt_context_length_9000_depth_percent10" "prompt_context_length_9000_depth_percent100" "prompt_context_length_9000_depth_percent20" "prompt_context_length_9000_depth_percent30" "prompt_context_length_9000_depth_percent40" "prompt_context_length_9000_depth_percent50" "prompt_context_length_9000_depth_percent60" "prompt_context_length_9000_depth_percent70" "prompt_context_length_9000_depth_percent80" "prompt_context_length_9000_depth_percent90" "prompt_context_length_10000_depth_percent0" "prompt_context_length_10000_depth_percent10" "prompt_context_length_10000_depth_percent100" "prompt_context_length_10000_depth_percent20" "prompt_context_length_10000_depth_percent30" "prompt_context_length_10000_depth_percent40" "prompt_context_length_10000_depth_percent50" "prompt_context_length_10000_depth_percent60" "prompt_context_length_10000_depth_percent70" "prompt_context_length_10000_depth_percent80" "prompt_context_length_10000_depth_percent90"; do
	python examples/run_streaming_llama.py  --enable_streaming --model_name_or_path meta-llama/Llama-2-7b-chat-hf --chunk_size 100 --recent_size 4096 --file_path "${file}.json" > "results/llama2_chunk_100_size_4096_no_retriever/${file}.txt"
done

for file in "prompt_context_length_1000_depth_percent0" "prompt_context_length_1000_depth_percent10" "prompt_context_length_1000_depth_percent100" "prompt_context_length_1000_depth_percent20" "prompt_context_length_1000_depth_percent30" "prompt_context_length_1000_depth_percent40" "prompt_context_length_1000_depth_percent50" "prompt_context_length_1000_depth_percent60" "prompt_context_length_1000_depth_percent70" "prompt_context_length_1000_depth_percent80" "prompt_context_length_1000_depth_percent90" "prompt_context_length_2000_depth_percent0" "prompt_context_length_2000_depth_percent10" "prompt_context_length_2000_depth_percent100" "prompt_context_length_2000_depth_percent20" "prompt_context_length_2000_depth_percent30" "prompt_context_length_2000_depth_percent40" "prompt_context_length_2000_depth_percent50" "prompt_context_length_2000_depth_percent60" "prompt_context_length_2000_depth_percent70" "prompt_context_length_2000_depth_percent80" "prompt_context_length_2000_depth_percent90" "prompt_context_length_3000_depth_percent0" "prompt_context_length_3000_depth_percent10" "prompt_context_length_3000_depth_percent100" "prompt_context_length_3000_depth_percent20" "prompt_context_length_3000_depth_percent30" "prompt_context_length_3000_depth_percent40" "prompt_context_length_3000_depth_percent50" "prompt_context_length_3000_depth_percent60" "prompt_context_length_3000_depth_percent70" "prompt_context_length_3000_depth_percent80" "prompt_context_length_3000_depth_percent90" "prompt_context_length_4000_depth_percent0" "prompt_context_length_4000_depth_percent10" "prompt_context_length_4000_depth_percent100" "prompt_context_length_4000_depth_percent20" "prompt_context_length_4000_depth_percent30" "prompt_context_length_4000_depth_percent40" "prompt_context_length_4000_depth_percent50" "prompt_context_length_4000_depth_percent60" "prompt_context_length_4000_depth_percent70" "prompt_context_length_4000_depth_percent80" "prompt_context_length_4000_depth_percent90" "prompt_context_length_5000_depth_percent0" "prompt_context_length_5000_depth_percent10" "prompt_context_length_5000_depth_percent100" "prompt_context_length_5000_depth_percent20" "prompt_context_length_5000_depth_percent30" "prompt_context_length_5000_depth_percent40" "prompt_context_length_5000_depth_percent50" "prompt_context_length_5000_depth_percent60" "prompt_context_length_5000_depth_percent70" "prompt_context_length_5000_depth_percent80" "prompt_context_length_5000_depth_percent90" "prompt_context_length_6000_depth_percent0" "prompt_context_length_6000_depth_percent10" "prompt_context_length_6000_depth_percent100" "prompt_context_length_6000_depth_percent20" "prompt_context_length_6000_depth_percent30" "prompt_context_length_6000_depth_percent40" "prompt_context_length_6000_depth_percent50" "prompt_context_length_6000_depth_percent60" "prompt_context_length_6000_depth_percent70" "prompt_context_length_6000_depth_percent80" "prompt_context_length_6000_depth_percent90" "prompt_context_length_7000_depth_percent0" "prompt_context_length_7000_depth_percent10" "prompt_context_length_7000_depth_percent100" "prompt_context_length_7000_depth_percent20" "prompt_context_length_7000_depth_percent30" "prompt_context_length_7000_depth_percent40" "prompt_context_length_7000_depth_percent50" "prompt_context_length_7000_depth_percent60" "prompt_context_length_7000_depth_percent70" "prompt_context_length_7000_depth_percent80" "prompt_context_length_7000_depth_percent90" "prompt_context_length_8000_depth_percent0" "prompt_context_length_8000_depth_percent10" "prompt_context_length_8000_depth_percent100" "prompt_context_length_8000_depth_percent20" "prompt_context_length_8000_depth_percent30" "prompt_context_length_8000_depth_percent40" "prompt_context_length_8000_depth_percent50" "prompt_context_length_8000_depth_percent60" "prompt_context_length_8000_depth_percent70" "prompt_context_length_8000_depth_percent80" "prompt_context_length_8000_depth_percent90" "prompt_context_length_9000_depth_percent0" "prompt_context_length_9000_depth_percent10" "prompt_context_length_9000_depth_percent100" "prompt_context_length_9000_depth_percent20" "prompt_context_length_9000_depth_percent30" "prompt_context_length_9000_depth_percent40" "prompt_context_length_9000_depth_percent50" "prompt_context_length_9000_depth_percent60" "prompt_context_length_9000_depth_percent70" "prompt_context_length_9000_depth_percent80" "prompt_context_length_9000_depth_percent90" "prompt_context_length_10000_depth_percent0" "prompt_context_length_10000_depth_percent10" "prompt_context_length_10000_depth_percent100" "prompt_context_length_10000_depth_percent20" "prompt_context_length_10000_depth_percent30" "prompt_context_length_10000_depth_percent40" "prompt_context_length_10000_depth_percent50" "prompt_context_length_10000_depth_percent60" "prompt_context_length_10000_depth_percent70" "prompt_context_length_10000_depth_percent80" "prompt_context_length_10000_depth_percent90"; do
	python examples/run_streaming_llama.py  --enable_streaming --model_name_or_path meta-llama/Llama-2-7b-chat-hf --enable_retriever --debug_retriever --enable_always_retriever --chunk_size 100 --recent_size 4096 --file_path "${file}.json" > "results/llama2_chunk_100_size_4096_w_always_retriever/${file}.txt"
done