import warnings

warnings.filterwarnings("ignore")

import json
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from tqdm import tqdm
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        # Construct the current prompt
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )


def load_prompts(file_path):
    # Read in all prompts into a list
    with open(file_path, "r") as f:
        file_content = json.load(f)
    # Merge all prompts into a single string
    return [i["content"] for i in file_content]


def load_prompt(file_path):
    # Read in all prompts into a list
    with open(file_path, "r") as f:
        file_content = json.load(f)
    # Merge all prompts into a single string
    return [". ".join([i["content"] for i in file_content])]


def main():
    # Parse the inputs
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    # Enable streaming if specified
    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    # Enable RAG if specified
    if args.enable_rag:
        # Tokenize and ingest to the vectorDB
        pass
    else:
        pass

    # Check the mode based on enable_iterative
    if args.enable_iterative:
        while True:
            # Get user input from the command line
            user_input = input("Enter a prompt (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            # Perform streaming inference for the user-provided input
            streaming_inference(
                model=model,
                tokenizer=tokenizer,
                prompts=[user_input],
                kv_cache=kv_cache,
            )
    else:
        # Load input from specified file path
        input_prompts = load_prompt(args.input_file_path)

        # Run streamingLLM inference
        streaming_inference(
            model=model,
            tokenizer=tokenizer,
            prompts=input_prompts,
            kv_cache=kv_cache,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--enable_rag", action="store_true")
    parser.add_argument(
        "--enable_iterative",
        action="store_true",
        default=False,
        help="Enable iterative input mode",
    )
    parser.add_argument("--input_file_path", type=str)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main()
