import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from retriever.retriever import Retriever
from typing import Optional


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
    return past_key_values, " ".join(generated_text[pos:])


def emit_retriever_results(results):
    """
    Print out all the retriever results to manually check that it's (roughly?) working
    """
    # want to make sure iterating over this doesn't consume it
    assert type(results) == list
    print("\n=== RETRIEVER RESULTS ===")
    for result in results:
        print(result)
    print("=== END RETRIEVER RESULTS ===\n")


@torch.no_grad()
def streaming_inference(
    model,
    tokenizer,
    prompts,
    kv_cache=None,
    max_gen_len=1000,
    retriever: Optional[Retriever] = None,
    debug_retriever=False,
    preamble=False,
    past_key_values = None
):
    for idx, prompt in enumerate(prompts):
        if preamble:
            print(f"\n\nUSER: {prompt}")
        if retriever:
            retriever_results = retriever.retrieve(prompt)
            if debug_retriever:
                emit_retriever_results(retriever_results)
            prompt = template_retriever.format(
                retrieved_context=retriever_results,
                user_message=prompt
            )
        else:
            prompt = template_simple.format(
                user_message=prompt
            )
        print("ASSISTANT:", end=" ")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values, output = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )

        if retriever:
            retriever.add_to_contextwindow(prompt)
            retriever.add_to_contextwindow(output)
    
    return past_key_values


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    retriever = Retriever(
        tokenizer, 
        context_limit=args.recent_size if not args.enable_always_retriever else 0, 
        chunk_limit=args.chunk_size
    ) if args.enable_retriever else None

    # Determine caching based on streaming
    kv_cache = (
        enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
        if args.enable_streaming else None
    )

    # Check the mode based on enable_interactive
    if args.enable_interactive:
        past_key_values = None
        while True:
            # Get user input from the command line
            user_input = input("\n\nUSER (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            # Perform streaming inference for the user-provided input
            past_key_values = streaming_inference(
                model, 
                tokenizer, 
                [user_input], 
                kv_cache, 
                retriever=retriever, 
                past_key_values=past_key_values,
                debug_retriever=args.debug_retriever,
            )
    else:
        test_filepath = os.path.join(args.data_root, args.file_path)
        print(f"Loading data from {test_filepath} ...")

        # Download the file if it doesn't exist
        if not os.path.exists(test_filepath):
            download_url(
                "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
                args.data_root,
            )
            os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

        # Load and process the file
        list_data = load_jsonl(test_filepath)[0]
        prompts = [sample['content'] for sample in list_data]

        # Perform streaming inference for the loaded prompts
        streaming_inference(
            model, 
            tokenizer, 
            prompts, 
            kv_cache, 
            retriever=retriever, 
            preamble=True, 
            debug_retriever=args.debug_retriever,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="meta-llama/Llama-2-7b-chat-hf",
        # default="lmsys/vicuna-13b-v1.3",
    )
    parser.add_argument("--debug_retriever", action="store_true", default=False)
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--file_path", type=str, default="mt_bench.jsonl")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--enable_interactive", action="store_true", default=False, help="Enable interactive input mode")
    parser.add_argument("--enable_retriever", action="store_true", default=False, help="Enable retrieval-augmented generation")
    parser.add_argument("--enable_always_retriever", action="store_true", default=False)
    args = parser.parse_args()

    template_retriever = \
    "<s>[INST] <<SYS>> You are a helpful assistant who provides concise and accurate answers. The following context is relevant to the user's query: " +\
    "{retrieved_context} <</SYS>> " +\
    "{user_message} [/INST]" if "Llama-2" in args.model_name_or_path else \
    "USER: This might be useful to answer the following question: {retrieved_context} \n" +\
    "{user_message} \n" +\
    "ASSISTANT: "\

    template_simple = \
    "<s>[INST] <<SYS>> You are a helpful assistant who provides concise and accurate answers. <</SYS>>" +\
    "{user_message} [/INST]" if "LLama-2" in args.model_name_or_path else \
    "USER: {user_message} \n" +\
    "ASSISTANT: "

    main(args)
