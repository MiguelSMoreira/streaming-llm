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
    print("=== RETRIEVER RESULTS ===")
    for result in results:
        print(result)
    print("=== END RETRIEVER RESULTS ===")


@torch.no_grad()
def streaming_inference(
    model,
    tokenizer,
    prompts,
    kv_cache=None,
    max_gen_len=1000,
    retriever: Optional[Retriever] = None,
    debug_retriever=False,
):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        if retriever:
            retriever_results = retriever.retrieve(prompt)
            if debug_retriever:
                emit_retriever_results(retriever_results)
            prompt = (
                " ".join(retriever_results) + "\n\nUSER: " + prompt + "\n\nASSISTANT: "
            )
        else:
            prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
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


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    retriever = Retriever(tokenizer, args.recent_size)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model, tokenizer, prompts, kv_cache,
        retriever=retriever,
        debug_retriever=args.debug_retriever,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--debug_retriever", action="store_true")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
