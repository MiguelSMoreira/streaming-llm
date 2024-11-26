import warnings

warnings.filterwarnings("ignore")

import json
import torch
import argparse

from retriever.retriever import Retriever
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm


TEMPLATE_RETRIEVER = """
<s>[INST] <<SYS>> You are a helpful assistant. Use the following information to answer the user's question. <</SYS>>
{retrieved_context}
USER: {user_message} [/INST]
"""

TEMPLATE_SIMPLE = """
<s>[INST] USER: {user_message} [/INST]
"""


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
    retriever=None,
    preamble=False,
    debug_retriever=False
):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        # Construct the current prompt
        if preamble:
            print(f"\n\nUSER: {prompt}")
        if retriever:
            retriever_results = retriever.retrieve(prompt)
            if debug_retriever:
                emit_retriever_results(retriever_results)
            prompt = TEMPLATE_RETRIEVER.format(
                retrieved_context=retriever_results, user_message=prompt
            )
        else:
            prompt = TEMPLATE_SIMPLE.format(user_message=prompt)
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

        # Store with retriever
        if retriever:
            retriever.add_to_contextwindow(prompt)
            retriever.add_to_contextwindow(output)


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
    if args.enable_retriever:
        # Tokenize and ingest to the vectorDB
        retriever = Retriever(tokenizer, args.recent_size)
    else:
        retriever = None

    # Check the mode based on enable_interactive
    if args.enable_interactive:
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
                retriever=retriever,
                debug_retriever=args.debug_retriever,
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
            retriever=retriever,
            preamble=True,
            debug_retriever=args.debug_retriever,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--debug_retriever", action="store_true")
    parser.add_argument(
        "--enable_retriever",
        action="store_true",
        default=False,
        help="Enable retrieval-augmented generation",
    )
    parser.add_argument(
        "--enable_interactive",
        action="store_true",
        default=False,
        help="Enable interactive input mode",
    )
    parser.add_argument("--input_file_path", type=str)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main()
