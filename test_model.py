import pandas as pd
import argparse
import pickle
import tqdm
import os
import time
import string

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


def print_utf8(x):
    return print(x.encode("utf-8").decode("utf-8"))

def load_model(cfg_ckpt, weights_ckpt, no_split):

    """

    Load the language model of your choosing. Input should be a string indicating model name

    Returns both model and tokenizer.

    """

    # Load config
    config = AutoConfig.from_pretrained(cfg_ckpt)

    # Load model with empty weights to not take memory
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Let accelerate automatically create optimal device map
    model = load_checkpoint_and_dispatch(model,
                                         weights_ckpt,
                                         device_map="auto",
                                         no_split_module_classes=no_split)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_ckpt)

    return model, tokenizer


def test_prompt(model, tokenizer, prompt, max_tokens=50, min_tokens=3):
    # Move prompt to gpu
    input = tokenizer(prompt, return_tensors="pt").to(0)

    # Get the length of the input tensor (how many tokens are in the prompt)
    input_length = input["input_ids"].shape[1]

    # Generate output based on prompt
    output = model.generate(input["input_ids"], max_new_tokens=max_tokens, min_new_tokens=min_tokens)

    # Slice the output tensor to only include the new tokens
    new_tokens = output[0][input_length:]

    # Decode the output
    #decoded_output = tokenizer.decode(output[0].tolist())
    decoded_output = tokenizer.decode(new_tokens.tolist())

    return decoded_output


def preprocess_prompt(example1, example2, target1, cot, cot_prompt, cot_format, add_format_demo):
    default_prompt = f"If {example1} is like {example2}, then {target1} is like"
    if not cot:
        return default_prompt

    assert cot_format in ['kojima', 'naive']
    if cot_format == 'naive':
        return cot_prompt + default_prompt
    elif cot_format == 'kojima':
        prompt = f"""Q: If {example1} is like {example2}, then what is {target1} like?"""
        if add_format_demo:
            format_demo = '\nNote: The final answer should be given in the format "If (U) is like (V) then (I) is like (J)"'
            prompt = prompt + format_demo
        prompt = prompt +  "\nReasoning: "

        return prompt + cot_prompt


def add_demonstrations(prompt, demos_df, filter_demos, n_demos, category):
    if filter_demos:
        demo_df = demos_df[demos_df["analogy_type"] == category].sample(n=n_demos)
    else:
        demo_df = demos_df.sample(n=n_demos)

    for _, demo_row in demo_df.iterrows():
        demo_first = f"If {demo_row['target']} is like {demo_row['source']} , "
        demo_second = f"then {demo_row['targ_word']} is like {demo_row['src_word']} . \n"
        prompt = demo_first + demo_second + prompt

    return prompt

def evaluate_answer(answer, target, alternatives):
    eval_summary = dict()
    eval_summary['label'] = target
    eval_summary["alternatives"] = alternatives
    targets = [target] + alternatives

    # create answer set

    answer_set = []
    answer_split = answer.split() 
    for i in range(len(answer_split)):
        for j in range(i, len(answer_split)):
            answer_set.append(' '.join(answer_split[i:j+1]))

    print("Answer set")
    print(answer_set)
    correct = False
    for word in targets:
        if word in answer_set:
            correct = True
            break

    eval_summary["correct"] = correct
    return eval_summary


def test_model(data_path, output_path,
               model_params,
               cot=False, cot_prompt="Let's think step by step. \n", cot_format='append',
               add_demos=False, n_demos=4, filter_demos=True,
               min_tokens=3, max_tokens=200, use_alts=True, reverse_analogy=False,
               debug=False, timing=False, add_print=False, add_dots=False, add_format_demo=False):

    # Load model and dataset
    model, tokenizer = load_model(**model_params)
    df = pd.read_csv(data_path)
    if debug:
        df = df.head(10)

    results = dict()
    for index, row in tqdm.tqdm(df.iterrows()):
        if timing:
            start_time = time.time()


        example1 = row["target"]
        example2 = row["source"]
        target1 = row["targ_word"]
        target2 = row["src_word"]

        # Examples with faulty duplicates are skipped
        if example1 == example2:
            continue
        if target1 == target2:
            continue

        if reverse_analogy:
            example1, example2 = example2, example1
            target1, target2 = target2, target1

        prompt = preprocess_prompt(example1, example2, target1, cot, cot_prompt, cot_format, add_format_demo)

        analogy_type = row["analogy_type"]
        if add_demos:
            assert n_demos > 0
            possible_demos = df[~df.index.isin([index])]
            prompt = add_demonstrations(prompt, possible_demos, filter_demos, n_demos, analogy_type)
        else:
            n_demos = 0


        # reasoning is not there if no cot
        reasoning = ""
        if add_print:
            print_utf8("input prompt")
            print_utf8(prompt)
        # If not cot we only generate the answer and no reasoning. So max_tokens is set to 10
        if not cot:
            max_tokens=15

        output = test_prompt(model, tokenizer, prompt, max_tokens, min_tokens)
        if add_print:
            print_utf8("first output")
            print_utf8(output)

        #answer_prompt = f".\nTherefore, the answer is: If {example1} is like {example2}, then {target1} is like"
        # Remove the "A:" in the initial prompt and output to distinguish from final answer
        #prompt = prompt.replace ("A:", "")
        #output = output.replace ("A:", "")

        answer_prompt = f".\nTherefore, the final answer is A: If {example1} is like {example2} then {target1} is like "

        #new_prompt = output + ".\nTherefore, the answer is: "

        if add_dots:
            if not output.endswith(".") or output.endswith("\n") or output.endswith("</s>") or output.endswith("<s>"):
                output = output +"..."
        new_prompt = prompt + output + answer_prompt

        if add_print:
            print_utf8("second prompt")
            print_utf8(new_prompt)
        if cot:
            if cot_format == 'kojima':
                reasoning = output
                output = test_prompt(model, tokenizer, new_prompt, 15, 1)
                if add_print:
                    print_utf8("final output")
                    print_utf8(output)

        alternatives = []
        if use_alts:
            assert type(row["alternatives"]) == str
            alternatives = row["alternatives"].split(", ")

        # old answer computation
        #answer = output.split("like ")[2 * (n_demos + 1)].split(".")[0]

        # Answer ix expected to be in the first 4 generated words
        answer_split = output.split()[:4]

        # Create a translation table mapping every punctuation to None
        table = str.maketrans('', '', string.punctuation)

        special_chars = ["</s>, <s>"]
        answer_list = []
        for word in answer_split:
            word = word.replace('<s>', '').replace('</s>', '')
            answer_list.append(word.translate(table).lower())



        answer = " ".join(answer_list)
        if add_print:
            print_utf8("final answer")
            print_utf8(answer)
            print_utf8("label")
            print_utf8(target2)

        results_summary = {"prompt": prompt,
                           "category": analogy_type,
                           "reasoning": reasoning,
                           "answer": answer}


        eval_summary = evaluate_answer(answer, target2, alternatives)
        results_summary.update(eval_summary)
        results[index] = results_summary

        if timing:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Loop {index} took {elapsed_time} seconds")
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"SAVED RESULTS AT: {output_path}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    # basic params
    argParser.add_argument("--dataset", help="Path to dataset to use",
                           type=str, default="data/SCAN/SCAN_dataset.csv")
    argParser.add_argument("--output", help="Path where results dict should be stored",
                           default="results/baseline.pckl")

    # model params
    argParser.add_argument("--model", default="llama7b", choices=['llama7b', 'gptj', 'alpaca-7b', 'llama-7b', 'vicuna-7b'], help="Which model to test", type=str)
    argParser.add_argument("--cfg_ckpt", help="Path to config checkpoint", type=str)
    argParser.add_argument("--weights_ckpt", help="Path to weights checkpoint", type=str)

    # demonstration params
    argParser.add_argument("--add_demos", help="Use demonstrations", action="store_true")
    argParser.add_argument("--n_demos", default=8, type=int)
    argParser.add_argument("--filter_demos", help="Limit demonstrations to same category as query",
                           action="store_true")

    # chain-of-thought params
    argParser.add_argument("--cot", help="Use CoT reasoning", action="store_true")
    argParser.add_argument("--cot_prompt", help="Sentence to use to prompt cot",
                           type=str, default="Let's think step by step. \n")
    argParser.add_argument("--cot_format", help="How to join cot prompt with the input prompt",
                           type=str, choices=['kojima', 'naive'], default='kojima')


    # other params
    argParser.add_argument("--min_tokens", help="Minimum number of new tokens to generate",
                           type=int, default=3)
    argParser.add_argument("--max_tokens", help="Maximum number of new tokens to generate",
                           type=int, default=50)
    argParser.add_argument("--use_alts", help="use the alternatives or not", action="store_true")
    argParser.add_argument("--reverse_analogy", help="Use function to analyse difference on reversed prompts on SCAN",
                           action="store_true")
    argParser.add_argument("--debug", help="Debug mode", action="store_true")
    argParser.add_argument("--timing", help="Print time of each loop iteration", action="store_true")
    argParser.add_argument("--print", help="Add prints for debugging", action="store_true")
    argParser.add_argument("--add_dots", help="adds '...' to the end of reasoning generated for final answer extraction in cot", action="store_true")
    argParser.add_argument("--add_format_demo", help="Adds format demonstration in cot question 'If A is like B then C is like D'", action="store_true")


    args = argParser.parse_args()

    shared_model_folder = "/project/gpuuva021/shared/analogies/models"
    local_model_folder = "/home/lcur1103/models"
    model = args.model
    model = model.lower()
    if model in ["llama-7b", "alpaca-7b"]:
        model_folder = shared_model_folder
    else:
        model_folder = local_model_folder

    # Set the specifics for the LLama model
    if model.lower() == "llama7b":
        checkpoint = "LLama/llama/converted_models/7B"
        checkpoint2 = checkpoint
        no_split = ["LlamaDecoderLayer"]

    # Set specifics for gptj model
    elif model.lower() == "gptj":
        checkpoint = "EleutherAI/gpt-j-6B"
        checkpoint2 = "sharded-gpt-j-6B"
        no_split = ["GPTJBlock"]
    # alpaca-7b model
    else:
        checkpoint = os.path.join(model_folder, model.lower())
        checkpoint2 = checkpoint
        no_split = ["LlamaDecoderLayer"]
    model_params = {
        "cfg_ckpt": checkpoint if args.cfg_ckpt is None else args.cfg_ckpt,
        "weights_ckpt": checkpoint2 if args.weights_ckpt is None else args.weights_ckpt,
        "no_split": no_split,
    }

    cot_params = {
        'cot': args.cot,
        'cot_prompt': args.cot_prompt,
        'cot_format': args.cot_format
    }

    demonstration_params = {
        'add_demos': args.add_demos,
        'n_demos': args.n_demos,
        'filter_demos': args.filter_demos
    }

    other_params = {
        'min_tokens': args.min_tokens,
        'max_tokens': args.max_tokens,
        'use_alts': args.use_alts,
        'reverse_analogy': args.reverse_analogy,
        'debug': args.debug,
        'timing': args.timing,
        'add_print': args.print,
        'add_dots': args.add_dots,
        'add_format_demo': args.add_format_demo
    }

    test_model(args.dataset, args.output,
               model_params,
               **cot_params, **demonstration_params, **other_params)
