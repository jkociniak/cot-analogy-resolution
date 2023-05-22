import pandas as pd
import argparse
import pickle
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


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

    # Generate output based on prompt
    output = model.generate(input["input_ids"], max_new_tokens=max_tokens, min_new_tokens=min_tokens)

    # Decode the output
    decoded_output = tokenizer.decode(output[0].tolist())

    return decoded_output


def preprocess_prompt(example1, example2, target1, cot, cot_prompt, cot_format):
    default_prompt = f"If {example1} is like {example2}, then {target1} is like"
    if not cot:
        return default_prompt

    assert cot_format in ['kojima', 'naive']
    if cot_format == 'naive':
        return cot_prompt + default_prompt
    elif cot_format == 'kojima':
        prompt = f"""Q: If {example1} is like {example2}, then what {target1} is like?
                     A: """
        return prompt + cot_prompt


def add_demonstrations(demos_df, filter_demos, n_demos, category):
    if filter_demos:
        demo_df = demos_df[demos_df["analogy_type"] == category].sample(n=n_demos)
    else:
        demo_df = demos_df.sample(n=n_demos)

    for _, demo_row in demo_df.iterrows():
        demo_first = f"If {demo_row['target']} is like {demo_row['source']} , "
        demo_second = f"then {demo_row['targ_word']} is like {demo_row['src_word']} . \n"
        prompt = demo_first + demo_second + prompt


def evaluate_answer(answer, target, alternatives):
    eval_summary = dict()
    eval_summary['label'] = target
    eval_summary["alternatives"] = alternatives
    targets = [target] + alternatives

    correct = False
    for word in targets:
        if word in answer:
            correct = True
            break

    eval_summary["correct"] = correct
    return eval_summary


def test_model(data_path, output_path,
               model_params,
               cot=False, cot_prompt="Let's think step by step. \n", cot_format='append',
               add_demos=False, n_demos=4, filter_demos=True,
               min_tokens=3, max_tokens=50, use_alts=True, reverse_analogy=False):

    # Load model and dataset
    model, tokenizer = load_model(**model_params)
    df = pd.read_csv(data_path)
    df = df.iloc[:5]

    results = dict()
    for index, row in tqdm.tqdm(df.iterrows()):
        example1 = row["target"]
        example2 = row["source"]
        target1 = row["targ_word"]
        target2 = row["src_word"]

        if reverse_analogy:
            example1, example2 = example2, example1
            target1, target2 = target2, target1

        prompt = preprocess_prompt(example1, example2, target1, cot, cot_prompt, cot_format)

        analogy_type = row["analogy_type"]
        if add_demos:
            assert n_demos > 0
            possible_demos = df[~df.index.isin([index])]
            add_demonstrations(possible_demos, filter_demos, n_demos, analogy_type)
        else:
            n_demos = 0

        output = test_prompt(model, tokenizer, prompt, max_tokens, min_tokens)

        results_summary = {"prompt": prompt,
                           "category": analogy_type,
                           "output": output}

        alternatives = None
        if use_alts:
            assert type(row["alternatives"]) == str
            alternatives = row["alternatives"].split(", ")

        answer = output.split("like ")[2 * (n_demos + 1)].split(".")[0]
        eval_summary = evaluate_answer(answer, target2, alternatives)
        results_summary.update(eval_summary)
        results[index] = results_summary

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
    argParser.add_argument("--model", default="llama7b", choices=['llama7b', 'gptj'], help="Which model to test", type=str)
    argParser.add_argument("--cfg_ckpt", help="Path to config checkpoint", type=str)
    argParser.add_argument("--weights_ckpt", help="Path to weights checkpoint", type=str)

    # demonstration params
    argParser.add_argument("--add_demos", help="Use demonstrations", action="store_true")
    argParser.add_argument("--n_demos", default=4, type=int)
    argParser.add_argument("--filter_demos", help="Limit demonstrations to same category as query",
                           action="store_true")

    # chain-of-thought params
    argParser.add_argument("--cot", help="Use CoT reasoning", action="store_true")
    argParser.add_argument("--cot_prompt", help="Sentence to use to prompt cot",
                           type=str, default="Let's think step by step. \n")
    argParser.add_argument("--cot_format", help="How to join cot prompt with the input prompt",
                           type=str, choices=['kojima', 'naive'], default='naive')


    # other params
    argParser.add_argument("--min_tokens", help="Minimum number of new tokens to generate",
                           type=int, default=3)
    argParser.add_argument("--max_tokens", help="Maximum number of new tokens to generate",
                           type=int, default=50)
    argParser.add_argument("--use_alts", help="use the alternatives or not", action="store_true")
    argParser.add_argument("--reverse_analogy", help="Use function to analyse difference on reversed prompts on SCAN",
                           action="store_true")

    args = argParser.parse_args()

    model_name = args.model.lower()
    if model_name == "llama7b":
        # Set the specifics for the LLama model
        checkpoint = "LLama/llama/converted_models/7B"
        checkpoint2 = checkpoint
        no_split = ["LlamaDecoderLayer"]
    elif model_name == "gptj":
        # Set specifics for gptj model
        checkpoint = "EleutherAI/gpt-j-6B"
        checkpoint2 = "sharded-gpt-j-6B"
        no_split = ["GPTJBlock"]
    else:
        raise ValueError("Model not supported")

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
        'reverse_analogy': args.reverse_analogy
    }

    test_model(args.dataset, args.output,
               model_params,
               **cot_params, **demonstration_params, **other_params)
