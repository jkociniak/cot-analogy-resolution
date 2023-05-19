import pandas as pd
import argparse
import pickle
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


def load_model(model):

    """

    Load the language model of your choosing. Input should be a string indicating model name

    Returns both model and tokenizer.

    """

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

    # Load config
    config = AutoConfig.from_pretrained(checkpoint)

    # Load model with empty weights to not take memory
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Let accelerate automatically create optimal device map
    model = load_checkpoint_and_dispatch(model, checkpoint2, device_map="auto",
                                         no_split_module_classes=no_split)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return model, tokenizer


def test_prompt(model, tokenizer, prompt, max_tokens=50, min_tokens=3):

    # Move prompt to gpu
    input = tokenizer(prompt, return_tensors="pt").to(0)

    # Generate output based on prompt
    output = model.generate(input["input_ids"], max_new_tokens=max_tokens, min_new_tokens=min_tokens)

    # Decode the output
    decoded_output = tokenizer.decode(output[0].tolist())

    return decoded_output


def test_model_scan(model, data_path, output_path, cot=False, demonstrations=False, n=4,
                    demo_cat=False, cot_sentence="Let's think step by step. \n", min=3, max=50):

    # Dictionary in which to store results
    result_dictionary = dict()

    model, tokenizer = load_model(model)

    df = pd.read_csv(data_path)

    for index, row in tqdm.tqdm(df.iterrows()):

        prompt1_dict = dict()
        prompt2_dict = dict()

        target = row["target"]
        source = row["source"]
        targ_word = row["targ_word"]
        src_word = row["src_word"]

        prompt1 = f"If {target} is like {source}, then {targ_word} is like"
        prompt2 = f"If {source} is like {target}, then {src_word} is like"

        if cot:
            prompt1 = cot_sentence + prompt1
            prompt2 = cot_sentence + prompt2
            n = 0

        elif demonstrations:
            if demo_cat:
                analogy_type = row["analogy_type"]
                demo_df = df[df["analogy_type"] == analogy_type].sample(n=n)
            else:
                demo_df = df[~df.index.isin([index])].sample(n=n)

            for _, demo_row in demo_df.iterrows():
                demo_first = f"If {demo_row['target']} is like {demo_row['source']} , "
                demo_second = f"then {demo_row['targ_word']} is like {demo_row['src_word']} . \n"
                prompt1 = demo_first + demo_second + prompt1
                prompt2 = demo_first + demo_second + prompt2
        else:
            n = 0

        prompt1_dict["prompt"] = prompt1
        prompt2_dict["prompt"] = prompt2

        correct1 = [src_word]

        if type(row["alternatives"]) == str:
            correct1 += row["alternatives"].split(", ")

        prompt1_dict["label"] = correct1
        prompt2_dict["label"] = targ_word

        output1 = test_prompt(model, tokenizer, prompt1, max, min)
        output2 = test_prompt(model, tokenizer, prompt2, max, min)

        prompt1_dict["output"] = output1
        prompt2_dict["output"] = output2

        for word in correct1:
            if word in output1.split("like ")[2 * (n+1)].split(".")[0]:
                prompt1_dict["pred"] = True
                break
            prompt1_dict["pred"] = False

        prompt2_dict["pred"] = True if targ_word in output2.split("like ")[2 * (n + 1)].split(".")[0] else False

        index_dict = {"normal":prompt1_dict, "reversed":prompt2_dict}

        result_dictionary[index] = index_dict

    with open(output_path, "wb") as f:
        pickle.dump(result_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"SAVED RESULTS AT: {output_path}")

    return


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", help="Use demonstrations", action="store_true")
    argParser.add_argument("-c", help="Use CoT reasoning", action="store_true")
    argParser.add_argument("-n", default=4, type=int)
    argParser.add_argument("-model", default="llama7b", help="What model to test", type=str)
    argParser.add_argument("-dataset", help="Path to dataset to use",
                           type=str, default="data/SCAN/SCAN_dataset.csv")
    argParser.add_argument("-cot_sent", help="Sentence to use to prompt cot",
                           type=str, default="Let's think step by step. \n")
    argParser.add_argument("-cat", help="Limit demonstrations to same category as query",
                           action="store_true")
    argParser.add_argument("-output", help="Path where results dict should be stored",
                           default="Results/baseline.pckl")
    argParser.add_argument("-min", help="Minimum number of new tokens to generate",
                           type=int, default=3)
    argParser.add_argument("-max", help="Maximum number of new tokens to generate",
                           type=int, default=50)

    args = argParser.parse_args()

    if "/SCAN/" in args.dataset:
        test_model_scan(args.model, args.dataset, args.output, cot=args.c, demonstrations=args.d,
                        n=args.n, demo_cat=args.cat, cot_sentence=args.cot_sent, min=args.min, max=args.max)
