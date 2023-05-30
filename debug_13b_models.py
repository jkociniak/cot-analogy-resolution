import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights


def test_prompt(model, tokenizer, prompt, max_tokens=50, min_tokens=3):
    # Move prompt to gpu
    print(f'Cuda available: {torch.cuda.is_available()}')
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


if __name__ == "__main__":
    # LLAMA 13B
    cfg_ckpt = '/home/lcur1680/uva-atcs-project/cot-analogy-resolution/pretrained/7B'
    model_params = {
        "cfg_ckpt": cfg_ckpt,
        "weights_ckpt": cfg_ckpt,
        "no_split": ["LlamaDecoderLayer"]
    }

    #start_time = time.time()
    model, tokenizer = load_model(**model_params)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Loading model took {elapsed_time} seconds")

    prompt = 'If atom is like solar system, then nucleus is like'
    #print(f'Prompt: {prompt}')

    #start_time = time.time()
    output = test_prompt(model, tokenizer, prompt, 3, 200)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Inference took {elapsed_time} seconds")

    #print(f'Output: {output}')

