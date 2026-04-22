import glob as glob_module
import os
import threading

import gradio as gr
import torch

from generate_samples import TextGenerator, load_model
from TokenizerFactory import TokenizerFactory

TOKENIZER_PATH = "tokenizer/trained_tokenizer/tokenizer.json"

_models: dict = {}
_tokenizer = None
_lock = threading.Lock()


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                _tokenizer = TokenizerFactory.create("huggingface", TOKENIZER_PATH)
    return _tokenizer


def _get_model(model_path: str):
    if model_path not in _models:
        with _lock:
            if model_path not in _models:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _models[model_path] = load_model(model_path, device)
    return _models[model_path]


def _available_models() -> list[str]:
    found = [f for f in glob_module.glob("*.pt")]
    if os.path.isdir("models"):
        found += glob_module.glob("models/*.pt")
    return sorted(found) or ["No .pt files found"]


def generate(model_name, prompt, method, max_new_tokens, temperature, k, p):
    if not prompt.strip():
        return "Please enter a sentence starter."
    if model_name == "No .pt files found":
        return "No model file found. Upload a trained .pt checkpoint to the Space."
    try:
        model = _get_model(model_name)
        tokenizer = _get_tokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = TextGenerator(model, tokenizer, device=device)

        if method == "Greedy":
            return gen.greedy_decode(prompt, max_new_tokens=max_new_tokens)
        elif method == "Top-K":
            return gen.top_k_decode(prompt, max_new_tokens=max_new_tokens, k=k, temperature=temperature)
        else:
            return gen.nucleus_decode(prompt, max_new_tokens=max_new_tokens, p=p, temperature=temperature)
    except Exception as exc:
        return f"Error: {exc}"


models = _available_models()

with gr.Blocks(title="GPT-2 Like Model") as demo:
    gr.Markdown("# GPT-2 Like Model\nSelect a model, enter a sentence starter, and generate text.")

    with gr.Row():
        model_dd = gr.Dropdown(choices=models, value=models[0], label="Model")
        method_dd = gr.Dropdown(
            choices=["Nucleus (top-p)", "Top-K", "Greedy"],
            value="Nucleus (top-p)",
            label="Decoding Method",
        )

    prompt_box = gr.Textbox(label="Sentence Starter", placeholder="Enter sentence starter here")

    with gr.Row():
        max_tokens = gr.Slider(1, 500, value=60, step=1, label="Max New Tokens")
        temperature = gr.Slider(0.1, 2.0, value=0.75, step=0.05, label="Temperature")

    with gr.Row():
        k_val = gr.Slider(1, 200, value=30, step=1, label="Top-K value")
        p_val = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="Nucleus p")

    generate_btn = gr.Button("Generate", variant="primary")
    output_box = gr.Textbox(label="Generated Text", lines=6)

    def _map_method(method_label):
        return {"Nucleus (top-p)": "Nucleus", "Top-K": "Top-K", "Greedy": "Greedy"}[method_label]

    generate_btn.click(
        fn=lambda m, pr, me, mt, t, k, p: generate(m, pr, _map_method(me), mt, t, k, p),
        inputs=[model_dd, prompt_box, method_dd, max_tokens, temperature, k_val, p_val],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
