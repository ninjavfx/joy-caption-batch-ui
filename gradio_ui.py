import gradio as gr
import torch
from PIL import Image
import torchvision.transforms.functional as TVF
from transformers import AutoTokenizer, LlavaForConditionalGeneration

# 1) Load your JoyCaption model & tokenizer once at startup
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="bfloat16", device_map="auto"
).to(device)
model.eval()

# 2) List all your “Extra Options” as in the script’s comment block
EXTRA_OPTIONS = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, age, etc).",
    "Include information about the subject expression",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include what camera was likely used (aperture, shutter speed, ISO, etc).",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "Do NOT mention the backgound unless is relevant to the subject's pose",
    "Include the subjective aesthetic quality of the image (low to very high).",
    "Include composition style (leading lines, rule of thirds, symmetry).",
    "Do NOT mention any text that is in the image.",
    "Specify depth of field and if background is blurred or in focus.",
    "If applicable, mention artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]


def caption_image(
    image: Image.Image, base_prompt: str, extras: list[str], prepend_str: str
):
    # Build the full prompt
    full_prompt = base_prompt.strip()
    for e in extras or []:
        full_prompt += " " + e

    # Preprocess image
    img = image.convert("RGB").resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(img).float() / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5]).unsqueeze(0)

    # Move & cast to model's vision dtype/device
    vision_embed = model.vision_tower.vision_model.embeddings.patch_embedding.weight
    vision_dtype, vision_device = vision_embed.dtype, vision_embed.device
    pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)

    # Tokenize conversation
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": full_prompt},
    ]
    convo_str = tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(
        convo_str, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.5,
            top_k=10,
            top_p=0.9,
        )

    # Trim off prompt tokens & decode
    tokens = output_ids[0].tolist()
    eoh_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # Remove header
    while eoh_id in tokens:
        idx = tokens.index(eoh_id)
        tokens = tokens[idx + 1 :]
    # Truncate at end-of-turn
    if eot_id in tokens:
        tokens = tokens[: tokens.index(eot_id)]

    caption = tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    # Prepend custom string if provided
    final_caption = f"{prepend_str}{caption}" if prepend_str else caption

    # Return both the full prompt and the final caption
    return full_prompt, final_caption


# Build the Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Joy Caption — Single Image Demo")

        with gr.Row():
            # Left column: image input & extras
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload Image")
                with gr.Accordion(
                    "Extra options (append rules to your prompt)", open=False
                ):
                    extras_in = gr.CheckboxGroup(
                        choices=EXTRA_OPTIONS, label="Select any extra prompt-rules"
                    )

            # Right column: inputs & outputs
            with gr.Column():
                prompt_in = gr.Textbox(
                    label="Base Prompt", value="Write a short description of the image."
                )
                prepend_in = gr.Textbox(
                    label="Prepend String",
                    value="",
                    placeholder="Text to prepend to the generated caption",
                )
                generated_prompt = gr.Textbox(
                    label="Generated Prompt", interactive=False
                )
                out_caption = gr.Textbox(label="Generated Caption", interactive=False)
                run = gr.Button("Generate Caption")

        # Set up callback
        run.click(
            fn=caption_image,
            inputs=[img_in, prompt_in, extras_in, prepend_in],
            outputs=[generated_prompt, out_caption],
        )

        demo.launch()


if __name__ == "__main__":
    main()

