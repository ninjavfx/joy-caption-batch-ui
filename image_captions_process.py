import os
import argparse
import gradio as gr
import glob
from PIL import Image


def get_image_and_caption_files(directory):
    image_files = []
    for ext in ["jpg", "jpeg", "png"]:
        image_files += glob.glob(os.path.join(directory, f"*.{ext}"))
        image_files += glob.glob(os.path.join(directory, f"*.{ext.upper()}"))
    image_files = sorted(image_files)
    caption_files = []
    for img_path in image_files:
        base = os.path.splitext(img_path)[0]
        caption_files.append(f"{base}.txt")
    return image_files, caption_files


def read_caption(path):
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read().strip()
    return ""


def save_caption(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_and_resize_image(path):
    try:
        img = Image.open(path)
        w, h = img.size
        return img.resize((w // 2, h // 2))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process image captions with Gradio UI"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory of images + caption files"
    )
    args = parser.parse_args()

    image_files, caption_files = get_image_and_caption_files(args.input_dir)
    if not image_files:
        print(f"No images in {args.input_dir}")
        return

    initial_captions = [read_caption(p) for p in caption_files]

    with gr.Blocks() as app:
        # Search & Replace row
        with gr.Row():
            search_txt = gr.Textbox(label="Search Text")
            replace_txt = gr.Textbox(label="Replace Text")
            search_btn = gr.Button("Search & Replace All")

        # Append row
        with gr.Row():
            append_txt = gr.Textbox(label="Append to All Captions")
            append_btn = gr.Button("Append to All")
            save_btn = gr.Button("Save All Changes", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        # Dynamically create image + caption textbox pairs
        caption_boxes = []
        for img_path, cap in zip(image_files, initial_captions):
            with gr.Row():
                gr.Image(
                    value=load_and_resize_image(img_path),
                    label=os.path.basename(img_path),
                    interactive=False,
                )
                cb = gr.Textbox(
                    value=cap,
                    label=f"Caption for {os.path.basename(img_path)}",
                    lines=5,
                )
                caption_boxes.append(cb)

        # ====== CALLBACKS ======

        def search_replace_all(search, replace, *texts):
            new_texts = [
                t.replace(search, replace) if search in t else t for t in texts
            ]
            status_msg = f"Replaced '{search}' with '{replace}' in all captions"
            return (*new_texts, status_msg)

        def append_to_all(*texts, append_str):
            new_texts = [t + append_str for t in texts]
            status_msg = f"Appended '{append_str}' to all captions"
            return (*new_texts, status_msg)

        def save_all_changes(*texts):
            for t, path in zip(texts, caption_files):
                save_caption(path, t)
            return f"Successfully saved {len(texts)} caption files"

        # Wire buttons to caption_boxes directly
        search_btn.click(
            fn=search_replace_all,
            inputs=[search_txt, replace_txt, *caption_boxes],
            outputs=[*caption_boxes, status],
        )

        append_btn.click(
            fn=append_to_all,
            inputs=[*caption_boxes, append_txt],
            outputs=[*caption_boxes, status],
        )

        save_btn.click(
            fn=save_all_changes,
            inputs=[*caption_boxes],
            outputs=[status],
        )

    app.launch()


if __name__ == "__main__":
    main()

