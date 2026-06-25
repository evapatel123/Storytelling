import gradio as gr
from huggingface_hub import InferenceClient
import numpy as np
from PIL import Image
import tensorflow as tf
tflite = tf.lite
import random

def get_quote():
    quotes = [
        "Believe you can and you're halfway there.",
        "Every day is a new beginning.",
        "Your potential is endless.",
        "The secret of getting ahead is getting started."
    ]
    return random.choice(quotes)

def update_quote():
    quote = get_quote()
    return f"""
    <div style="
        background-color: #FEFEFE; 
        border: 2px solid #D2E186; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center; 
        margin: 20px 0;
        box-shadow: 2px 2px 10px rgba(65, 81, 17, 0.1);
    ">
        <p style="color: #D2E186; font-size: 0.9rem; margin: 0; font-weight: bold;">✨ QUOTE OF THE DAY</p>
        <p style="color: #415111; font-size: 1.2rem; font-style: italic; margin: 10px 0 0 0;">
            "{quote}"
        </p>
    </div>
    """

interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

def predict_asl(frame):
    if frame is None:
        return "–"
    img = Image.fromarray(frame).resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    best_idx = int(np.argmax(predictions))
    return labels[best_idx]

spotify_embed = """
<iframe data-testid="embed-iframe" 
style="border-radius:12px" 
src="https://open.spotify.com/embed/playlist/3fZeYquijfHzPFtIMXzzUY?utm_source=generator" 
width="100%" 
height="352" 
frameBorder="0" 
allowfullscreen="" 
allow="autoplay; 
clipboard-write; 
encrypted-media; 
fullscreen; 
picture-in-picture" 
loading="lazy"></iframe>
"""

theme = gr.themes.Soft().set(
    body_background_fill="#e8ede8",
    block_background_fill="#f5f7f4cc",
    border_color_primary="#c8d4c8",
    button_primary_background_fill="#8aaa94",
    button_primary_background_fill_hover="#6a8a74",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#d4c49a",
    button_secondary_background_fill_hover="#b8a878",
    button_secondary_text_color="#ffffff",
    body_text_color="#4a5a50",
    block_title_text_color="#6a8a74",
    block_label_text_color="#8aaa94",
    input_background_fill="#f5f7f4",
    input_border_color="#c8d4c8",
    link_text_color="#8aaa94"
)



with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge_base = f.read()

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

SYSTEM_MESSAGES = {
    "wellness": (
        "You are a kind wellness chatbot. "
        "Give practical, supportive advice."
    ),
    "story": (
        "You are a creative storytelling assistant. "
        "Create engaging stories."
    )
}

def respond(message, history, mode):
    if mode is None:
        yield "Please select a mode first.", mode
        return

    messages = [{
        "role": "system",
        "content": SYSTEM_MESSAGES[mode] + "\n\n" + knowledge_base
    }]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = ""

    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        stream=True,
    ):
        delta = chunk.choices[0].delta
        token = delta.content if delta and hasattr(delta, "content") else ""
        response += token
        yield response, mode


def add_letter(letter, word):
    if letter and letter != "–":
        return word + letter
    return word

def add_space(word):
    return word + " "

def clear_word():
    return ""

def send_word(word):
    return word, ""


with gr.Blocks(theme=theme) as demo:


    mode_state = gr.State(None)

    gr.Image("Website Banner.png", show_label=False, container=False)

    gr.Markdown("# Your Wellness and Storytelling Companion")

    quote_box = gr.Markdown()
    demo.load(fn=update_quote, outputs=quote_box)
    
    gr.HTML("""
<div style="text-align:center; color:#c77dff; font-size:0.8rem; margin-bottom:6px;">
  How can I help you today?
</div>
""")

    with gr.Row():
        wellness_btn = gr.Button("🌿 Wellness", variant="primary")
        story_btn = gr.Button("📖 Story", variant="secondary")

    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[mode_state],
        additional_outputs=[mode_state]
    )

    wellness_btn.click(
        fn=lambda: ([{"role": "assistant", "content": "Wellness mode"}], "wellness"),
        outputs=[chatbot.chatbot, mode_state]
    )

    story_btn.click(
        fn=lambda: ([{"role": "assistant", "content": "Story mode"}], "story"),
        outputs=[chatbot.chatbot, mode_state]
    )

    gr.Markdown("### 🤟 ASL Input")
    with gr.Group():
        gr.Markdown("*Sign a letter in front of your camera, then click Add Letter to build a word and Send to chat.*")
        with gr.Row():
            with gr.Column(scale=1):
                asl_cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Camera",
                    height=300
                )
            with gr.Column(scale=1):
                asl_detected = gr.Textbox(label="Detected Letter", interactive=False, value="–")
                asl_word_box = gr.Textbox(label="Current Word", interactive=False, value="")
                with gr.Row():
                    asl_add_btn = gr.Button("Add Letter", variant="primary")
                    asl_space_btn = gr.Button("Space", variant="secondary")
                    asl_clear_btn = gr.Button("Clear", variant="secondary")
                asl_send_btn = gr.Button("Send to Chat ➤", variant="primary")

    word_state = gr.State("")

    asl_cam.stream(
        fn=predict_asl,
        inputs=[asl_cam],
        outputs=[asl_detected]
    )

    asl_add_btn.click(fn=add_letter, inputs=[asl_detected, word_state], outputs=[word_state]).then(
        fn=lambda w: w, inputs=[word_state], outputs=[asl_word_box]
    )
    asl_space_btn.click(fn=add_space, inputs=[word_state], outputs=[word_state]).then(
        fn=lambda w: w, inputs=[word_state], outputs=[asl_word_box]
    )
    asl_clear_btn.click(fn=clear_word, outputs=[word_state]).then(
        fn=lambda w: w, inputs=[word_state], outputs=[asl_word_box]
    )
    asl_send_btn.click(fn=send_word, inputs=[word_state], outputs=[chatbot.textbox, word_state]).then(
        fn=lambda w: w, inputs=[word_state], outputs=[asl_word_box]
    )

    gr.HTML("""
<div style="display:flex;align-items:center;gap:10px;margin:16px 0 8px;">
  <div style="flex:1;height:1px;background:linear-gradient(to right,transparent,#f3c4d7);"></div>
  <span style="color:#c77dff;font-size:0.85rem;">🎵Music</span>
  <div style="flex:1;height:1px;background:linear-gradient(to left,transparent,#f3c4d7);"></div>
</div>
""")
    gr.HTML(spotify_embed)

demo.launch(debug=True, allowed_paths=["."])