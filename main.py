import gradio as gr
from transformers import MarianMTModel, MarianTokenizer


# Define the translation function
def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text

    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)[0]
    translated_text = tokenizer.decode(output, skip_special_tokens=True)
    return translated_text


# Create the Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.inputs.Textbox(label="Text"),
        gr.inputs.Radio(["en", "fr", "es", "de"], label="Source Language"),
        gr.inputs.Radio(["en", "fr", "es", "de"], label="Target Language")
    ],
    outputs=gr.outputs.Textbox(label="Translation"),
    title="Multilingual Translator",
    description="Translate text between different languages [en: English | fr: French | es: Spanish | de: German]",
)

# Launch the interface
iface.launch(share=True)
