from ray.serve.gradio_integrations import GradioServer
from ray import serve
import gradio as gr

from transformers import pipeline

example_input = "SO"


def gradio_summarizer_builder2():
    generator = pipeline("text-generation", model="Fan2/gpt2-confluence")

    def model(text):
        summary_list = generator(text)
        summary = summary_list[0]["summary_text"]
        return summary

    # gr.Interface.load("models/Fan2/gpt2-confluence").launch()

    return gr.Interface(
        fn=model,
        inputs=[gr.inputs.Textbox(
            default=example_input, label="Input confluence page name")],
        outputs=[gr.outputs.Textbox(label="Hugging face model checkpoint")],
    )


app = GradioServer.options(num_replicas=1, ray_actor_options={"num_cpus": 4}).bind(
    gradio_summarizer_builder2
)

serve.run(app)
