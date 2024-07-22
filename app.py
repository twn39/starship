import gradio as gr
from gradio import FileData
from langchain_core.messages import HumanMessage, AIMessage
from llm import DeepSeekLLM, OpenRouterLLM, TongYiLLM
from config import settings
import base64
from PIL import Image
import io


deep_seek_llm = DeepSeekLLM(api_key=settings.deep_seek_api_key)
open_router_llm = OpenRouterLLM(api_key=settings.open_router_api_key)
tongyi_llm = TongYiLLM(api_key=settings.tongyi_api_key)


def init_chat():
    return deep_seek_llm.get_chat_engine()


def predict(message, history, chat):
    file_len = len(message.files)
    if chat is None:
        chat = init_chat()
    history_messages = []
    for human, assistant in history:
        history_messages.append(HumanMessage(content=human))
        if assistant is not None:
            history_messages.append(AIMessage(content=assistant))

    if file_len == 0:
        history_messages.append(HumanMessage(content=message.text))
    else:
        file = message.files[0]
        with Image.open(file.path) as img:
            buffer = io.BytesIO()
            img = img.convert('RGB')
            img.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            history_messages.append(HumanMessage(content=[
                {"type": "text", "text": message.text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]))

    response_message = ''
    for chunk in chat.stream(history_messages):
        response_message = response_message + chunk.content
        yield response_message


def update_chat(_provider: str, _chat, _model: str, _temperature: float, _max_tokens: int):
    print('?????', _provider, _chat, _model, _temperature, _max_tokens)
    if _provider == 'DeepSeek':
        _chat = deep_seek_llm.get_chat_engine(model=_model, temperature=_temperature, max_tokens=_max_tokens)
    if _provider == 'OpenRouter':
        _chat = open_router_llm.get_chat_engine(model=_model, temperature=_temperature, max_tokens=_max_tokens)
    if _provider == 'Tongyi':
        _chat = tongyi_llm.get_chat_engine(model=_model, temperature=_temperature, max_tokens=_max_tokens)
    return _chat


with gr.Blocks() as app:
    with gr.Tab('聊天'):
        chat_engine = gr.State(value=None)
        with gr.Row():
            with gr.Column(scale=2, min_width=600):
                chatbot = gr.ChatInterface(
                    predict,
                    multimodal=True,
                    chatbot=gr.Chatbot(elem_id="chatbot", height=600, show_share_button=False),
                    textbox=gr.MultimodalTextbox(lines=1),
                    additional_inputs=[chat_engine]
                )
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion('参数设置', open=True):
                    with gr.Column():
                        provider = gr.Dropdown(
                            label='模型厂商',
                            choices=['DeepSeek', 'OpenRouter', 'Tongyi'],
                            value='DeepSeek',
                            info='不同模型厂商参数，效果和价格略有不同，请先设置好对应模型厂商的 API Key。',
                        )

                    @gr.render(inputs=provider)
                    def show_model_config_panel(_provider):
                        _support_llm = deep_seek_llm
                        if _provider == 'OpenRouter':
                            _support_llm = open_router_llm
                        if _provider == 'Tongyi':
                            _support_llm = tongyi_llm
                        with gr.Column():
                            model = gr.Dropdown(
                                label='模型',
                                choices=_support_llm.support_models,
                                value=_support_llm.default_model
                            )
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=_support_llm.default_temperature,
                                label="Temperature",
                                key="temperature",
                            )
                            max_tokens = gr.Slider(
                                minimum=1024,
                                maximum=_support_llm.default_max_tokens,
                                step=128,
                                value=_support_llm.default_max_tokens,
                                label="Max Tokens",
                                key="max_tokens",
                            )
                        model.change(
                            fn=update_chat,
                            inputs=[provider, chat_engine, model, temperature, max_tokens],
                            outputs=[chat_engine],
                        )
                        temperature.change(
                            fn=update_chat,
                            inputs=[provider, chat_engine, model, temperature, max_tokens],
                            outputs=[chat_engine],
                        )
                        max_tokens.change(
                            fn=update_chat,
                            inputs=[provider, chat_engine, model, temperature, max_tokens],
                            outputs=[chat_engine],
                        )


app.launch(debug=settings.debug, show_api=False)
