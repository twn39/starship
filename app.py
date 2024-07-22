import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from llm import DeepSeekLLM, OpenRouterLLM, TongYiLLM
from config import settings


deep_seek_llm = DeepSeekLLM(api_key=settings.deep_seek_api_key)
open_router_llm = OpenRouterLLM(api_key=settings.open_router_api_key)
tongyi_llm = TongYiLLM(api_key=settings.tongyi_api_key)


def init_chat():
    return deep_seek_llm.get_chat_engine()


def predict(message, history, chat):
    if chat is None:
        chat = init_chat()
    history_messages = []
    for human, assistant in history:
        history_messages.append(HumanMessage(content=human))
        history_messages.append(AIMessage(content=assistant))
    history_messages.append(HumanMessage(content=message.text))

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
                        if _provider == 'DeepSeek':
                            with gr.Column():
                                model = gr.Dropdown(
                                    label='模型',
                                    choices=deep_seek_llm.support_models,
                                    value=deep_seek_llm.default_model
                                )
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    value=deep_seek_llm.default_temperature,
                                    label="Temperature",
                                    key="temperature",
                                )
                                max_tokens = gr.Slider(
                                    minimum=1024,
                                    maximum=1024 * 20,
                                    step=128,
                                    value=deep_seek_llm.default_max_tokens,
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
                        if _provider == 'OpenRouter':
                            with gr.Column():
                                model = gr.Dropdown(
                                    label='模型',
                                    choices=open_router_llm.support_models,
                                    value=open_router_llm.default_model
                                )
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    value=open_router_llm.default_temperature,
                                    label="Temperature",
                                    key="temperature",
                                )
                                max_tokens = gr.Slider(
                                    minimum=1024,
                                    maximum=1024 * 20,
                                    step=128,
                                    value=open_router_llm.default_max_tokens,
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
                        if _provider == 'Tongyi':
                            with gr.Column():
                                model = gr.Dropdown(
                                    label='模型',
                                    choices=tongyi_llm.support_models,
                                    value=tongyi_llm.default_model
                                )
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    value=tongyi_llm.default_temperature,
                                    label="Temperature",
                                    key="temperature",
                                )
                                max_tokens = gr.Slider(
                                    minimum=1000,
                                    maximum=2000,
                                    step=100,
                                    value=tongyi_llm.default_max_tokens,
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
