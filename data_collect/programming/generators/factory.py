from .py_generate import PyGenerator
from .model import Llama, ModelBase, GPT4, GPT35, Phi3, GPT3512, GPT4o, GPT4o_mini, GPT5, DeepSeek_r1, Qwen3

def model_factory(model_name: str, port: str = "", key: str = "") -> ModelBase:
    if  model_name=="gpt-4":
        return GPT4(key)
    elif model_name == "gpt-4o":
        return GPT4o(key)
    elif model_name == "gpt-5":
        return GPT5(key)
    elif model_name == "deepseek-r1":
        return DeepSeek_r1(key)
    elif model_name == "gpt-4o-mini":
        return GPT4o_mini(key)
    elif model_name == "gpt-3.5-turbo-0613":
        return GPT35(key)
    elif model_name == "gpt-3.5-turbo-0125":
        return GPT3512(key)
    elif model_name == "llama3":
        return Llama(port)
    elif model_name == "phi-3":
        return Phi3(port)
    elif model_name == "Qwen3":
        return Qwen3("8000")
    else:
        raise ValueError(f"Invalid model name: {model_name}")
