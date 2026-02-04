import os

# 必须在导入 transformers 或 vllm 之前设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Union, Optional, Literal
import dataclasses
import os
# from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI
from transformers import GPT2Tokenizer, AutoTokenizer
import random
import time

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.tokenize(l))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.tokenize(messages[0].content))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.tokenize(msg.content))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2,
                      num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None,
                 temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str, key: str = "", base_url: str = None):
        self.name = model_name
        self.is_chat = True
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # 构建 OpenAI client 参数，支持 base_url
        client_kwargs = {}
        if key != "":
            client_kwargs["api_key"] = key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def gpt_chat(
            self,
            messages,
            stop: List[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.0,
            num_comps=1,
    ) -> Union[List[str], str]:
        try:
            new_messages = change_messages(self.tokenizer, messages, 24000)
            messages = new_messages
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[dataclasses.asdict(message) for message in messages],
                temperature=temperature,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=num_comps,
                stop=stop
            )
        except Exception as e:
            print("GPT Error:", str(e))
            if "context_length_exceeded" in str(e):
                messages = change_messages(self.tokenizer, messages, 24000)
                print("AFTER CHANGE MESSAGE LEN:", len(messages))
                print(messages)
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[dataclasses.asdict(message) for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                )
            else:
                assert False, "GPT API error: " + str(e)
        if num_comps == 1:
            return response.choices[0].message.content  # type: ignore
        return [choice.message.content for choice in response.choices]  # type: ignore

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = 1024,
                      temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)
        return res

# gpt-4o-2024-05-13
class GPT4o(GPTChat):
    def __init__(self, key):
        super().__init__("gpt-5.2-2025-12-11", key, base_url="https://api.holdai.top/v1")


class GPT4o_mini(GPTChat):
    def __init__(self, key):
        super().__init__("gpt-4o-mini-2024-07-18", key, base_url="https://api.holdai.top/v1")


class GPT4(GPTChat):
    def __init__(self, key):
        super().__init__("gpt-4-1106-preview", key, base_url="https://api.holdai.top/v1")


class GPT35(GPTChat):
    def __init__(self, key):
        super().__init__("gpt-3.5-turbo-0613", key, base_url="https://api.holdai.top/v1")


class GPT3512(GPTChat):
    def __init__(self, key):
        super().__init__("gpt-3.5-turbo-0125", key, base_url="https://api.holdai.top/v1")


# 新增：GPT-5 类 (教师模型)，支持自定义 base_url
class GPT5(GPTChat):
    def __init__(self, key, base_url=None):
        # 假设模型名称为 "gpt-5"，如果中转站映射的名称不同，请在此修改
        super().__init__("gpt-5-mini", key, base_url="https://api.holdai.top/v1/chat/completions")

class DeepSeek_r1(GPTChat):
    def __init__(self, key, base_url=None):
        super().__init__("deepseek-r1", key, base_url="https://api.holdai.top/v1/chat/completions")
class VLLMModelBase(ModelBase):
    """
    Base for huggingface chat models using vLLM server
    """

    def __init__(self, model, port="8000"):
        super().__init__(model)
        self.model = model
        self.is_chat = True
        # vLLM 启动后会提供一个兼容 OpenAI 格式的 API
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        try:
            local_model_path = "/root/autodl-tmp/attentioncode/qwen2.5-coder-1.5B"
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                # 添加这一行来修复警告
                fix_mistral_regex=True
            )
            print(f"tokenizer length:{len(self.tokenizer)}")
        except:
            print("Warning: Qwen tokenizer not found, falling back to GPT2 for length calc.")
        self.max_length = 8000


    def vllm_chat(
            self,
            prompt: str,
            stop: List[str] = [""],
            max_tokens: int = 18000,
            temperature: float = 0.0,
            num_comps=1,
    ) -> Union[List[str], str]:

        max_length = self.max_length
        Internal_Server_Error = 0
        request_timeout = 0
        timeout = 1800

        # 清洗 stop
        final_stop = None
        if stop:
            cleaned_stop = [s for s in stop if s and len(str(s)) > 0]
            if cleaned_stop:
                final_stop = cleaned_stop

        while True:
            # 执行处理函数
            prompt_processed = change_messages(self.tokenizer, prompt, max_length)

            # 转换格式
            try:
                final_messages = [dataclasses.asdict(message) for message in prompt_processed]
                # print(f"final_messages:{final_messages}")
            except TypeError:
                final_messages = prompt_processed
            try:
                responses = self.vllm_client.chat.completions.create(
                    model="qwen-lora",
                    messages=[dataclasses.asdict(message) for message in prompt_processed],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1,
                    stop=final_stop,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                    timeout=timeout,
                )
            except Exception as e:
                print("VLLM Error:", str(e))
                if "Internal Server Error" in str(e):
                    if Internal_Server_Error <= 5:
                        print("try again Server Error")
                        num = round(random.uniform(0, 2), 4)
                        time.sleep(num + Internal_Server_Error)
                        Internal_Server_Error += 1
                        continue
                    else:
                        print("try 5 times Internal Server Error")
                        assert False, "VLLM API error: " + str(e)
                if "Request timed out" in str(e):
                    if request_timeout <= 5:
                        max_tokens = 8192 - 1024
                        print("try again Request timed out")
                        num = round(random.uniform(0, 2), 4)
                        time.sleep(num + request_timeout)
                        request_timeout += 1
                        continue
                    else:
                        print("try 5 Request timed out")
                        assert False, "VLLM API error: " + str(e)
                if "maximum context length" in str(e):
                    max_length -= 2000
                else:
                    assert False, "VLLM API error: " + str(e)
            else:
                break
        if num_comps == 1:
            return responses.choices[0].message.content  # type: ignore
        return [responses.choices[0].message.content for response in responses]  # type: ignore

    def generate_completion(self, messages: str, stop: List[str] = [""], max_tokens: int = 8192,
                            temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        ret = self.vllm_chat(messages, stop, max_tokens, temperature, num_comps)
        return ret

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = 8192,
                      temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        res = self.generate_completion(messages, stop, max_tokens, temperature, num_comps)
        return res

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += message.content + "\n"
            if i == len(messages) - 1:
                prompt += "\n"
        return prompt

    def extract_output(self, output: str) -> str:
        return output


class Phi3(VLLMModelBase):
    def __init__(self, port=""):
        super().__init__("phi-3", port)


class Llama(VLLMModelBase):
    def __init__(self, port=""):
        super().__init__("Llama3.0", port)


class Qwen3(VLLMModelBase):
    def __init__(self, port: str):
        # 修正：将传入的 port 传递给父类，而不是写死 "8000"
        super().__init__("qwen-lora", port=port if port else "8000")

    def _remove_thinking_process(self, text: str) -> str:
        """
        内部辅助函数：移除 <think>...</think> 及其之前的内容
        """
        # 检查是否存在思考结束标签
        if "</think>" in text:
            # 使用 split 分割，取最后一部分（即 </think> 之后的内容）
            # Qwen3 的标准格式是：<think>内容</think>正式回复
            formal_response = text.split("</think>")[-1]
            return formal_response.strip()  # 去除首尾可能的换行符

        # 如果没有标签（可能是因为 max_tokens 不够导致截断，或者模型未思考），直接返回原文本
        return text

    def vllm_chat(self, *args, **kwargs) -> Union[List[str], str]:
        # 1. 调用父类方法获取原始回复（包含思考过程）
        raw_response = super().vllm_chat(*args, **kwargs)

        # 2. 根据返回类型（单个字符串 或 字符串列表）分别处理
        if isinstance(raw_response, list):
            return [self._remove_thinking_process(r) for r in raw_response]
        else:
            return self._remove_thinking_process(raw_response)

# 新增：Qwen2.5-Coder 类 (学生模型)
class Qwen25Coder(VLLMModelBase):
    def __init__(self, port="8000"):
        # 这里的模型名称 "Qwen/Qwen2.5-Coder-7B-Instruct" 必须与 vLLM 启动参数一致
        super().__init__("Qwen/Qwen2.5-Coder-7B-Instruct", port)