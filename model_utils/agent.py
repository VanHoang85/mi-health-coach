import os
import sys
import time
import litellm

# import openai
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage


class Agent:
    def __init__(self,
                 args,
                 role="",
                 system_prompt="",
                 model_type=None):
        self.args = args
        self.role = role
        self.system_prompt = system_prompt

        self.model_type = model_type
        self.load_api_key()
        # self.llm = self.load_model()

    def send_message(self,
                     prompt: str,
                     task: str,
                     system_prompt: str = "",
                     stream: bool = False):
        messages = [{"role": "user", "content": prompt}]
        if len(system_prompt) > 0:
            messages.insert(0, {"role": "system", "content": system_prompt})

        num_retry = 0
        response = None
        while not response:  # to make sure not return NoneType
            response = litellm.completion(model=self.model_type,
                                          messages=messages,
                                          temperature=self.args.temperature,
                                          top_p=self.args.top_p,
                                          # num_retries=self.args.num_retries,
                                          # stream_timeout=self.args.stream_timeout if "stream_timeout" in self.args else None,
                                          timeout=self.args.request_timeout if "request_timeout" in self.args else None,  # raise Timeout error if call takes longer than 10s
                                          fallbacks=self.args.fallback_models if "fallback_models" in self.args else None,
                                          stream=stream
                                          )  # may add fallback
            if not stream:
                response = response.choices[0].message.content
        return response

    def load_api_key(self):
        os.environ["OPENAI_API_KEY"] = self.args.openai_key
        os.environ['GROQ_API_KEY'] = self.args.groq_key

    def update_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
