import os
from enum import Enum
import argparse

class LLM_Models(Enum):
    OPENAI_GPT4_MINI = 0
    OPENAI_GTP4O = 1
    ANTHROPIC_CLAUDE3_HAIKU = 2
    LLAMA_LOCAL = 3
    GOOGLE_GEMINI = 4


class LLM_Wrapper:
    """
    This is a wrapper for some popular LLMs currently available. It is a simple
    code that adapts the input and output for four different APIs:

    - OpenAI GPT-4 Mini (OPENAI_GPT4_MINI): the famous OpenAI GPT-4o-mini model.
    - Anthropic Claude 3 Haiky (ANTHROPIC_CLAUDE3_HAIKU): Claude Haiku model.
    - Llama (LLAMA_LOCAL): Meta's model running locally, interfaced by ollama.
    - Google Gemini (GOOGLE_GEMINI): Google's Gemini model.

    You're expected to have set environment variables for the API keys:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_GEMINI_API_KEY

    1. Create the model like this:
    
        llm = LLM_wrapper(LLM_Models.OPENAI_GPT4_MINI, max_tokens=1024)
    
    Notice that llama implementation does not provide a max_tokens parameter, 
    so it will be ignored.

    2. Start sending messages:

        assistant_response = llm.send_message("Hello, I'm a user", "user")

    For each message it would be send the whole conversation history to the API.

    3. Optionally, you can print the trace of the conversation:

        llm.print_trace()

    Have fun and remember that LLMs lie! ;)
    """
    def __init__(self, llm_model: LLM_Models, max_tokens: int = 1024, temperature: float = 0.5):
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if llm_model == LLM_Models.OPENAI_GPT4_MINI or llm_model == LLM_Models.OPENAI_GTP4O:
            self.init_openai(llm_model)
        elif llm_model == LLM_Models.ANTHROPIC_CLAUDE3_HAIKU:
            self.init_anthropic()
        elif llm_model == LLM_Models.LLAMA_LOCAL:
            self.init_ollama()
        elif llm_model == LLM_Models.GOOGLE_GEMINI:
            self.init_google_gemini()

        self.system_message = ""
        self.user_messages = []
        
        self.trace = ""
        self.message_history = []

    def get_model_name(self):
        return str(self.llm_model)


    def init_openai(self, llm_model):
        from openai import OpenAI

        self.client = OpenAI(
                    project='proj_XCapd6U3grhxrToccuW6UdNO',
                    api_key=os.environ.get('OPENAI_API_KEY')
        )
        self.model_name = "gpt-4o-mini" if llm_model == LLM_Models.OPENAI_GPT4_MINI else "gpt-4o"
        self.chat_function = self.client.chat.completions.create

        self.response_content_fun = lambda response: response.choices[0].message.content

    def init_anthropic(self):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        # self.model_name = "claude-3-5-sonnet-20240620"
        self.model_name = "claude-3-haiku-20240307"

        self.chat_function = self.client.messages.create

        self.response_content_fun = lambda response: response.content[0].text

    def init_ollama(self):
        import ollama

        self.model_name = "llama3.1"
        self.chat_function = ollama.chat
        self.max_tokens = None

        self.response_content_fun = lambda response: response["message"]["content"]

    def init_google_gemini(self):
        import google.generativeai as genai

        self.model_name = "gemini-1.5-flash"
        genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

        self.client = genai.GenerativeModel()
        self.chat = self.client.start_chat()
        self.chat_function = self.chat.send_message

        self.response_content_fun = lambda response: response.candidates[0].content.parts[0].text

    def print_trace(self):
        print("\nMESSAGES TRACE")
        print(self.trace)

    def add_to_trace(self, message, role):
        self.trace += f"{role}: {message}\n\n"

    def call_api_chat_w_history(self):
        # print(f"Calling api with the following messages\n{self.message_history}")
        if self.llm_model != LLM_Models.GOOGLE_GEMINI:
            chat_kwargs = {
                "model": self.model_name,
                "messages": self.message_history,
            }
            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens
            chat_kwargs["temperature"] = self.temperature
            res_api = self.chat_function(**chat_kwargs)
        else:
            last_user_message = self.message_history[-1]["parts"]
            res_api = self.chat_function(last_user_message)
        
        self.add_to_trace(">> API called.", str(self.llm_model))
        return res_api
    
    def add_message_history(self, message: str, role: str):
        content_field = "content" if self.llm_model != LLM_Models.GOOGLE_GEMINI else "parts"
        self.message_history.append({"role": role, content_field: message})
        self.add_to_trace(message, role)
    
    def clear_history(self):
        self.message_history = []

    def send_message(self, message: str, role: str = "user"):
        if role == "system" \
                and (self.llm_model == LLM_Models.ANTHROPIC_CLAUDE3_HAIKU or self.llm_model == LLM_Models.GOOGLE_GEMINI):
            print(f"WARNING: {self.llm_model} does not support system messages, changing to user message.")
            self.add_message_history(message, "user")
        else:
            self.add_message_history(message, role)

        llm_res = self.call_api_chat_w_history()
        assistant_response = self.response_content_fun(llm_res)

        assistant_role = "assistant" if self.llm_model != LLM_Models.GOOGLE_GEMINI else "model"
        self.add_message_history(assistant_response, assistant_role)
        return assistant_response

    
if __name__ == "__main__":
    # set an argument parser so that I can get the desired llm
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="OPENAI_GPT4_MINI")
    args = parser.parse_args()

    llm = LLM_Wrapper(LLM_Models[args.llm.upper()])
    messages = [
        ["Hello, you are a chatbot", "system"],
        ["Hello, I'm user 1", "user"],
        ["Hello, I'm user 2", "user"],
    ]
    for msg in messages: 
        llm.send_message(*msg)
    llm.print_trace()
