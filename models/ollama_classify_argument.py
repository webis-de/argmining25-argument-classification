import ollama
from loguru import logger

import model_settings as ms
import models.models_base_class as mbc
import settings as s


class OllamaBase(mbc.LLMPrompt):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.model = s.OLLAMA_MODEL
        self.client = ollama.Client( host=s.OLLAMA_URL)


    def get_chat_response(self, prompt):

        options = {
            "top_p" : s.TOP_P,
            "temperature" : s.TEMPERATURE,
            "max_new_tokens" : s.MAX_NEW_TOKENS,
        }
        
        # Send the POST request 
        response = None
        try :
            response = self.client.generate(model=s.OLLAMA_MODEL, prompt=prompt, options=options)
        except ollama.ResponseError as e :
            logger.error('Error:', e.error.lower())
            if "not found" in e.error.lower():
                self.client.pull(s.OLLAMA_MODEL)
                response = self.client.generate(model=s.OLLAMA_MODEL, prompt=prompt, options=options)
        if response is not None :
            response = response.get('response',None)
        return response


    def _do_prompt(self, prompt: str) -> str:
        return self.get_chat_response(prompt)



if __name__ == "__main__":
    argu = """

    Task:
    An argument consists of one or more premises that support a conclusion. You are given an argument and a topic. Your objective is to identify the premise(s) and conclusion of the given argument.

    Instructions:

    Identify only the premise(s) and conclusion that are explicitly stated in the argument.
    If the conclusion is not explicitly stated, generate a short conclusion that logically follows from the premise(s) and aligns with the given topic.
    Output Format:
    Provide your answer in the following exact format:

    Part-1: [Identify the premise(s), separated by a semicolon]
    Part-2: [Provide the conclusion, either given or generated]

    Important:
    Strictly follow the output format with no additional explanations or details.

    Example:
    Argument: Euthanasia can end unnecessary suffering.
    Topic: Should euthanasia be legalized?

    Part-1: [Euthanasia can end unnecessary suffering.]
    Part-2: [Euthanasia should be legalized.]

    Input:
    Argument: The universe probably isn't anthropocentric.
    Topic: Would the world be a better place without humans? """


    x = OllamaBase().get_chat_response(argu)
    print(x)








