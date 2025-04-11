import sys
import mongodb.mongo_handler as mdb
from openai import OpenAI

import model_settings as ms
import prompt_templates.load_templates as plt
import settings as s


class GenerateMissingArgumentGPT4Mini():
    def __init__(self,data_list_to_use):
        
        self.data_list_to_use = data_list_to_use
        self.prompt = plt.load_prompt_files("argu-generation-prompt.txt")
        self.model = s.OPEN_AI_MODEL

        key = s.OPENAI_KEY
        self.client = OpenAI(
                api_key=key
            )
        
        # parameters used for this classification task
        self.temperature = 0.9
        self.top_p = 1
        self.max_completion_tokens = 1000

    def get_chat_response(self, prompt):

        try:
            completion = self.client.chat.completions.create(
                model=s.OPEN_AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=s.TEMPERATURE,
                top_p=s.TOP_P,
                max_completion_tokens=s.MAX_NEW_TOKENS
            )

            argument_answer = completion.choices[0].message.content
            return argument_answer

        except Exception as e:
            print(f"API call failed: {e}")
            sys.exit()

    def generate_missing_argument(self):
        generated_data = []
        for data in self.data_list_to_use:
            prompt = self.prompt.format(**data)
            response = self.get_chat_response(prompt)
            generated_data.append(response)
        return generated_data


if __name__ == "__main__":
    argu = {ms.SCHEME : "scheme", "argument" : "argument", "topic": "topic",  "new_topic" : "new_topic"}
    x = GenerateMissingArgumentGPT4Mini(argu).generate_missing_argument()
    print(x)
    