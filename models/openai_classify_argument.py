import sys

from openai import OpenAI
import settings as s
import models.models_base_class as mbc

class OpenAIModel(mbc.LLMPrompt):
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

        self.model = s.OPEN_AI_MODEL
        key = s.OPENAI_KEY
        self.client = OpenAI(
            api_key=key
        )

    def get_chat_response(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=s.OPEN_AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=s.TEMPERATURE,
                top_p=s.TOP_P,
                max_completion_tokens=s.MAX_NEW_TOKENS
            )
            return  completion.choices[0].message.content

        except Exception as e :
            print(f"API call failed: {e}")
            sys.exit()

    def _do_prompt(self, prompt: str) -> str:
        return self.get_chat_response(prompt)


if __name__ == "__main__":
    argu = "Secretary Clinton and others , politicians , should have been doing this for years , not right now , because of the fact that we 've created a movement ."
    x = OpenAIModel().get_chat_response(argu)
    print(x)









