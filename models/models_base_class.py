import model_settings as ms
import prompt_templates.load_templates as lt


def retry_on_none(retries=3):
    def decorator(func):
        def wrapper(self, **kwargs):
            for _ in range(retries):
                answer = func(self, **kwargs)
                if answer is not None:
                    return answer
            return None
        return wrapper
    return decorator


def init_model_prompt_base(**kwargs): # init the model so the underlying data can be loaded
    experiment_description = kwargs[ms.META][ms.EXPERIMENT_DESCRIPTION]

    model_parameters = kwargs[ms._DATACLASS]

    model = model_parameters[ms._MODELCLASS]
    prompt_template_raw = model_parameters[ms.PROMPT_TEMPLATE]
    llm = model(prompt_template_raw=prompt_template_raw,experiment_description=experiment_description)

    return llm

class LLMPrompt():
    """
    A base class for handling model prompts, inheriting from a dynamically specified class.
    """

    def __init__(self,prompt_template_raw=None,experiment_description=None):
        self.prompt_counter = 0
        
        self.prompt_template_raw = lt.load_prompt_files(prompt_template_raw)

          
        self.experiment_description = experiment_description

    def reset_prompt_counter(self):
        """Resets the prompt counter."""
        self.prompt_counter = 0

    def format_prompt(self, **kwargs):
        """Formats the prompt using the raw template and provided arguments."""
        return self.prompt_template_raw.format(**kwargs)
    

    @retry_on_none(retries=3)
    def do_prompt(self, **kwargs):
        

        prompt_template = self.format_prompt(**kwargs)
        prompt_template = prompt_template.strip()
        self.prompt_counter += 1
        
        print(f"Doing Prompt {prompt_template}")
        answer = self._do_prompt(prompt_template)
        if answer is not None :
            return answer


