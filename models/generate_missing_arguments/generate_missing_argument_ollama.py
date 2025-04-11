import ollama
import prompt_templates.load_templates as plt
import settings as s
from loguru import logger
import data_handling.data_loader as dla
import model_settings as ms

class GenerateMissingArgumentOllama(): 
    
    def __init__(self,data_list_to_use):
        self.data_list_to_use = data_list_to_use 
        self.prompt = plt.load_prompt_files("argu-generation-prompt.txt") # parameters used for this generation task self.temperature = 0.9 self.top_p = 1
        self.max_completion_tokens = 1000 
        self.final_data_list = []

        # higher values to the model can utilize more creative and diverse responses
        self.model_name = s.OLLAMA_MODEL
        self.top_p = 0.95
        self.temperature = 1

    def get_chat_response(self,prompt): 

        options = {"top_p" : self.top_p , "temperature" : self.temperature, } 
        client = ollama.Client( host=s.OLLAMA_URL, )
        
        # Send the POST request 
        response = None
        try :
            response = client.generate(model=s.OLLAMA_MODEL, prompt=prompt, options=options)
        except ollama.ResponseError as e :
            logger.error('Error:', e.error.lower())
            if "not found" in e.error.lower():
                client.pull(s.OLLAMA_MODEL)
                response = client.generate(model=s.OLLAMA_MODEL, prompt=prompt, options=options)
        if response is not None :
            response = response.get('response',None)
        return response

    def generate_missing_argument(self): 
        generated_data = [] 
        for data in self.data_list_to_use: 
            prompt = self.prompt.format(**data) 

            generated_argument = self.get_chat_response(prompt) 
            data_object = {ms.SCHEME : data[ms.SCHEME], ms.ARGUMENT : generated_argument, ms.TOPIC : data["new_topic"], ms.MODEL_NAME : s.OLLAMA_MODEL, ms.DATASET_NAME : ms.ETHIX_SYNTHETIC} # append the data object with required information to list
            generated_data.append(data_object) 
                             
        return generated_data # Clear the list after uploading 



if __name__ == "__main__": 
    argu = {ms.SCHEME : "scheme", "argument" : "argument", "topic": "topic",  "new_topic" : "new_topic"}
    x = GenerateMissingArgumentOllama(argu).generate_missing_argument()
    print(x)