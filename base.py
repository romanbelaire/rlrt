from transformers import AutoModelForCausalLM, AutoConfig
import torch

class BaseModelSingleton:
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls, model_name, token=None):
        if cls._instance is None:
            cls._instance = cls(model_name, token)
        return cls._instance
    
    def __init__(self, model_name, token=None):
        if BaseModelSingleton._model is None:
            config = AutoConfig.from_pretrained(model_name, token=token)
            config.output_hidden_states = True
            config.return_dict = True
            config.return_dict_in_generate = True
            config.use_cache = False
            
            BaseModelSingleton._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                token=token
            ).to('cuda')
            
            # Freeze the base model
            for param in BaseModelSingleton._model.parameters():
                param.requires_grad = False
                
    @property
    def model(self):
        return BaseModelSingleton._model
    
    def to(self, device):
        if BaseModelSingleton._model is not None:
            BaseModelSingleton._model.to(device)