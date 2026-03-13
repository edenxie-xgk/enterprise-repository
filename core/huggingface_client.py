from huggingface_hub import InferenceClient

from core.settings import settings


def connect_huggingface_model(model_name:str,token:str = settings.hf_token):
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        model=model_name,
    )
    return client