from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP


class LLM:
    def __init__(
            self,
            model_url: str ='https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf',
            embedded_model: str = 'BAAI/bge-small-en'
    ):
        self.model_url = model_url
        self.embedded_model = embedded_model

    def apply_embedded(self):

        embed_model = HuggingFaceEmbedding(model_name=self.embedded_model)
        return embed_model


    def get_llm_instance(self):
        model_url = self.model_url
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": 1},
            verbose=True,
        )
        return llm
