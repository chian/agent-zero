from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
import json
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from llama_cpp.llama_grammar import LlamaGrammar
from types import SimpleNamespace

class ARGO_LLM:
    """
    A class representing an ARGO language model.

    This class uses the invoke model helper function and implements the _call function.
    Autogen changes can be found between *call and *identifying_params methods.

    Attributes:
        argo (ArgoWrapper): An instance of ArgoWrapper.
        name (str): The name of the model, set to 'ARGO_LLM'.
    """

    argo: ArgoWrapper
    name = 'ARGO_LLM'

    def __init__(self, argo, model_type='gpt4', temperature=0.7):
        """
        Initialize the ARGO_LLM instance.

        Args:
            argo: An instance of ArgoWrapper or a function to create one.
            model_type (str): The type of model to use. Defaults to 'gpt4'.
            temperature (float): The temperature parameter for text generation. Defaults to 0.7.
        """
        self.argo = argo
        self.chat = chat(self.argo)

    base_url = ArgoWrapper.default_url  # AutoGen Required

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        """Placeholder for generations property."""
        return

class chat:
    """
    A class representing a chat interface for ARGO_LLM.

    Attributes:
        completions (completions): An instance of the completions class.
    """

    def __init__(self, argo: ArgoWrapper):
        """
        Initialize the chat instance.

        Args:
            argo (ArgoWrapper): An instance of ArgoWrapper.
        """
        self.completions = completions(argo)

class completions:
    """
    A class handling completions for ARGO_LLM.

    This class provides methods to create completions based on given messages.
    """

    def __init__(self, argo: ArgoWrapper):
        """
        Initialize the completions instance.

        Args:
            argo (ArgoWrapper): An instance of ArgoWrapper.
        """
        def create(messages: List[str], stream: bool):
            """
            Create a completion based on the given messages.

            Args:
                messages (List[str]): A list of message strings.
                stream (bool): Whether to stream the response or not.

            Returns:
                SimpleNamespace: An object containing the response details.
            """
            length = len(messages)
            prompt = 'These are the previous messages for context:\n'
            for message in messages[:length - 1]:
                prompt += message['content'] + '\n'
            prompt += 'The current prompt is:\n' + messages[-1]['content']
            response = self.argo.invoke(prompt)
            
            message = SimpleNamespace(
                function_call = None,
                tool_calls = None,
                content = response['response'],
            )
            choice = SimpleNamespace(
                text = response['response'],
                message = message
            )
            result = SimpleNamespace(
                model = 'argo',
                usage = None,
                choices = [choice],
            )
            return result

        self.argo = argo
        self.create = create

class ARGO_EMBEDDING:
    """
    A class representing ARGO embeddings.

    This class provides methods to generate embeddings for documents and queries.

    Attributes:
        argo (ArgoEmbeddingWrapper): An instance of ArgoEmbeddingWrapper.
    """

    argo: ArgoEmbeddingWrapper

    def __init__(self, argo_wrapper: ArgoEmbeddingWrapper):
        """
        Initialize the ARGO_EMBEDDING instance.

        Args:
            argo_wrapper (ArgoEmbeddingWrapper): An instance of ArgoEmbeddingWrapper.
        """
        self.argo = argo_wrapper

    @property
    def _llm_type(self) -> str:
        """Get the LLM type."""
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Generate an embedding for the given prompt.

        Args:
            prompt (str): The input text to embed.
            stop (Optional[List[str]]): A list of stop sequences (not used in this implementation).
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated embedding.
        """
        if stop is not None:
            print(f"STOP={stop}")
        response = self.argo.invoke(prompt)
        return response['embedding']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        """Placeholder for generations property."""
        return

    def embed_documents(self, texts):
        """
        Embed multiple documents.

        Args:
            texts: The input texts to embed.

        Returns:
            The embeddings for the input texts.
        """
        return self._call(texts)

    def embed_query(self, query: str):
        """
        Embed a single query string.

        Args:
            query (str): The query string to embed.

        Returns:
            The embedding for the query string.
        """
        return self._call(query)[0]
