"""Wrapper for retrieval-augmented multimodal interaction execution."""

from .semantic_retrieval_and_response_generation import build_prompt, retrieve_relevant_history
from .speech_input_output import text_to_speech


class RetrievalAugmentedMultimodalInteraction:
    """Thin orchestrator to keep paper-aligned module naming."""

    def __init__(self, llm, tts_engine, embedding_model, dialogue_history_ref, object_infos_ref, stop_audio_fn):
        self.llm = llm
        self.tts_engine = tts_engine
        self.embedding_model = embedding_model
        self.dialogue_history_ref = dialogue_history_ref
        self.object_infos_ref = object_infos_ref
        self.stop_audio_fn = stop_audio_fn

    def process_query(self, query, max_tokens, temperature, stop_token):
        """Process one query using the same semantics as the original script."""
        retrieved_history = retrieve_relevant_history(
            query,
            self.dialogue_history_ref(),
            self.embedding_model,
            top_k=2,
        )
        retrieved_history_text = "\n".join(retrieved_history)
        prompt = build_prompt(query, retrieved_history_text, self.object_infos_ref())
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            stop=stop_token,
        )
        answer = response["choices"][0]["text"].strip()
        print("LLM answer:", answer)
        text_to_speech(answer, self.tts_engine, self.stop_audio_fn)
        self.dialogue_history_ref().append({"user": query, "assistant": answer})

