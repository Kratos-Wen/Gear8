"""Semantic retrieval and response generation matching original prompt behavior."""

import json
import numpy as np


def retrieve_relevant_history(current_query, history, embedding_model, top_k=2):
    """Retrieve top-k relevant dialogue history items."""
    if not history:
        return []

    history_texts = [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in history]
    history_embeddings = embedding_model.encode(history_texts)
    query_embedding = embedding_model.encode([current_query])
    similarities = np.dot(history_embeddings, query_embedding[0])
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [history_texts[i] for i in top_indices]


def build_prompt(query, retrieved_history_text, object_infos):
    """Build the original response prompt text."""
    reduced_object_infos = object_infos[:3]
    prompt = f"""
You are an industrial assembly assistant. 
Previous related interactions:
{retrieved_history_text}
You have access to internal information about detected objects:
{json.dumps(reduced_object_infos, ensure_ascii=False, indent=2)}
The user cannot see the internal information.
User's question: {query}
Please keep in mind:
1. If the user's question refers to a **single part**, provide a **concise and precise answer**, explicitly mentioning the exact part name(s). 
2. If the user's question refers to **multiple parts**, ask the user to clarify which specific part they are referring to. 
3. Do **not include any additional details** or context. Only provide the necessary part information in your response.
4. Do **not mention the internal database** or any internal data sources in your response.
5. End your response with the tag <end> to indicate the completion.
6. Please make sure to **avoid** duplicate information in your response.
7. Ensure that your answer is the only content, followed by a newline.
"""
    return prompt

