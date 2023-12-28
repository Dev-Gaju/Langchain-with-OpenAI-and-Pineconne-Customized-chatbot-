from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = "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"
# model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='dc75306b-a1a9-407b-aa5a-2c3240f3d31e', environment='us-west4-gcp-free')
index = pinecone.Index('langchain-chatbot')
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    # return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
    if 'matches' in result and len(result['matches']) >= 2:
        metadata_0 = result['matches'][0].get('metadata', {}).get('text', 'No metadata available for match 0')
        metadata_1 = result['matches'][1].get('metadata', {}).get('text', 'No metadata available for match 1')
        return f"{metadata_0}\n{metadata_1}"
    else:
        return "No matches or insufficient matches found."

def query_refiner(conversation, query):
    response = openai.completions.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string