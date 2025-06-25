import streamlit as st
from src.pipeline_scripts import QAPipeline


@st.cache_resource
def get_pipeline():
    return QAPipeline()


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_prompt = st.chat_input("Ask your question:")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})

        try:
            qa_pipeline = get_pipeline()
            chain = qa_pipeline.get_chain()
            response = chain.invoke({'query': user_prompt})

            result = response["result"]
            source_docs = response["source_documents"]
            formatted = result + "\n\n**Source Documents:**\n" + "\n".join(str(doc.metadata) for doc in source_docs)

            st.chat_message("assistant").markdown(formatted)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted})
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
