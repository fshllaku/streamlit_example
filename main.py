import streamlit as st
from transformers import pipeline

from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def load_file():
    """Load text from file"""
    uploaded_file = st.file_uploader("Upload Files",type=['txt'])

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            raw_text = str(uploaded_file.read(),"utf-8")
        return raw_text


if __name__ == "__main__":

    # App title and description
    st.title("Answering questions from text")
    st.write("Upload text, pose questions, get answers")

    # Load file
    raw_text = load_file()
    if raw_text != None and raw_text != '':

        # Display text
        with st.expander("See text"):
            st.write(raw_text)

        # Perform question answerin
        model_name = "deepset/roberta-base-squad2"
        question_answerer = pipeline('question-answering', model=model_name, tokenizer=model_name)

        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        answer = ''
        question = st.text_input('Ask a question')

        if question != '' and raw_text != '':
            answer = question_answerer({
                'question': question,
                'context': raw_text
            })

        st.write(answer)
        st.write(question_answerer)