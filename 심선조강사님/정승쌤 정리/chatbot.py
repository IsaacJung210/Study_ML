# pip install streamlit
# pip install streamlit_chat
# streamlit run chatbot.py 로 실행

import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True) #한번만 모델을 로드하고 결과값은 변경되도록
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('data/wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('심리상담 챗봇')
st.markdown("[참고사이트](https://streamlit.io/)")

if 'generated' not in st.session_state: #쳇봇의 대화저장용 세션 -스트림릿이 재시작해도 유지된다. 
    st.session_state['generated'] = []

if 'past' not in st.session_state: #나의 대화내용 저장
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('상담 내용 입력 : ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')