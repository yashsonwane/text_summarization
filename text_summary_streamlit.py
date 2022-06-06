from math import fabs
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from summarizer import Summarizer
import os
# from keybert import KeyBERT as kb
nltk.download('punkt')
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
# from keybert import KeyBERT
#page layout
st.set_page_config(layout = 'wide', initial_sidebar_state='collapsed')

with st.spinner('Wait for app to load...'):
    if 'model' not in st.session_state:
        st.session_state.model=T5ForConditionalGeneration.from_pretrained(('t5-small'))

    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer=T5Tokenizer.from_pretrained("t5-small")

    if 'device' not in st.session_state:
        st.session_state.device=torch.device('cpu')








def cleanText(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[\([{})\]]", "", text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text


def getSummary(text,tokenizer):
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # device = torch.device('cpu')
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(st.session_state.device)

    summary_ids = st.session_state.model.generate(tokenized_text,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=96,
                                    early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output


def abstractive(body):



        
        # tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # nltk.download('punkt')


        summary = ""

        text = ""

        for line in body:
            line = cleanText(line)
            line = line.replace("\n", " ")
            text += line
#creating chunks of 400 words
        textTokens = word_tokenize(text)
        totalTokens = len(textTokens)
        chunkCounter = 0
        maxTokenLen = 400
        chunkList = []
        start = 0
        end = maxTokenLen

        if(totalTokens % maxTokenLen) == 0:
            totalChunks = int(totalTokens / maxTokenLen)

            for i in range(0,totalChunks):
                tempTokens = textTokens[start:end]
                chunkText = ' '.join([str(elem) for elem in tempTokens])
                chunkList.append(chunkText)
                start = end
                end += maxTokenLen
                chunkText = ""

        else:
            totalChunks = int(totalTokens / maxTokenLen) + 1

            for i in range(0,(totalChunks-1)):
                tempTokens = textTokens[start:end]
                chunkText = ' '.join([str(elem) for elem in tempTokens])
                chunkList.append(chunkText)
                start = end
                end += maxTokenLen
                chunkText = ""

            tempTokens = textTokens[start:totalTokens]
            chunkText = ' '.join([str(elem) for elem in tempTokens])
            chunkList.append(chunkText)

        for chunk in chunkList:
            tempSummary = getSummary(chunk, st.session_state.tokenizer)
            summary += tempSummary


        return summary


def get_topic_list(text, num_topics):
#     tokenizing in the sentences
    x = sent_tokenize(text)
    
#     generating the list to store tokenized sentence
    sentences = []
    for line in x:
                sentences.extend(sent_tokenize(line))
            
#     creating a dataframe of sentences        
    df = pd.DataFrame({'text' : sentences})
    
#     vectorizing the sentences
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df['text'])
    
#     using nmf model for topic modeling
    nmf_model = NMF(n_components=num_topics,random_state=42)
    nmf_model.fit(dtm)
    
#     getting the topics separated
    topic_results = nmf_model.transform(dtm)
#     print(topic_results)
    
#     adding the topic numbers to the dataframe
    df['Topic'] = topic_results.argmax(axis=1)
    
#     getting the list of unique topics
    topics = list(df.Topic.unique())
#     print(topics)
    
#     generating the list of differentiated topics
    b = []
    for topic in topics:
        b.append(' '.join(list(df[df.Topic == topic].text)))
        
#     getting the list of sentences             
    return b

 
def extra_abst(body):
    bert_model = Summarizer()
    result = bert_model(body, ratio = 0.8)
    full = ''.join(result)
    # print(full)
    # model = T5ForConditionalGeneration.from_pretrained(('t5-small'))
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # device = torch.device('cpu')
    preprocessed_text = full.strip().replace('\n','')
    t5input_text = "summarize:" + preprocessed_text
    tokenized_text = st.session_state.tokenizer.encode(t5input_text, return_tensors='pt', max_length=512,truncation=True).to(st.session_state.device)
    summary_ids = st.session_state.model.generate(tokenized_text, min_length=30,max_length=100)
    summary = st.session_state.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

      
def summary_on_topics(topics):
    summaries = []
    for topic in topics:
        summary = extra_abst(topic)
        print(summary)
        summaries.append(summary)
        summaries_df=pd.DataFrame(summaries,columns=["Topics"])
    return summaries_df


##################Streamlit code###########################
st.title("Discite Text Summarizer")


if 'text' not in st.session_state:
    st.session_state.text='your text'


# with st.form(key="first", clear_on_submit=False):
#     text = st.text_area('Please input your text here:')  
#     st.session_state.text=text
    # submitted = st.form_submit_button("Generate ")

    # if submitted:
    #     with st.expander("Expand to see your Text"):
    #         st.text_input(st.session_state.text)
            

with st.form(key="second", clear_on_submit=False):
    text = st.text_area('Please input your text here:')
    st.session_state.text=text
    first, second, third = st.columns(3)
    num_topics = first.text_input("Please input the number of topics here:")

    submitted = st.form_submit_button("Generate ")

    if submitted:
        
        with st.spinner('Generating Summary...'):
            topic_list = get_topic_list(st.session_state.text, int(num_topics))
            abstractive_summary = abstractive(st.session_state.text)
            try:
                summary2 = summary_on_topics(topic_list)
            except Exception as e:
                print(e)
                pass

            st.subheader("Overall Summary")
            st.write(abstractive_summary)
            st.subheader("Topicwise Summary")
            st.table(summary2)

        
# with st.form(key="second", clear_on_submit=True):
#     pass
#     # topic = st.text_input("Please input the number of topics here:")

# print(abstractive(text))
