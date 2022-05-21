import pandas as pd
import streamlit as st
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import os

API_keys = open("Password/keys.txt", "r").read().splitlines()

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] in API_keys:
            st.session_state["password_correct"] = True
            st.session_state["load_model"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "API Key", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "API Key", type="password", on_change=password_entered, key="password"
        )
        st.error("Error: 401 Unauthorized Response")
        return False
    else:
        # Password correct.
        return True

if check_password():
    # if st.session_state["load_model"]:
    with st.spinner('Loading data...'):
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
        # st.session_state["load_model"] = False
    st.title('Sentiment Analysis')

    uploaded_file = st.file_uploader("Upload a file", type=["csv"])

    if os.path.exists("storage/limits.txt"):    
        limits = open("storage/limits.txt", "r+").read().splitlines()
    else:
        limits =[0,300]
    if os.path.exists("storage/add_text.txt"):  
        add_text = open("storage/add_text.txt", "r").read().splitlines()
    else:
        add_text =[]
    if os.path.exists("storage/add_star.txt"):  
        add_star = open("storage/add_star.txt", "r").read().splitlines()
    else:
        add_star =[]  

    lower_limit = int(limits[-2])
    higher_limit = int(limits[-1])

    def get_data(lower_limit, higher_limit, df):
        df = df[pd.notnull(df["Text"])]
        texts = list(df["Text"])
        texts = texts[lower_limit:higher_limit]
        
        stars = list(df["Star"])
        
        with st.spinner('Predicting stuff...'):
            inputs = tokenizer(texts, return_tensors="tf", padding='max_length',max_length = 30, truncation=True)
            outputs = model(inputs)
            logits = outputs.logits    
            for x in range(len(texts)):
                if softmax(logits[x].numpy())[2] * 100 >= 90.0 and stars[x] < 3:
                    add_text.append(str(texts[x]))
                    add_star.append(str(stars[x]))
        if len(add_text) == 0:
            lower_limit = higher_limit
            higher_limit += 300
            get_data(lower_limit, higher_limit, df)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        while len(add_text) <= 10:
            lower_limit = higher_limit
            higher_limit += 300
            get_data(lower_limit, higher_limit, df)
        if st.button('View More', on_click=None)  == True:
            print("button clicked")
            lower_limit = higher_limit
            higher_limit += 300
            get_data(lower_limit, higher_limit, df)
            
        limits = open("storage/limits.txt", "w")
        add_text_txt = open("storage/add_text.txt", "w")
        add_star_txt = open("storage/add_star.txt", "w")
        limits.write(str(lower_limit)  +  "\n")
        limits.write(str(higher_limit)  +  "\n")
        add_text_txt.write("\n".join(add_text))
        add_star_txt.write("\n".join(str(x) for x in add_star))
        add_text_txt.close()
        add_star_txt.close()
        limits.close()
        
        new_df = pd.DataFrame({"Text":add_text, "Star":add_star})
        st.dataframe(new_df)
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(new_df)
        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )