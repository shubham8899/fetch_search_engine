from txtai import Embeddings
import pandas as pd
import streamlit as st

def preprocess_text(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = ' '.join(text.split())
    return text

dataset = pd.read_csv('dataset.csv', index_col=0)

dataset_concatenated = []

for index, row in dataset.iterrows():
    non_nan_elements = [str(element) for column, element in row.items() if column != 'OFFER' and not pd.isna(element)]
    concatenated_row = ' '.join(non_nan_elements)
    dataset_concatenated.append(concatenated_row)

embeddings = Embeddings()
embeddings.load("dataset_index")

st.set_page_config(page_title="Fetch Rewards Search Engine", page_icon="🐕")
st.title("Fetch Rewards Search Engine 🐕")

text_search = st.text_input("Search Offers by Category, Brand or Retailer", value="")

if text_search:
    # st.empty()
    text_search = preprocess_text(text_search)
    search_data = embeddings.search(text_search, 30)

    search_results = pd.DataFrame(columns=['OFFER', 'SCORE'])

    for index, score in search_data:
        search_results = pd.concat([search_results, pd.DataFrame.from_records([{'OFFER': dataset.iloc[index]['OFFER'], 'SCORE':score }])], ignore_index=True)

    if search_results.iloc[0]['SCORE'] < 0.25:
        st.write("We could not find a close match, but here are some relevant suggestions: ")

    st.dataframe(search_results, width=1200)

    