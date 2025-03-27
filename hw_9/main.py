import os
import pandas as pd
import streamlit as st
from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Forecasting the value of real estate",
)

model_path = 'rf_fitted.pkl'
total_square = st.sidebar.number_input("Enter the total area", 8, 2070, 30)
rooms = st.sidebar.number_input(
    "Enter the number of rooms",
    1, 15, 1,
)
floor = st.sidebar.number_input(
    "Enter the floor?",
    1, 66, 1,
)
inputDF = pd.DataFrame(
    {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor
    },
    index=[0],
)
if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('realty_data.csv')
    train_model(train_data)

model = read_model('rf_fitted.pkl')

preds = model.predict(inputDF)
preds = preds.round(1)

st.image("imgs/scale.jpg", use_column_width=True)
st.write(f"The cost of real estate: {preds} p.")