
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="Airbnb Dashboard", layout="wide")

# Dark theme for Plotly
plotly_template = "plotly_dark"

# Load Data
listings = pd.read_csv("listings.csv")
calendar = pd.read_csv("mock_calendar.csv")  # your generated calendar
reviews = pd.read_csv("reviews.csv")

# --- Preprocessing ---
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)
calendar['date'] = pd.to_datetime(calendar['date'])

# --- Sidebar filters ---
st.sidebar.header("Filters")
room_types = listings['room_type'].unique()
selected_room = st.sidebar.multiselect("Room Type", room_types, default=list(room_types))

neighs = listings['neighbourhood'].unique()
selected_neigh = st.sidebar.multiselect("Neighbourhood", neighs, default=list(neighs))

price_min, price_max = listings['price'].min(), listings['price'].max()
price_range = st.sidebar.slider("Price Range", min_value=int(price_min), max_value=int(price_max), value=(int(price_min), int(price_max)))

# --- Filtered Data ---
filtered = listings[
    (listings['room_type'].isin(selected_room)) &
    (listings['neighbourhood'].isin(selected_neigh)) &
    (listings['price'].between(price_range[0], price_range[1]))
]

# --- Dashboard Title ---
st.title("📊 Airbnb Listings Dashboard with ML")

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Listings", len(filtered))
col2.metric("Average Price", f"${filtered['price'].mean():.2f}")
col3.metric("Avg. Availability (days)", int(filtered['availability_365'].mean()))

st.markdown("---")

# --- Plot 1: Room Type Share ---
room_fig = px.pie(filtered, names='room_type', title="Room Type Share", template=plotly_template)
st.plotly_chart(room_fig, use_container_width=True)

# --- Plot 2: Listings per Neighbourhood ---
bar_fig = px.bar(filtered['neighbourhood'].value_counts().reset_index(),
                 x='index', y='neighbourhood',
                 labels={'index': 'Neighbourhood', 'neighbourhood': 'Count'},
                 title="Listings per Neighbourhood", template=plotly_template)
st.plotly_chart(bar_fig, use_container_width=True)

# --- Plot 3: Box Plot of Price by Room Type ---
box_fig = px.box(filtered, x='room_type', y='price', title="Price Distribution by Room Type", template=plotly_template)
st.plotly_chart(box_fig, use_container_width=True)

# --- Plot 4: Price Trend Over Time ---
calendar_sample = calendar[calendar['listing_id'].isin(filtered['id'])]
price_trend = calendar_sample.groupby('date')['price'].mean().reset_index()
line_fig = px.line(price_trend, x='date', y='price', title="📆 Average Price Trend Over Time", template=plotly_template)
st.plotly_chart(line_fig, use_container_width=True)

# --- Table: Top Listings ---
st.subheader("📌 Top Listings by Reviews")
top_listings = filtered[['name', 'host_name', 'price', 'availability_365', 'number_of_reviews']]                .sort_values(by='number_of_reviews', ascending=False).head(10)
st.dataframe(top_listings)

st.markdown("---")

# --- ML Prediction: Predict Price ---
st.subheader("🧠 Price Prediction with Random Forest")

ml_data = filtered[['price', 'availability_365', 'number_of_reviews', 'reviews_per_month', 'minimum_nights']].dropna()
X = ml_data.drop('price', axis=1)
y = ml_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

st.write(f"Model RMSE: **${rmse:.2f}**")

st.markdown("#### Try Your Own Inputs")
input_cols = st.columns(len(X.columns))
user_input = []
for i, col in enumerate(X.columns):
    val = input_cols[i].number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    user_input.append(val)

predicted_price = model.predict([user_input])[0]
st.success(f"🎯 Predicted Price: ${predicted_price:.2f}")
