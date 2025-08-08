
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🏠 Airbnb Analysis Dashboard")

@st.cache_data
def load_data():
    listings = pd.read_csv("listings.csv")
    calendar = pd.read_csv("mock_calendar.csv")
    reviews = pd.read_csv("reviews.csv")

    calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)
    calendar['date'] = pd.to_datetime(calendar['date'])

    return listings, calendar, reviews

listings, mock_calendar, reviews = load_data()

st.subheader("Room Type Distribution")
try:
    room_counts = listings['room_type'].value_counts().reset_index()
    room_counts.columns = ['Room Type', 'Count']
    fig_pie = px.pie(room_counts, values='Count', names='Room Type',
                     title='Room Type Distribution', template='plotly_dark')
    st.plotly_chart(fig_pie)
except Exception as e:
    st.error(f"Error in room type pie chart: {e}")

st.subheader("Price Trend Over Time")
try:
    daily_price = mock_calendar.groupby('date')['price'].mean().reset_index()
    fig_line = px.line(daily_price, x='date', y='price',
                       title='Average Price Trend Over Time',
                       labels={'price': 'Average Price', 'date': 'Date'},
                       template='plotly_dark')
    st.plotly_chart(fig_line)
except Exception as e:
    st.error(f"Error in price trend chart: {e}")

st.subheader("Top 10 Listings by Number of Reviews")
try:
    top_reviews = listings.sort_values(by='number_of_reviews', ascending=False).head(10)
    st.dataframe(top_reviews[['name', 'neighbourhood', 'room_type', 'price', 'number_of_reviews']])
except Exception as e:
    st.error(f"Error displaying top listings table: {e}")

st.subheader("🧠 Predictive Model Output")
st.info("This section is a placeholder for integrating predictive models (e.g., price forecasting or review sentiment classification).")

st.subheader("📌 Key Business Insights")
st.markdown("""
- Most listings are **Entire home/apt**, indicating Airbnb's shift toward full-property rentals.
- Room types vary significantly in pricing; **pricing visualization** can aid pricing strategy.
- Prices fluctuate seasonally, suggesting **dynamic pricing models** can improve revenue.
- Top-reviewed listings tend to have **competitive pricing** and **high availability**.
""")
