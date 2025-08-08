
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Airbnb Dashboard", layout="wide", page_icon="🏠")
st.title("🏠 Airbnb Analysis Dashboard")

# Placeholder: Assume data is already loaded externally
# listings, mock_calendar, reviews are assumed to be DataFrames
st.markdown("**Note:** This dashboard assumes preloaded datasets: `listings`, `mock_calendar`, and `reviews`.")

# Section 1: Room Type Distribution Pie Chart
st.subheader("Room Type Distribution")
try:
    fig = px.pie(listings, names='room_type', title='Room Type Distribution', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error in room type pie chart: {e}")

# Section 2: Price Trend by Date (Line Chart)
st.subheader("Price Trend Over Time")
try:
    mock_calendar['date'] = pd.to_datetime(mock_calendar['date'])
    price_by_date = mock_calendar.groupby('date')['price'].mean().reset_index()
    fig = px.line(price_by_date, x='date', y='price', title='Average Price Trend by Date', template='plotly_dark')
    fig.update_traces(line_color='cyan')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error in price trend chart: {e}")

# Section 3: Top 10 Reviewed Listings (Table)
st.subheader("Top 10 Listings by Number of Reviews")
try:
    top_df = listings[['name', 'host_name', 'price', 'availability_365', 'number_of_reviews']].sort_values(
        by='number_of_reviews', ascending=False).head(10)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(top_df.columns),
                    fill_color='darkslategray',
                    font=dict(color='white', size=12),
                    align='left'),
        cells=dict(values=[top_df[col] for col in top_df.columns],
                   fill_color='black',
                   font=dict(color='white', size=11),
                   align='left'))
    ])
    fig.update_layout(title='Top 10 Listings by Reviews')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error displaying top listings table: {e}")

# Section 4: Prediction Placeholder
st.subheader("🧠 Predictive Model Output")
st.markdown("This section is a placeholder for integrating predictive models (e.g., price forecasting or review sentiment classification).")
st.info("To display predictions, add model inference code with preprocessed input.")

# Section 5: Key Insights
st.subheader("📌 Key Business Insights")
st.markdown("""
- Most listings are **Entire home/apt**, indicating Airbnb's shift toward full-property rentals.
- **Room types** vary significantly in pricing; pricing visualization can aid pricing strategy.
- Prices fluctuate **seasonally**, suggesting dynamic pricing models can improve revenue.
- Top-reviewed listings tend to have competitive pricing and high availability.
""")
