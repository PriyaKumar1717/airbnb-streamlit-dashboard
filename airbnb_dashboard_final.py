
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Airbnb Dashboard", layout="wide")
st.title("🏠 Airbnb Analysis Dashboard")

@st.cache_data
def load_data():
    listings = pd.read_csv("listings.csv")
    calendar = pd.read_csv("mock_calendar.csv")
    reviews = pd.read_csv("reviews.csv")
    
    calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)
    calendar['date'] = pd.to_datetime(calendar['date'])
    return listings, calendar, reviews

listings, calendar, reviews = load_data()

# Sidebar filters
st.sidebar.header("Filter Listings")
neighborhoods = st.sidebar.multiselect("Neighborhood", listings['neighbourhood'].dropna().unique(), default=listings['neighbourhood'].dropna().unique())
room_types = st.sidebar.multiselect("Room Type", listings['room_type'].dropna().unique(), default=listings['room_type'].dropna().unique())
min_price, max_price = st.sidebar.slider("Price Range", int(listings['price'].min()), int(listings['price'].max()), (50, 500))

filtered = listings[
    (listings['neighbourhood'].isin(neighborhoods)) &
    (listings['room_type'].isin(room_types)) &
    (listings['price'] >= min_price) &
    (listings['price'] <= max_price)
]

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Price", f"${filtered['price'].mean():.2f}")
col2.metric("Total Listings", filtered.shape[0])
col3.metric("Average Reviews", f"{filtered['number_of_reviews'].mean():.2f}")

# Charts
st.subheader("Room Type Distribution")
fig_room = px.pie(filtered, names='room_type', title='Room Type Distribution', template='plotly_dark')
st.plotly_chart(fig_room, use_container_width=True)

st.subheader("Average Price by Neighborhood")
price_by_area = filtered.groupby("neighbourhood")['price'].mean().sort_values(ascending=False).reset_index()
fig_price = px.bar(price_by_area, x='neighbourhood', y='price', title='Avg Price by Neighborhood', template='plotly_dark')
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Review Scatter Plot")
fig_scatter = px.scatter(filtered, x='number_of_reviews', y='price', color='room_type',
                         hover_data=['name'], title='Price vs. Number of Reviews', template='plotly_dark')
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("📊 Export Filtered Data")
if st.button("Export to CSV"):
    filtered.to_csv("filtered_listings.csv", index=False)
    st.success("Exported to filtered_listings.csv")

# ML Model: Predict Price
st.subheader("💡 Price Prediction (ML Model)")
try:
    features = filtered[['number_of_reviews', 'availability_365']].fillna(0)
    labels = filtered['price']
    model = LinearRegression()
    model.fit(features, labels)
    predicted = model.predict(features)
    st.write(f"R² Score: {model.score(features, labels):.2f}")
except Exception as e:
    st.error(f"Model error: {e}")

# Clustering
st.subheader("🔍 Property Clustering (KMeans)")
try:
    cluster_features = filtered[['price', 'number_of_reviews', 'availability_365']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
    cluster_features['Cluster'] = kmeans.labels_
    fig_cluster = px.scatter(cluster_features, x='price', y='number_of_reviews', color='Cluster',
                             title='KMeans Property Clusters', template='plotly_dark')
    st.plotly_chart(fig_cluster, use_container_width=True)
except Exception as e:
    st.error(f"Clustering error: {e}")

# Report and Business Insights
st.subheader("📌 Key Findings & Recommendations")
st.markdown("""
- 💰 **Highest Price Factors**: Entire homes, more reviews, and high availability lead to higher prices.
- 🌍 **Popular Locations**: Certain neighborhoods consistently outperform others in terms of price and reviews.
- 📈 **Recommendations**:
    - Hosts should improve availability and request more reviews to boost prices.
    - Airbnb can enhance pricing tools by integrating demand-based dynamic pricing models.
""")
