import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. Page Basic Configuration
st.set_page_config(
    page_title="📊 Mobile Reviews Overview",
    page_icon="📊",
    layout="wide"
)

# 2. Simple & Fast Data Loading
@st.cache_data
def load_data():
    if os.path.exists("Mobile_Reviews_Cleaned.csv"):
        return pd.read_csv("Mobile_Reviews_Cleaned.csv")
    elif os.path.exists("Mobile Reviews Sentiment.csv"):
        return pd.read_csv("Mobile Reviews Sentiment.csv")
    return None

df = load_data()

# Check if data exists
if df is not None:
    
    # --- SECTION 1: WELCOME HEADER ---
    st.title("📱 Mobile Customer Reviews Analytics")
    st.divider()
    
    # --- SECTION 2: DATASET BRIEF ---
    st.markdown("#### 📝 Dataset Brief")
    st.markdown("""
    This dataset contains consumer reviews and feedback for various mobile phones across multiple e-commerce platforms. 
    It combines **customer demographics** (age, country), **device details** (brand, model, price), and **text properties** (review length, word count) 
    to understand and analyze customer satisfaction and sentiment trends.
    """)
    st.divider()

    # --- SECTION 3: POLISHED KPI CARDS ---
    st.markdown("#### ⚡ Quick Dataset Highlights")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.subheader("📝 Total Dataset Rows")
        st.info(f"**{df.shape[0]:,}** Verified Reviews")
    with kpi2:
        st.subheader("📐 Total Features")
        st.success(f"**{df.shape[1]}** Data Columns")
    with kpi3:
        st.subheader("🏢 Market Brands")
        st.warning(f"**{df['brand'].nunique()}** Smartphone Makers")
    with kpi4:
        st.subheader("💵 Financial Value")
        st.error(f"**${df['price_usd'].mean().round(2)}** Average Price")
        
    st.divider()

    # --- SECTION 4: CLEAN DATA PREVIEW WITH DYNAMIC SLIDER ---
    st.markdown("#### 🔍 Dataset Structure Preview")
    
    # Dynamic slider allowing users to choose how many rows to display
    max_available_rows = min(100, df.shape[0])
    rows_to_show = st.slider(
        label="Select number of rows to display below:", 
        min_value=5, 
        max_value=max_available_rows, 
        value=5, 
        step=5
    )
    
    st.dataframe(
        df.head(rows_to_show), 
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()

    # --- SECTION 5: ELEGANT VISUALIZATION ---
    st.markdown("#### 🎯 Overall Sentiment Share")

    # Group data for the chart
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment Class", "Total Reviews"]

    # Beautiful premium color palette
    clean_colors = {
        'Positive': '#2ca02c',   # Deep Emerald Green
        'Neutral': '#ffbb78',    # Pastel Amber Yellow
        'Negative': '#d62728'    # Soft Crimson Red
    }

    # Creating a fully interactive and clean Pie Chart
    fig = px.pie(
        sentiment_counts,
        names="Sentiment Class",
        values="Total Reviews",
        hole=0.5,
        color="Sentiment Class",
        color_discrete_map=clean_colors
    )
    
    # Clean layout styling for the chart
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Display the final polished chart
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("⚠️ System Alert: Could not locate the dataset file. Please check your storage directory.")

# --- SECTION 6: FOOTER WITH CLOSING REMARKS ---
st.divider()
st.caption("Mobile Sentiment Analysis Dashboard | Developed by Yasser ")