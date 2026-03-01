import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from ml_pipeline import load_and_preprocess_data, build_and_train_model, save_model, load_model, predict_demand

# Set page config
st.set_page_config(page_title="EV Infrastructure AI", layout="wide", page_icon=":material/ev_station:")

# Custom CSS for ultra-premium modern EV look (Rounded, Green Theme)
st.markdown("""
<style>
    /* Global Variables */
    :root {
        --ev-green: #00E676;
        --ev-dark-green: #00C853;
        --ev-light-green: #E8F5E9;
        --border-radius-lg: 20px;
        --border-radius-md: 12px;
        --box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    /* Main Background & Text */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Metric Cards - Modern Squircle Design */
    .metric-card {
        background-color: #ffffff;
        border-left: 6px solid var(--ev-green);
        padding: 24px;
        border-radius: var(--border-radius-lg);
        box-shadow: var(--box-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 230, 118, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A1A1A;
        font-family: 'Inter', sans-serif;
        margin-top: 8px;
    }
    .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1A1A1A;
        font-weight: 700 !important;
    }
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: var(--border-radius-md);
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--ev-dark-green) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: var(--border-radius-md) !important;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--ev-dark-green) 0%, var(--ev-green) 100%);
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(0, 230, 118, 0.4);
        transform: translateY(-2px);
    }
    
    /* Expander & Forms */
    .st-emotion-cache-1vt4ygl {
        border-radius: var(--border-radius-md);
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #eee;
    }
    
    /* Info/Success Boxes */
    .stAlert {
        border-radius: var(--border-radius-md);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title(":material/electric_car: Macroscopic EV Demand Forecast")
st.markdown("**Autonomous Infrastructure Analytics System**")

# Sidebar
st.sidebar.header(":material/settings: Configuration")
data_path = st.sidebar.text_input("Dataset Source", "ev_charging_patterns.csv")

# Tabs
tab1, tab2, tab3 = st.tabs([
    ":material/insert_chart: Analytics Dashboard", 
    ":material/memory: ML Pipeline", 
    ":material/online_prediction: Predictive Simulation"
])

# --- Tab 1: Usage Analytics ---
with tab1:
    st.header("Infrastructure Utilization")
    if os.path.exists(data_path):
        demand_df, raw_df = load_and_preprocess_data(data_path)
        
        # --- Top KPIs ---
        with st.container():
            kpi1, kpi2, kpi3 = st.columns(3)
            st.markdown(f"""
            <div style="display: flex; gap: 24px; margin-bottom: 24px;">
                <div class="metric-card" style="flex:1;">
                    <div class="metric-label">Total Unique Stations</div>
                    <div class="metric-value">{demand_df['Charging Station Location'].nunique():,}</div>
                </div>
                <div class="metric-card" style="flex:1;">
                    <div class="metric-label">Max Hourly Demand</div>
                    <div class="metric-value">{demand_df['Total_Demand'].max():,.1f} <span style="font-size:1rem;color:#888;">kWh</span></div>
                </div>
                <div class="metric-card" style="flex:1;">
                    <div class="metric-label">Raw Session Rows processed</div>
                    <div class="metric-value">{len(raw_df):,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Interactive Charts using Plotly ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Hourly Charging Demand Trends")
            hourly_demand = demand_df.groupby('Hour')['Total_Demand'].mean().reset_index()
            fig1 = px.line(hourly_demand, x='Hour', y='Total_Demand', 
                           title='Average Hourly Charging Demand',
                           markers=True, line_shape='spline', template='plotly_dark')
            fig1.update_traces(line_color='#00E676', line_width=3, marker=dict(size=8, color='#00C853'))
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Total Energy Consumption by Location")
            location_demand = demand_df.groupby('Charging Station Location')['Total_Demand'].sum().reset_index()
            fig3 = px.bar(location_demand, x='Charging Station Location', y='Total_Demand',
                          title='Total Energy Consumption by Location',
                          color='Charging Station Location', template='plotly_dark')
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.subheader("Demand Distribution by Day of Week")
            day_demand = demand_df.groupby('DayOfWeek')['Total_Demand'].sum().reset_index()
            # Map day of week to names
            days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
            day_demand['Day Name'] = day_demand['DayOfWeek'].map(days)
            fig2 = px.bar(day_demand, x='Day Name', y='Total_Demand', 
                          title='Total Demand by Day of Week',
                          color='Day Name', template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)
            
        st.divider()
        st.header("Advanced Demand Drivers")
        
        st.subheader("Environmental Impact (Temperature vs Demand)")
        # Add a jitter to temperature to make the scatter plot more readable
        fig_temp = px.scatter(demand_df, x='Temperature (°C)', y='Total_Demand', 
                              color='IsWeekend', 
                              color_continuous_scale='Greens',
                              trendline="ols",
                              title='How Weather affects Infrastructural Load',
                              template='plotly_dark',
                              opacity=0.7)
        # Map IsWeekend continuous to Discrete legend
        fig_temp.update_layout(coloraxis_showscale=False)
        fig_temp.data[0].name = 'Weekday'
        fig_temp.data[0].showlegend = True
        
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("Weekend vs Weekday Load Variance")
        # Map 1/0 to Yes/No
        demand_df['Weekend Label'] = demand_df['IsWeekend'].map({1: 'Weekend', 0: 'Weekday'})
        fig_box = px.box(demand_df, x='Weekend Label', y='Total_Demand', 
                         color='Weekend Label',
                         title='Statistical Variance by Commute Type',
                         template='plotly_dark',
                         color_discrete_sequence=['#1E88E5', '#00E676'])
        st.plotly_chart(fig_box, use_container_width=True)
        st.subheader("Macroscopic Feature Correlations")
        # Select numeric cols only, safely drop specific columns if they exist
        numeric_cols = demand_df.select_dtypes(include=[np.number]).drop(columns=['DayOfWeek', 'Location_Encoded', 'Month', 'IsWeekend'], errors='ignore')
        corr_matrix = numeric_cols.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                             title='Interconnectivity of Demand Variables',
                             color_continuous_scale='Greens')
        fig_corr.update_layout(template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander(":material/database: View Aggregated Datastore"):
            st.dataframe(demand_df, use_container_width=True)
    else:
        st.error(f"Dataset not found at {data_path}. Please check the path.")

# --- Tab 2: ML Model Training ---
with tab2:
    st.header(":material/model_training: Algorithm Calibration")
    st.write("Train the Random Forest Regressor to predict aggregated **Total Demand (kWh)** based on time, location, and conditions.")
    
    if st.button("Train Macroscopic Model", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Calibrating Decision Trees on Aggregated Data..."):
            if os.path.exists(data_path):
                demand_df, _ = load_and_preprocess_data(data_path)
                model, metrics, results_df = build_and_train_model(demand_df)
                
                # Save model
                save_model(model, "/Users/agnik/Desktop/genai/ev_demand_model.pkl")
                
                st.success("Model compiled and verified successfully!")
                
                # Model Evaluation Metrics
                eval_col1, eval_col2, eval_col3 = st.columns(3)
                
                with eval_col1:
                    st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")
                with eval_col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f}")
                with eval_col3:
                    st.metric("R-Squared (R2)", f"{metrics['R2']:.2f}")
                
                st.divider()
                st.subheader("Demand Prediction: Actual vs Predicted")
                
                # Plotly Actual vs Predicted Line Chart as requested
                fig_res = px.line(results_df, 
                                  title='Actual vs Predicted Demand (Validation Sample)',
                                  labels={'value': 'Total Demand (kWh)', 'index': 'Sample Index'},
                                  template='plotly_dark')
                # Change the colors for the lines
                fig_res.data[0].line.color = '#1E88E5' # Actual
                fig_res.data[1].line.color = '#00E676' # Predicted
                st.plotly_chart(fig_res, use_container_width=True)
                
            else:
                st.error("Dataset not found.")

# --- Tab 3: Demand Prediction ---
with tab3:
    st.header(":material/batch_prediction: Session Simulator")
    st.write("Input macroscopic conditions to forecast aggregate hourly electricity load for a specific location.")
    
    if os.path.exists("/Users/agnik/Desktop/genai/ev_demand_model.pkl") and os.path.exists("/Users/agnik/Desktop/genai/location_encoder.pkl"):
        if os.path.exists(data_path):
            demand_df, _ = load_and_preprocess_data(data_path)
            
            with st.form("prediction_form", border=True):
                st.subheader(":material/settings_input_component: Environmental & Temporal Parameters")
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    location = st.selectbox("Charging Station Location", sorted(demand_df['Charging Station Location'].unique()))
                    hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
                    
                with col_p2:
                    day_of_week = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
                    month = st.slider("Month", min_value=1, max_value=12, value=6)
                    is_weekend = st.radio("Is Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", index=0 if day_of_week < 5 else 1)
                    
                with col_p3:
                    temp = st.slider("Average Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
                    batt_cap = st.number_input("Average Block Battery Capacity (kWh)", min_value=10.0, max_value=200.0, value=75.0)

                submit = st.form_submit_button("Forecast Macroscopic Demand", type="primary", icon=":material/science:")
                
                if submit:
                    prediction = predict_demand(hour, day_of_week, month, is_weekend, temp, batt_cap, location)
                    
                    st.divider()
                    st.subheader(":material/monitoring: Forecasted Load")
                    
                    st.markdown(f"""
                    <div style="background-color: #ffffff; padding: 24px; border-radius: 20px; border-left: 6px solid #00E676; box-shadow: 0 8px 30px rgba(0,0,0,0.08); text-align: center;">
                        <h4 style="color: #666; font-size: 1rem; text-transform: uppercase;">Aggregated Hourly Demand</h4>
                        <h1 style="color: #1A1A1A; font-size: 3.5rem; margin: 10px 0;">{prediction:.1f} <span style="font-size: 1.5rem; color: #888;">kWh</span></h1>
                        <p style="color: #888; margin: 0;">Predicted for {location} at {hour}:00, Day {day_of_week}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
    else:
        st.warning(":material/warning: Predictive intelligence core offline. Please train the model in the ML Pipeline tab to generate the required binaries.")

