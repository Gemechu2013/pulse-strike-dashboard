import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob
import warnings
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Pulse Strike Studio Dashboard", layout="wide", page_icon="🥊")

# ============================================================================
# DATA LOADING & PROCESSING (with caching)
# ============================================================================
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the CSV file with all transformations."""
    if uploaded_file is None:
        return None, None, None, None
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, None, None
    
    # 1. Sentiment Analysis
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity
    
    df['SentimentScore'] = df['OpenFeedback'].apply(get_sentiment)
    df['SentimentLabel'] = pd.cut(
        df['SentimentScore'], 
        bins=[-1, -0.1, 0.1, 1], 
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # 2. K-Means Clustering for Archetypes
    features_cluster = ['CombatPref', 'RhythmPref', 'FlowStateScore', 'FatigueScore']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features_cluster])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['ClusterID'] = kmeans.fit_predict(x_scaled)
    
    # Name clusters based on characteristics
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=features_cluster)
    
    def name_cluster(row):
        if row['FatigueScore'] > 0.5:
            return 'Fatigue Dropout'
        elif row['CombatPref'] > 0.5 and row['RhythmPref'] < 0:
            return 'Combat Warrior'
        elif row['RhythmPref'] > 0.5 and row['CombatPref'] < 0:
            return 'Rhythm Dancer'
        else:
            return 'Balanced Mover'
    
    cluster_names = dict(zip(centers.index, centers.apply(name_cluster, axis=1)))
    df['Archetype'] = df['ClusterID'].map(cluster_names)
    
    # 3. Train Churn Prediction Model
    model_features = [
        'FlowStateScore', 'FatigueScore', 'OverallSatisfaction',
        'InstructorRating', 'MusicVolRating', 'FloorQualRating',
        'SocialAtmRating', 'MonthsEnrolled'
    ]
    
    X = df[model_features]
    y = df['Churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return df, model, acc, model_features


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🥊 Pulse Strike Studio: Member Retention Dashboard")
    st.caption("Upload your `pulse_strike_survey.csv` file to begin analysis")
    
    # Sidebar: File Upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type="csv")
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file in the sidebar to get started.")
        st.markdown("""
        ### Expected Columns:
        - MemberID, Location, MonthsEnrolled, TeachStyle
        - CombatPref, RhythmPref, FlowStateScore, FatigueScore, EnergyPeak
        - MusicVolRating, FloorQualRating, InstructorRating, SocialAtmRating
        - OverallSatisfaction, Churned, OpenFeedback
        """)
        return
    
    # Load and process data
    df, model, acc, model_features = load_and_process_data(uploaded_file)
    
    if df is None:
        st.error("Failed to load data. Please check your file format.")
        return
    
    # Sidebar: Filters
    st.sidebar.header("🔍 Filters")
    
    location_filter = st.sidebar.multiselect(
        "Location", 
        options=sorted(df['Location'].unique()), 
        default=sorted(df['Location'].unique())
    )
    
    teach_style_filter = st.sidebar.multiselect(
        "Teaching Style", 
        options=sorted(df['TeachStyle'].unique()), 
        default=sorted(df['TeachStyle'].unique())
    )
    
    # Apply filters
    df_filt = df[
        df['Location'].isin(location_filter) & 
        df['TeachStyle'].isin(teach_style_filter)
    ].copy()
    
    if df_filt.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filters.")
        return
    
    # ========================================================================
    # DOWNLOAD SECTION (Sidebar)
    # ========================================================================
    st.sidebar.header("📥 Download Data")
    
    # Download filtered raw data
    csv_raw = df_filt.to_csv(index=False)
    st.sidebar.download_button(
        label="📄 Download Filtered Data (CSV)",
        data=csv_raw,
        file_name="pulse_strike_filtered.csv",
        mime="text/csv",
        help="Download the filtered dataset with original columns"
    )
    
    # Download processed data with new columns
    csv_processed = df_filt.to_csv(index=False)
    st.sidebar.download_button(
        label="📊 Download Processed Data (CSV)",
        data=csv_processed,
        file_name="pulse_strike_processed.csv",
        mime="text/csv",
        help="Download data with SentimentScore, Archetype, and other derived columns"
    )
    
    # Download summary statistics
    summary_stats = df_filt.describe().T
    csv_summary = summary_stats.to_csv()
    st.sidebar.download_button(
        label="📈 Download Summary Stats (CSV)",
        data=csv_summary,
        file_name="pulse_strike_summary_stats.csv",
        mime="text/csv",
        help="Download descriptive statistics for numeric columns"
    )
    
    # ========================================================================
    # KPI METRICS
    # ========================================================================
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Members", len(df_filt))
    with col2:
        churn_rate = df_filt['Churned'].mean() * 100
        st.metric("Avg Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate:.1f}% churned")
    with col3:
        avg_flow = df_filt['FlowStateScore'].mean()
        st.metric("Avg Flow State", f"{avg_flow:.1f}/10")
    with col4:
        st.metric("Model Accuracy", f"{acc*100:.1f}%")
    
    st.divider()
    
    # ========================================================================
    # TABS FOR DIFFERENT ANALYSIS SECTIONS
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview & EDA", 
        "👥 Member Archetypes", 
        "💬 Sentiment Analysis", 
        "🔮 Churn Predictor"
    ])
    
    # ------------------------------------------------------------------------
    # TAB 1: EXPLORATORY DATA ANALYSIS
    # ------------------------------------------------------------------------
    with tab1:
        st.header("📈 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Members by Location")
            loc_counts = df_filt['Location'].value_counts()
            fig1 = px.bar(
                x=loc_counts.index, 
                y=loc_counts.values, 
                color=loc_counts.values,
                color_continuous_scale='Blues',
                labels={'x': 'Location', 'y': 'Number of Members'},
                text_auto=True
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📊 Distribution of Flow State Score")
            fig2 = px.histogram(
                df_filt, 
                x='FlowStateScore', 
                nbins=10, 
                labels={'x': 'Flow State Score (1-10)', 'y': 'Number of Members'},
                color_discrete_sequence=['#9B59B6']
            )
            fig2.add_vline(
                x=df_filt['FlowStateScore'].mean(), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {df_filt['FlowStateScore'].mean():.1f}"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("🚨 Churn Rate by Month Enrolled")
        churn_month = df_filt.groupby('MonthsEnrolled')['Churned'].mean() * 100
        colors = ['#E74C3C' if 3 <= m <= 6 else '#3498DB' for m in churn_month.index]
        
        fig3 = go.Figure(go.Bar(
            x=churn_month.index, 
            y=churn_month.values, 
            marker_color=colors,
            text=[f"{v:.1f}%" for v in churn_month.values],
            textposition='auto'
        ))
        fig3.update_layout(
            title='Red = Danger Zone (Months 3-6)',
            xaxis_title='Months Enrolled',
            yaxis_title='Churn Rate (%)',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # ------------------------------------------------------------------------
    # TAB 2: MEMBER ARCHETYPES (CLUSTERING)
    # ------------------------------------------------------------------------
    with tab2:
        st.header("👥 Member Segmentation by Archetype")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Churn Rate by Archetype")
            churn_arch = df_filt.groupby('Archetype')['Churned'].mean().sort_values(ascending=False) * 100
            
            archetype_colors = {
                'Combat Warrior': '#E74C3C',
                'Rhythm Dancer': '#9B59B6',
                'Balanced Mover': '#2ECC71',
                'Fatigue Dropout': '#F39C12'
            }
            
            fig4 = px.bar(
                x=churn_arch.index, 
                y=churn_arch.values, 
                color=churn_arch.index,
                color_discrete_map=archetype_colors,
                labels={'y': 'Churn Rate (%)', 'x': 'Archetype'},
                text_auto='.1f%'
            )
            fig4.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            st.subheader("👥 Member Count by Archetype")
            arch_counts = df_filt['Archetype'].value_counts()
            
            fig5 = px.pie(
                values=arch_counts.values, 
                names=arch_counts.index,
                color=arch_counts.index,
                color_discrete_map=archetype_colors,
                hole=0.4
            )
            fig5.update_traces(textinfo='percent+label')
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        st.subheader("📋 Archetype Details")
        archetype_summary = df_filt.groupby('Archetype').agg({
            'FlowStateScore': 'mean',
            'FatigueScore': 'mean',
            'CombatPref': 'mean',
            'RhythmPref': 'mean',
            'OverallSatisfaction': 'mean',
            'Churned': 'mean'
        }).round(2)
        archetype_summary['Churned'] = archetype_summary['Churned'] * 100
        archetype_summary.columns = [
            'Avg Flow State', 'Avg Fatigue', 'Avg Combat Pref', 
            'Avg Rhythm Pref', 'Avg Satisfaction', 'Churn Rate (%)'
        ]
        st.dataframe(archetype_summary.style.format({
            'Churn Rate (%)': '{:.1f}%'
        }), use_container_width=True)
    
    # ------------------------------------------------------------------------
    # TAB 3: SENTIMENT ANALYSIS
    # ------------------------------------------------------------------------
    with tab3:
        st.header("💬 Sentiment Analysis of Open Feedback")
        
        # Sentiment counts
        sent_counts = df_filt['SentimentLabel'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Responses", len(df_filt))
        with col2:
            st.metric("😊 Positive", sent_counts.get('Positive', 0))
        with col3:
            st.metric("😐 Neutral", sent_counts.get('Neutral', 0))
        with col4:
            st.metric("😞 Negative", sent_counts.get('Negative', 0))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            sent_colors = {'Positive': '#2ECC71', 'Neutral': '#F39C12', 'Negative': '#E74C3C'}
            fig6 = px.pie(
                values=sent_counts.values, 
                names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map=sent_colors,
                hole=0.4
            )
            fig6.update_traces(textinfo='percent+label')
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Score Distribution")
            fig7 = px.histogram(
                df_filt, 
                x='SentimentScore', 
                nbins=20,
                color='SentimentLabel',
                color_discrete_map=sent_colors,
                labels={'x': 'Sentiment Score (-1 to +1)', 'y': 'Count'}
            )
            fig7.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig7, use_container_width=True)
        
        st.subheader("📝 Recent Negative Feedback")
        neg_fb = df_filt[df_filt['SentimentLabel'] == 'Negative']['OpenFeedback'].head(5)
        if len(neg_fb) > 0:
            for i, fb in enumerate(neg_fb, 1):
                st.warning(f"{i}. {fb}")
        else:
            st.info("No negative feedback in the filtered data.")
    
    # ------------------------------------------------------------------------
    # TAB 4: CHURN PREDICTION
    # ------------------------------------------------------------------------
    with tab4:
        st.header("🔮 Live Churn Prediction")
        st.markdown("Enter member metrics below to predict their churn probability.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            flow = col1.slider("Flow State Score", 1.0, 10.0, 5.0, 0.5)
            fatigue = col1.slider("Fatigue Score", 1.0, 10.0, 5.0, 0.5)
            sat = col1.slider("Overall Satisfaction", 1.0, 10.0, 5.0, 0.5)
        
        with col2:
            instr = col2.slider("Instructor Rating", 1.0, 5.0, 3.0, 0.5)
            music = col2.slider("Music Volume Rating", 1.0, 5.0, 3.0, 0.5)
            floor_q = col2.slider("Floor Quality Rating", 1.0, 5.0, 3.0, 0.5)
        
        with col3:
            social = col3.slider("Social Atmosphere", 1.0, 5.0, 3.0, 0.5)
            months = col3.slider("Months Enrolled", 1, 24, 6)
        
        if st.button("👉 Predict Churn", type="primary", use_container_width=True):
            input_df = pd.DataFrame([[
                flow, fatigue, sat, instr, music, floor_q, social, months
            ]], columns=model_features)
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            
            # Create result for download
            pred_result = pd.DataFrame({
                'FlowStateScore': [flow],
                'FatigueScore': [fatigue],
                'OverallSatisfaction': [sat],
                'InstructorRating': [instr],
                'MusicVolRating': [music],
                'FloorQualRating': [floor_q],
                'SocialAtmRating': [social],
                'MonthsEnrolled': [months],
                'Predicted_Churn': ['Yes' if pred == 1 else 'No'],
                'Churn_Probability': [f"{prob:.2%}"]
            })
            
            # Download button for prediction
            csv_pred = pred_result.to_csv(index=False)
            st.download_button(
                label="📥 Download This Prediction",
                data=csv_pred,
                file_name="churn_prediction.csv",
                mime="text/csv"
            )
            
            # Display result
            if pred == 1:
                st.error(f"⚠️ **High Churn Risk** | Probability: {prob:.1%}")
                st.info("💡 *Recommended Actions:* Check fatigue patterns, adjust class tempo, schedule 1-on-1 check-in")
            else:
                st.success(f"✅ **Likely to Stay** | Churn Probability: {prob:.1%}")
                st.info("💡 *Recommended Actions:* Continue engagement, consider referral rewards")
        
        # Model Performance Summary
        st.divider()
        st.subheader("📊 Model Performance Summary")
        
        # Calculate confusion matrix on filtered data
        X_filt = df_filt[model_features]
        y_filt = df_filt['Churned']
        y_filt_pred = model.predict(X_filt)
        cm = confusion_matrix(y_filt, y_filt_pred)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("True Negatives", cm[0][0], help="Correctly predicted: Will Stay")
        with col_b:
            st.metric("False Positives", cm[0][1], help="Wrongly predicted: Will Quit")
        with col_c:
            st.metric("False Negatives", cm[1][0], help="Missed: Actually Quit")
        with col_d:
            st.metric("True Positives", cm[1][1], help="Correctly predicted: Will Quit")
        
        # Feature Importance
        st.subheader("🎯 Feature Influence on Churn")
        coefs = pd.DataFrame({
            'Feature': model_features,
            'Importance': model.coef_[0]
        }).sort_values('Importance', key=abs, ascending=False)
        
        fig_coefs = px.bar(
            coefs, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color='Importance',
            color_continuous_scale='RdBu_r'
        )
        fig_coefs.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_coefs, use_container_width=True)
        
        # Bulk Prediction Export
        st.divider()
        st.subheader("📦 Bulk Prediction Export")
        
        if st.button("🔄 Run Bulk Predictions for Filtered Data", use_container_width=True):
            df_filt['Predicted_Churn'] = model.predict(X_filt)
            df_filt['Churn_Probability'] = model.predict_proba(X_filt)[:, 1]
            
            export_cols = [
                'MemberID', 'Location', 'MonthsEnrolled', 'Archetype',
                'OverallSatisfaction', 'Churned', 'Predicted_Churn', 'Churn_Probability'
            ]
            export_df = df_filt[[c for c in export_cols if c in df_filt.columns]].copy()
            export_df['Churn_Probability'] = export_df['Churn_Probability'].apply(lambda x: f"{x:.2%}")
            export_df['Predicted_Churn'] = export_df['Predicted_Churn'].apply(lambda x: 'Yes' if x == 1 else 'No')
            
            csv_bulk = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Bulk Predictions (CSV)",
                data=csv_bulk,
                file_name="pulse_strike_bulk_predictions.csv",
                mime="text/csv"
            )
            
            # Summary of bulk predictions
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                stay_count = len(export_df[export_df['Predicted_Churn'] == 'No'])
                st.metric("Predicted to Stay", stay_count)
            with col_p2:
                churn_count = len(export_df[export_df['Predicted_Churn'] == 'Yes'])
                st.metric("Predicted to Churn", churn_count)
    
    # Footer
    st.divider()
    st.caption("🥊 Pulse Strike Studio Dashboard | Built with Streamlit")


# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()