"""
MMA Fight Predictor 2.0 - Main Streamlit Application
Complete web application with data scraping, model training, and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import our custom modules
try:
    from src.scraper import EnhancedMMAScaper, scrape_fresh_data, load_or_scrape_data
    from src.data_processor import MMADataProcessor, process_mma_data
    from src.model_trainer import MMAModelTrainer, train_mma_models, load_trained_models
    from config import STREAMLIT_CONFIG, FILE_PATHS, WEIGHT_CLASSES
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .fighter-card {
        border: 3px solid #4ECDC4;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 5px solid #FF6B6B;
    }
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'data' not in st.session_state:
    st.session_state.data = None

class MMAApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_sidebar()
    
    def setup_sidebar(self):
        """Setup sidebar with controls"""
        st.sidebar.markdown("## ⚙️ Control Panel")
        
        # Data Management Section
        st.sidebar.markdown("### 📊 Data Management")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Scrape Data", help="Scrape fresh UFC data"):
                self.scrape_fresh_data()
        
        with col2:
            if st.button("🔧 Process Data", help="Clean and process scraped data"):
                self.process_data()
        
        # Model Management Section
        st.sidebar.markdown("### 🤖 Model Management")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🏋️ Train Models", help="Train prediction models"):
                self.train_models()
        
        with col2:
            if st.button("📁 Load Models", help="Load existing models"):
                self.load_models()
        
        # Settings
        st.sidebar.markdown("### ⚙️ Settings")
        
        self.num_events = st.sidebar.slider(
            "Events to scrape", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Number of recent events to scrape"
        )
        
        self.show_debug = st.sidebar.checkbox("Debug mode", help="Show debug information")
        
        # Status indicators
        st.sidebar.markdown("### 📊 Status")
        
        # Data status
        data_status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not loaded"
        st.sidebar.markdown(f"**Data:** {data_status}")
        
        # Model status
        model_status = "✅ Loaded" if st.session_state.models_loaded else "❌ Not loaded"
        st.sidebar.markdown(f"**Models:** {model_status}")
        
        # Last updated
        if FILE_PATHS['raw_fights'].exists():
            last_updated = datetime.fromtimestamp(FILE_PATHS['raw_fights'].stat().st_mtime)
            st.sidebar.markdown(f"**Last updated:** {last_updated.strftime('%Y-%m-%d %H:%M')}")
    
    def scrape_fresh_data(self):
        """Scrape fresh data from UFC sources"""
        with st.spinner("🕷️ Scraping fresh UFC data..."):
            try:
                data = scrape_fresh_data(num_events=self.num_events, save=True)
                st.session_state.data = data
                st.session_state.data_loaded = True
                
                st.sidebar.success(f"✅ Scraped {len(data['fights'])} fights!")
                
            except Exception as e:
                st.sidebar.error(f"❌ Scraping failed: {str(e)}")
                if self.show_debug:
                    st.sidebar.exception(e)
    
    def process_data(self):
        """Process and clean the scraped data"""
        if not st.session_state.data_loaded:
            st.sidebar.warning("⚠️ Load data first!")
            return
        
        with st.spinner("🔧 Processing data..."):
            try:
                processed_data = process_mma_data()
                st.session_state.data = processed_data
                
                st.sidebar.success("✅ Data processed successfully!")
                
            except Exception as e:
                st.sidebar.error(f"❌ Processing failed: {str(e)}")
                if self.show_debug:
                    st.sidebar.exception(e)
    
    def train_models(self):
        """Train machine learning models"""
        with st.spinner("🏋️ Training models... This may take a few minutes."):
            try:
                report = train_mma_models()
                
                # Load the trained models
                trainer = load_trained_models()
                st.session_state.trainer = trainer
                st.session_state.models_loaded = True
                
                st.sidebar.success(f"✅ Models trained! Best: {report['best_model']}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Training failed: {str(e)}")
                if self.show_debug:
                    st.sidebar.exception(e)
    
    def load_models(self):
        """Load existing trained models"""
        try:
            trainer = load_trained_models()
            st.session_state.trainer = trainer
            st.session_state.models_loaded = True
            
            st.sidebar.success("✅ Models loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Loading failed: {str(e)}")
            if self.show_debug:
                st.sidebar.exception(e)
    
    def load_data_if_available(self):
        """Load data if available"""
        if not st.session_state.data_loaded:
            try:
                data = load_or_scrape_data(num_events=10)  # Load smaller dataset initially
                st.session_state.data = data
                st.session_state.data_loaded = True
            except:
                pass  # Will handle in the UI
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">🥊 MMA Fight Predictor 2.0</h1>', unsafe_allow_html=True)
        
        # Load data if available
        self.load_data_if_available()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Fight Predictor", 
            "📊 Data Overview", 
            "📈 Analytics", 
            "🤖 Model Performance",
            "ℹ️ About"
        ])
        
        with tab1:
            self.show_predictor_tab()
        
        with tab2:
            self.show_data_overview_tab()
        
        with tab3:
            self.show_analytics_tab()
        
        with tab4:
            self.show_model_performance_tab()
        
        with tab5:
            self.show_about_tab()
    
    def show_predictor_tab(self):
        """Main fight prediction interface"""
        st.header("🥊 Fight Prediction Engine")
        
        if not st.session_state.data_loaded:
            st.markdown('<div class="warning-message">⚠️ No data loaded. Please scrape or load data first using the sidebar.</div>', unsafe_allow_html=True)
            return
        
        if not st.session_state.models_loaded:
            st.markdown('<div class="warning-message">⚠️ No models loaded. Please train or load models first using the sidebar.</div>', unsafe_allow_html=True)
            return
        
        # Get fighter list
        fighters_df = st.session_state.data.get('fighters', pd.DataFrame())
        if fighters_df.empty:
            st.error("No fighter data available")
            return
        
        fighter_names = sorted(fighters_df['name'].unique())
        
        # Fighter selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="fighter-card">', unsafe_allow_html=True)
            st.subheader("🔴 Fighter 1")
            fighter1 = st.selectbox(
                "Select Fighter 1:",
                options=fighter_names,
                key="fighter1",
                help="Choose the first fighter"
            )
            
            if fighter1:
                fighter1_info = fighters_df[fighters_df['name'] == fighter1].iloc[0]
                col1a, col1b = st.columns(2)
                with col1a:
                    st.metric("Record", f"{fighter1_info.get('wins_total', 0)}-{fighter1_info.get('losses_total', 0)}")
                with col1b:
                    st.metric("Weight Class", fighter1_info.get('weight_class', 'Unknown'))
                
                st.write(f"**Height:** {fighter1_info.get('height', 'Unknown')}")
                st.write(f"**Reach:** {fighter1_info.get('reach', 'Unknown')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="fighter-card">', unsafe_allow_html=True)
            st.subheader("🔵 Fighter 2")
            fighter2 = st.selectbox(
                "Select Fighter 2:",
                options=fighter_names,
                key="fighter2",
                help="Choose the second fighter"
            )
            
            if fighter2:
                fighter2_info = fighters_df[fighters_df['name'] == fighter2].iloc[0]
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Record", f"{fighter2_info.get('wins_total', 0)}-{fighter2_info.get('losses_total', 0)}")
                with col2b:
                    st.metric("Weight Class", fighter2_info.get('weight_class', 'Unknown'))
                
                st.write(f"**Height:** {fighter2_info.get('height', 'Unknown')}")
                st.write(f"**Reach:** {fighter2_info.get('reach', 'Unknown')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = list(st.session_state.trainer.models.keys())
        selected_model = st.selectbox(
            "Choose prediction model:",
            options=available_models,
            index=available_models.index(st.session_state.trainer.best_model_name) if st.session_state.trainer.best_model_name in available_models else 0,
            help="Select which trained model to use for prediction"
        )
        
        # Prediction button
        if st.button("🎲 Predict Fight Outcome", type="primary", use_container_width=True):
            if fighter1 and fighter2 and fighter1 != fighter2:
                self.make_prediction(fighter1, fighter2, selected_model)
            else:
                st.error("Please select two different fighters")
    
    def make_prediction(self, fighter1: str, fighter2: str, model_name: str):
        """Make fight prediction and display results"""
        try:
            # Get fighter stats
            stats_df = st.session_state.data.get('stats', pd.DataFrame())
            if stats_df.empty:
                st.error("No statistics data available")
                return
            
            # Aggregate fighter stats
            f1_stats = stats_df[stats_df['fighter_name'] == fighter1]
            f2_stats = stats_df[stats_df['fighter_name'] == fighter2]
            
            if f1_stats.empty or f2_stats.empty:
                st.error("Insufficient statistics for selected fighters")
                return
            
            # Calculate average stats
            f1_avg = f1_stats.select_dtypes(include=[np.number]).mean().to_dict()
            f2_avg = f2_stats.select_dtypes(include=[np.number]).mean().to_dict()
            
            # Make prediction
            prediction = st.session_state.trainer.predict_fight(
                f1_avg, f2_avg, model_name
            )
            
            # Display results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("🏆 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                winner = fighter1 if prediction['predicted_winner'] == 1 else fighter2
                st.metric("🎯 Predicted Winner", winner)
            with col2:
                st.metric("🎲 Confidence", f"{prediction['confidence']:.1f}%")
            with col3:
                st.metric("🤖 Model Used", model_name.replace('_', ' ').title())
            
            st.subheader("📊 Win Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"🔴 {fighter1}", f"{prediction['fighter1_win_prob']*100:.1f}%")
            with col2:
                st.metric(f"🔵 {fighter2}", f"{prediction['fighter2_win_prob']*100:.1f}%")
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    name='Win Probability',
                    x=[fighter1, fighter2],
                    y=[prediction['fighter1_win_prob']*100, prediction['fighter2_win_prob']*100],
                    marker_color=['#FF6B6B', '#4ECDC4'],
                    text=[f"{prediction['fighter1_win_prob']*100:.1f}%", f"{prediction['fighter2_win_prob']*100:.1f}%"],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Win Probability Comparison",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Fighter comparison chart
            self.show_fighter_comparison(f1_avg, f2_avg, fighter1, fighter2)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            if hasattr(self, 'show_debug') and self.show_debug:
                st.exception(e)
    
    def show_fighter_comparison(self, f1_stats: dict, f2_stats: dict, fighter1: str, fighter2: str):
        """Show detailed fighter comparison"""
        st.subheader("⚔️ Fighter Comparison")
        
        # Select key stats for comparison
        comparison_stats = [
            'sig_strikes_landed', 'takedowns_landed', 'knockdowns',
            'striking_accuracy', 'takedown_accuracy', 'submission_attempts'
        ]
        
        fighter1_values = []
        fighter2_values = []
        stat_labels = []
        
        for stat in comparison_stats:
            if stat in f1_stats and stat in f2_stats:
                fighter1_values.append(f1_stats[stat])
                fighter2_values.append(f2_stats[stat])
                stat_labels.append(stat.replace('_', ' ').title())
        
        if fighter1_values and fighter2_values:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=fighter1_values,
                theta=stat_labels,
                fill='toself',
                name=fighter1,
                line_color='#FF6B6B'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=fighter2_values,
                theta=stat_labels,
                fill='toself',
                name=fighter2,
                line_color='#4ECDC4'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(max(fighter1_values), max(fighter2_values)) * 1.1]
                    )
                ),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_data_overview_tab(self):
        """Data overview and statistics"""
        st.header("📊 Data Overview")
        
        if not st.session_state.data_loaded:
            st.info("No data loaded. Please scrape or load data first.")
            return
        
        data = st.session_state.data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fights_count = len(data.get('fights', []))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🥊 Total Fights", fights_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            fighters_count = len(data.get('fighters', []))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("👤 Total Fighters", fighters_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            stats_count = len(data.get('stats', []))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Stats Records", stats_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            weight_classes = len(data.get('fighters', pd.DataFrame()).get('weight_class', pd.Series()).unique())
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚖️ Weight Classes", weight_classes)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🆕 Recent Fights")
            fights_df = data.get('fights', pd.DataFrame())
            if not fights_df.empty:
                display_fights = fights_df[['event_name', 'fighter1', 'fighter2', 'winner', 'method']].head(10)
                st.dataframe(display_fights, use_container_width=True)
            else:
                st.info("No fight data available")
        
        with col2:
            st.subheader("👥 Fighter Roster")
            fighters_df = data.get('fighters', pd.DataFrame())
            if not fighters_df.empty:
                display_fighters = fighters_df[['name', 'weight_class', 'wins_total', 'losses_total']].head(10)
                st.dataframe(display_fighters, use_container_width=True)
            else:
                st.info("No fighter data available")
        
        # Data quality indicators
        st.subheader("📋 Data Quality")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not fights_df.empty:
                complete_fights = fights_df.dropna().shape[0]
                completeness = (complete_fights / len(fights_df)) * 100
                st.metric("Fight Data Completeness", f"{completeness:.1f}%")
        
        with col2:
            if not fighters_df.empty:
                complete_fighters = fighters_df.dropna().shape[0]
                completeness = (complete_fighters / len(fighters_df)) * 100
                st.metric("Fighter Data Completeness", f"{completeness:.1f}%")
        
        with col3:
            if FILE_PATHS['raw_fights'].exists():
                last_update = datetime.fromtimestamp(FILE_PATHS['raw_fights'].stat().st_mtime)
                days_old = (datetime.now() - last_update).days
                st.metric("Data Age", f"{days_old} days")
    
    def show_analytics_tab(self):
        """Analytics and visualizations"""
        st.header("📈 Fight Analytics")
        
        if not st.session_state.data_loaded:
            st.info("No data loaded. Please scrape or load data first.")
            return
        
        data = st.session_state.data
        fights_df = data.get('fights', pd.DataFrame())
        fighters_df = data.get('fighters', pd.DataFrame())
        stats_df = data.get('stats', pd.DataFrame())
        
        if fights_df.empty:
            st.info("No fight data available for analytics")
            return
        
        # Weight class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚖️ Fights by Weight Class")
            if 'weight_class' in fights_df.columns:
                weight_class_counts = fights_df['weight_class'].value_counts()
                fig = px.pie(
                    values=weight_class_counts.values,
                    names=weight_class_counts.index,
                    title="Distribution of Fights by Weight Class"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥊 Fight Methods")
            if 'method' in fights_df.columns:
                method_counts = fights_df['method'].value_counts()
                fig = px.bar(
                    x=method_counts.values,
                    y=method_counts.index,
                    orientation='h',
                    title="Fight Finish Methods"
                )
                fig.update_layout(yaxis_title="Method", xaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        if not stats_df.empty:
            st.subheader("📊 Performance Metrics")
            
            # Fighter performance analysis
            fighter_performance = stats_df.groupby('fighter_name').agg({
                'striking_accuracy': 'mean',
                'takedown_accuracy': 'mean',
                'sig_strikes_landed': 'mean',
                'knockdowns': 'sum'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Strikers (Accuracy)")
                top_strikers = fighter_performance.nlargest(10, 'striking_accuracy')
                fig = px.bar(
                    top_strikers,
                    x='striking_accuracy',
                    y='fighter_name',
                    orientation='h',
                    title="Top 10 Striking Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🤼 Top Grapplers (Takedown Accuracy)")
                top_grapplers = fighter_performance.nlargest(10, 'takedown_accuracy')
                fig = px.bar(
                    top_grapplers,
                    x='takedown_accuracy',
                    y='fighter_name',
                    orientation='h',
                    title="Top 10 Takedown Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Performance Correlations")
            numeric_stats = stats_df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_stats.corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Fighter Statistics Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance_tab(self):
        """Model performance and evaluation"""
        st.header("🤖 Model Performance")
        
        if not st.session_state.models_loaded:
            st.info("No models loaded. Please train or load models first.")
            return
        
        trainer = st.session_state.trainer
        
        # Model comparison
        st.subheader("📊 Model Comparison")
        
        performance_data = []
        for model_name, metrics in trainer.model_performance.items():
            performance_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'AUC': f"{metrics['auc']:.3f}",
                'CV Mean': f"{metrics['cv_mean']:.3f}",
                'CV Std': f"{metrics['cv_std']:.3f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Best model highlight
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.write(f"🏆 **Best Model:** {trainer.best_model_name.replace('_', ' ').title()}")
        best_accuracy = trainer.model_performance[trainer.best_model_name]['accuracy']
        st.write(f"🎯 **Accuracy:** {best_accuracy:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            models = list(trainer.model_performance.keys())
            accuracies = [trainer.model_performance[model]['accuracy'] for model in models]
            
            fig = px.bar(
                x=[model.replace('_', ' ').title() for model in models],
                y=accuracies,
                title="Model Accuracy Comparison",
                labels={'x': 'Model', 'y': 'Accuracy'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cross-validation scores
            cv_means = [trainer.model_performance[model]['cv_mean'] for model in models]
            cv_stds = [trainer.model_performance[model]['cv_std'] for model in models]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='CV Mean',
                x=[model.replace('_', ' ').title() for model in models],
                y=cv_means,
                error_y=dict(type='data', array=cv_stds)
            ))
            
            fig.update_layout(
                title="Cross-Validation Performance",
                xaxis_title="Model",
                yaxis_title="CV Score",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for applicable models)
        st.subheader("🎯 Feature Importance")
        
        feature_importance_models = [name for name in trainer.models.keys() 
                                   if hasattr(trainer.models[name], 'feature_importances_')]
        
        if feature_importance_models:
            selected_model = st.selectbox(
                "Select model for feature importance:",
                feature_importance_models,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            model = trainer.models[selected_model]
            importances = model.feature_importances_
            feature_names = trainer.feature_columns
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Features - {selected_model.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_about_tab(self):
        """About page with project information"""
        st.header("ℹ️ About MMA Fight Predictor 2.0")
        
        st.markdown("""
        ## 🥊 Welcome to the Ultimate MMA Fight Predictor!
        
        This application represents a complete evolution from the original fight predictor, 
        featuring real-time data scraping, advanced machine learning, and an intuitive interface.
        
        ### ✨ Key Features
        
        - **🕷️ Live Data Scraping**: Fresh fighter statistics and recent fight results
        - **🤖 Multiple ML Models**: Gradient Boosting, Random Forest, Neural Networks, and more
        - **📊 Real-time Analytics**: Interactive visualizations and performance metrics
        - **🎯 Accurate Predictions**: Sophisticated feature engineering and model ensemble
        - **📱 Modern Interface**: Built with Streamlit for optimal user experience
        
        ### 🛠️ Technology Stack
        
        - **Data Collection**: requests, lxml, BeautifulSoup
        - **Machine Learning**: scikit-learn, pandas, numpy
        - **Visualization**: plotly, streamlit
        - **Deployment**: Streamlit Cloud
        
        ### 📊 Data Sources
        
        - **UFC.com**: Official fighter statistics and fight results
        - **Sherdog.com**: Comprehensive fighter records and fight history
        - **UFC Stats**: Detailed performance metrics
        
        ### 🚀 How to Use
        
        1. **Load Data**: Use the sidebar to scrape fresh data or load existing data
        2. **Train Models**: Process the data and train machine learning models
        3. **Make Predictions**: Select two fighters and get AI-powered predictions
        4. **Explore Analytics**: Dive into fight statistics and trends
        
        ### 🎯 Prediction Methodology
        
        Our prediction system analyzes dozens of factors including:
        - Historical fight performance
        - Striking and grappling statistics
        - Physical attributes
        - Recent form and trends
        - Head-to-head comparisons
        
        ### 📈 Model Performance
        
        The system typically achieves 65-75% accuracy on fight predictions,
        which is competitive with expert human predictions and betting markets.
        
        ### 🔮 Future Enhancements
        
        - Real-time odds integration
        - Injury and training camp analysis
        - Style matchup analysis
        - Historical betting performance tracking
        
        ---
        
        **Built with ❤️ for MMA fans and data enthusiasts**
        
        *Disclaimer: This tool is for entertainment and educational purposes. 
        Please gamble responsibly and never bet more than you can afford to lose.*
        """)
        
        # Version and stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Version", "2.0.0")
        
        with col2:
            if st.session_state.data_loaded:
                total_fights = len(st.session_state.data.get('fights', []))
                st.metric("Fights Analyzed", total_fights)
            else:
                st.metric("Fights Analyzed", "N/A")
        
        with col3:
            if st.session_state.models_loaded:
                models_count = len(st.session_state.trainer.models)
                st.metric("Models Trained", models_count)
            else:
                st.metric("Models Trained", "N/A")

# Main application
def main():
    """Main application entry point"""
    app = MMAApp()
    app.run()

if __name__ == "__main__":
    main()