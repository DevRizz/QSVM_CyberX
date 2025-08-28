import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantum_anomaly.ui import theme
from quantum_anomaly.models.advanced_models import AdvancedModelManager, StreamingAdvancedModels
from quantum_anomaly.orchestrator import Orchestrator

st.set_page_config(page_title="Advanced ML Models", layout="wide", page_icon="🤖")
theme.load_theme_css()
theme.top_navbar(team=["Your Name 1", "Your Name 2", "Your Name 3", "Your Name 4"])
theme.hero(
    "Advanced ML Models & Optimization",
    "Expand your anomaly detection with XGBoost, LightGBM, Neural Networks, and more",
    lottie_url="https://lottie.host/embed/b5c6d7e8-9f1a-2b3c-4d5e-6f7g8h9i0j1k/2L3M4N5O6P.json"
)

# Initialize session state
if "advanced_manager" not in st.session_state:
    st.session_state["advanced_manager"] = AdvancedModelManager()
    st.session_state["advanced_manager"].initialize_models()

if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()

manager = st.session_state["advanced_manager"]
orch = st.session_state["orchestrator"]

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Selection", "⚡ Optimization", "🎯 Training & Results", "📊 Performance Analysis"])

with tab1:
    st.subheader("Available Advanced Models")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Supervised Models")
        supervised_models = ['xgboost', 'lightgbm', 'neural_network', 'random_forest']
        
        for model in supervised_models:
            with st.expander(f"📈 {model.upper().replace('_', ' ')}"):
                if model == 'xgboost':
                    st.markdown("""
                    **XGBoost Classifier**
                    - Gradient boosting with advanced regularization
                    - Excellent for structured/tabular data
                    - Handles missing values automatically
                    - Built-in feature importance
                    
                    **Best for**: High accuracy, feature importance analysis
                    """)
                elif model == 'lightgbm':
                    st.markdown("""
                    **LightGBM Classifier**
                    - Fast gradient boosting with leaf-wise growth
                    - Memory efficient and fast training
                    - Good performance with less overfitting
                    - Categorical feature support
                    
                    **Best for**: Large datasets, fast training
                    """)
                elif model == 'neural_network':
                    st.markdown("""
                    **Multi-layer Perceptron**
                    - Deep learning with multiple hidden layers
                    - Can learn complex non-linear patterns
                    - Adaptive learning rate
                    - Early stopping to prevent overfitting
                    
                    **Best for**: Complex patterns, non-linear relationships
                    """)
                elif model == 'random_forest':
                    st.markdown("""
                    **Random Forest Classifier**
                    - Ensemble of decision trees
                    - Robust to overfitting
                    - Good baseline performance
                    - Feature importance ranking
                    
                    **Best for**: Robust baseline, interpretability
                    """)
    
    with col2:
        st.markdown("### Unsupervised Models")
        
        with st.expander("🔍 ISOLATION FOREST"):
            st.markdown("""
            **Isolation Forest**
            - Unsupervised anomaly detection
            - Isolates anomalies using random splits
            - No need for labeled data
            - Fast and memory efficient
            
            **Best for**: Unlabeled data, pure anomaly detection
            """)
        
        st.markdown("### Streaming Models")
        
        with st.expander("🌊 ADAPTIVE RANDOM FOREST"):
            st.markdown("""
            **Adaptive Random Forest**
            - Online ensemble learning
            - Adapts to concept drift
            - Maintains multiple models
            - Drift detection built-in
            
            **Best for**: Real-time learning, concept drift
            """)

with tab2:
    st.subheader("Hyperparameter Optimization")
    
    # Get labeled data for optimization
    X_lab, y_lab = orch.buffer.labeled()
    
    if len(X_lab) < 50:
        st.warning("Need at least 50 labeled samples for optimization. Label more data first!")
        st.info(f"Currently have {len(X_lab)} labeled samples")
    else:
        st.success(f"Ready for optimization with {len(X_lab)} labeled samples")
        
        # Model selection for optimization
        selected_models = st.multiselect(
            "Select models to optimize:",
            options=list(manager.models.keys()),
            default=['xgboost', 'lightgbm']
        )
        
        # Optimization settings
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.slider("Number of optimization trials", 10, 200, 50)
        with col2:
            optimization_method = st.selectbox(
                "Optimization method",
                ["Optuna (Recommended)", "Random Search", "Grid Search"]
            )
        
        if st.button("🚀 Start Optimization", type="primary"):
            if selected_models:
                progress_bar = st.progress(0)
                results_container = st.container()
                
                optimization_results = {}
                
                for i, model_name in enumerate(selected_models):
                    st.write(f"Optimizing {model_name}...")
                    
                    try:
                        if optimization_method == "Optuna (Recommended)":
                            result = manager.optimize_model_with_optuna(
                                model_name, X_lab, y_lab, n_trials=n_trials
                            )
                        else:
                            st.warning("Other optimization methods not implemented yet. Using Optuna.")
                            result = manager.optimize_model_with_optuna(
                                model_name, X_lab, y_lab, n_trials=n_trials
                            )
                        
                        optimization_results[model_name] = result
                        
                    except Exception as e:
                        st.error(f"Error optimizing {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                # Display results
                with results_container:
                    st.subheader("Optimization Results")
                    
                    results_df = pd.DataFrame([
                        {
                            'Model': name,
                            'Best Score': f"{result['best_score']:.4f}",
                            'Trials': result['n_trials']
                        }
                        for name, result in optimization_results.items()
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best parameters
                    for name, result in optimization_results.items():
                        with st.expander(f"Best parameters for {name}"):
                            st.json(result['best_params'])
            else:
                st.warning("Please select at least one model to optimize")

with tab3:
    st.subheader("Model Training & Results")
    
    X_lab, y_lab = orch.buffer.labeled()
    
    if len(X_lab) < 20:
        st.warning("Need at least 20 labeled samples for training")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### Training Controls")
            
            if st.button("🎯 Train All Models", type="primary"):
                with st.spinner("Training models..."):
                    try:
                        manager.train_optimized_models(X_lab, y_lab)
                        st.success("All models trained successfully!")
                        
                        # Create ensemble
                        ensemble = manager.create_ensemble_model(X_lab, y_lab)
                        if ensemble:
                            st.success("Ensemble model created!")
                        
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
            
            # Model status
            st.markdown("### Model Status")
            for model_name in manager.models:
                status = "✅ Trained" if model_name in manager.trained_models else "⏳ Not trained"
                st.write(f"**{model_name}**: {status}")
        
        with col1:
            st.markdown("### Predictions Comparison")
            
            if manager.trained_models:
                # Get recent data for prediction
                X_recent, _ = orch.buffer.all()
                if len(X_recent) > 0:
                    # Use last 20 samples
                    X_sample = X_recent[-20:] if len(X_recent) >= 20 else X_recent
                    
                    # Get predictions from all models
                    predictions = manager.predict_all_models(X_sample)
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    colors = ['#ff0080', '#00e0ff', '#00ff88', '#ffaa00', '#aa00ff']
                    
                    for i, (model_name, pred) in enumerate(predictions.items()):
                        fig.add_trace(go.Scatter(
                            x=list(range(len(pred))),
                            y=pred,
                            mode='lines+markers',
                            name=model_name,
                            line=dict(color=colors[i % len(colors)])
                        ))
                    
                    fig.update_layout(
                        title="Model Predictions Comparison",
                        xaxis_title="Sample Index",
                        yaxis_title="Anomaly Score/Probability",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction statistics
                    pred_stats = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Mean Score': [np.mean(pred) for pred in predictions.values()],
                        'Std Score': [np.std(pred) for pred in predictions.values()],
                        'Max Score': [np.max(pred) for pred in predictions.values()]
                    })
                    
                    st.dataframe(pred_stats, use_container_width=True)
                else:
                    st.info("No data available for predictions")
            else:
                st.info("Train models first to see predictions")

with tab4:
    st.subheader("Performance Analysis")
    
    if manager.optimization_results:
        # Performance summary
        summary = manager.get_model_performance_summary()
        
        # Create performance comparison chart
        models = list(summary.keys())
        scores = [summary[model]['best_score'] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=scores, 
                  marker_color=['#ff0080', '#00e0ff', '#00ff88', '#ffaa00', '#aa00ff'][:len(models)])
        ])
        
        fig.update_layout(
            title="Model Performance Comparison (Optimization Scores)",
            xaxis_title="Model",
            yaxis_title="Best Score",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("### Detailed Performance Results")
        
        detailed_results = []
        for model_name, results in manager.optimization_results.items():
            detailed_results.append({
                'Model': model_name,
                'Best Score': f"{results['best_score']:.4f}",
                'Trials Completed': results['n_trials'],
                'Parameters Optimized': len(results['best_params'])
            })
        
        results_df = pd.DataFrame(detailed_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Model recommendations
        st.markdown("### 🎯 Model Recommendations")
        
        best_model = max(summary.keys(), key=lambda x: summary[x]['best_score'])
        best_score = summary[best_model]['best_score']
        
        st.success(f"**Best performing model**: {best_model} (Score: {best_score:.4f})")
        
        # Recommendations based on performance
        recommendations = []
        
        if 'xgboost' in summary and summary['xgboost']['best_score'] > 0.85:
            recommendations.append("✅ XGBoost shows excellent performance - recommended for production")
        
        if 'lightgbm' in summary and summary['lightgbm']['best_score'] > 0.85:
            recommendations.append("✅ LightGBM shows excellent performance - good for fast inference")
        
        if 'ensemble' in manager.trained_models:
            recommendations.append("🎯 Ensemble model available - typically provides best overall performance")
        
        if len(recommendations) == 0:
            recommendations.append("⚠️ Consider collecting more labeled data to improve model performance")
        
        for rec in recommendations:
            st.write(rec)
    
    else:
        st.info("Run model optimization first to see performance analysis")
        
        # Show theoretical performance comparison
        st.markdown("### Expected Model Performance")
        
        theoretical_performance = {
            'XGBoost': {'Accuracy': 0.92, 'Speed': 0.8, 'Memory': 0.7, 'Interpretability': 0.8},
            'LightGBM': {'Accuracy': 0.91, 'Speed': 0.95, 'Memory': 0.9, 'Interpretability': 0.8},
            'Neural Network': {'Accuracy': 0.89, 'Speed': 0.6, 'Memory': 0.6, 'Interpretability': 0.3},
            'Random Forest': {'Accuracy': 0.85, 'Speed': 0.7, 'Memory': 0.8, 'Interpretability': 0.9},
            'Isolation Forest': {'Accuracy': 0.82, 'Speed': 0.9, 'Memory': 0.9, 'Interpretability': 0.7}
        }
        
        # Create radar chart
        categories = ['Accuracy', 'Speed', 'Memory', 'Interpretability']
        
        fig = go.Figure()
        
        colors = ['#ff0080', '#00e0ff', '#00ff88', '#ffaa00', '#aa00ff']
        
        for i, (model, metrics) in enumerate(theoretical_performance.items()):
            values = [metrics[cat] for cat in categories]
            values += values[:1]  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Theoretical Model Performance Comparison",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer with recommendations
st.markdown("---")
st.subheader("🚀 Next Steps & Best Practices")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Model Selection**
    - XGBoost: Best overall performance
    - LightGBM: Fastest training
    - Neural Network: Complex patterns
    - Ensemble: Combine strengths
    """)

with col2:
    st.markdown("""
    **⚡ Optimization Tips**
    - Use Optuna for efficient search
    - Start with 50-100 trials
    - Focus on F1-score for imbalanced data
    - Cross-validate results
    """)

with col3:
    st.markdown("""
    **📊 Performance Monitoring**
    - Track model drift over time
    - Retrain when performance drops
    - Use ensemble for robustness
    - Monitor false positive rates
    """)

theme.footer()
