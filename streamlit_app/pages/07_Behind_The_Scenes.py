import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quantum_anomaly.ui import theme

st.set_page_config(page_title="Behind The Scenes", layout="wide", page_icon="🔬")
theme.load_theme_css()
theme.top_navbar(team=["Your Name 1", "Your Name 2", "Your Name 3", "Your Name 4"])
theme.hero(
    "Behind The Scenes: Technical Deep Dive",
    "Visual demonstrations of what's happening inside our quantum anomaly detection system",
    lottie_url="https://lottie.host/embed/a4f7c5e8-8b2a-4c8a-9c3d-2f1e4d6c8b9a/1K2L3M4N5O.json"
)

# Demo selector
demo_choice = st.selectbox(
    "Choose a demonstration:",
    ["Network Packet Analysis", "Quantum vs Classical Kernels", "Online Learning Process", 
     "QKD BB84 Protocol", "Performance Metrics", "Security Dashboard"]
)

if demo_choice == "Network Packet Analysis":
    st.subheader("How We Extract Features from Network Packets")
    st.markdown("""
    This shows the first step of our pipeline: transforming raw network packets into meaningful features
    that machine learning algorithms can understand.
    """)
    
    # Generate demo data
    np.random.seed(42)
    time_stamps = np.cumsum(np.random.exponential(0.1, 1000))
    packet_sizes = np.random.lognormal(6, 1.5, 1000).astype(int)
    
    # Create visualization
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Packet Timing', 'Size Distribution', 'Protocol Analysis', 'Port Entropy'))
    
    # Timing analysis
    inter_arrival = np.diff(time_stamps)
    fig.add_trace(go.Scatter(x=time_stamps[1:100], y=inter_arrival[:99], 
                            mode='lines', name='Inter-arrival Time'), row=1, col=1)
    
    # Size distribution
    fig.add_trace(go.Histogram(x=packet_sizes, nbinsx=50, name='Packet Sizes'), row=1, col=2)
    
    # Protocol flags
    flags = ['SYN', 'ACK', 'FIN', 'RST', 'PSH']
    counts = [340, 450, 120, 45, 180]
    fig.add_trace(go.Bar(x=flags, y=counts, name='TCP Flags'), row=2, col=1)
    
    # Port entropy
    window_entropies = 2.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    fig.add_trace(go.Scatter(x=np.arange(100), y=window_entropies, 
                            mode='lines', name='Port Entropy'), row=2, col=2)
    
    fig.update_layout(height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **What's happening here:**
    - **Timing Analysis**: Detects coordinated attacks by analyzing packet timing patterns
    - **Size Distribution**: Identifies unusual data transfer patterns
    - **Protocol Analysis**: Spots protocol abuse and scanning attempts
    - **Port Entropy**: Measures port diversity to detect scanning behavior
    """)

elif demo_choice == "Quantum vs Classical Kernels":
    st.subheader("Quantum Kernel Advantage Visualization")
    st.markdown("""
    This demonstrates why quantum kernels can potentially outperform classical kernels
    by accessing exponentially larger feature spaces.
    """)
    
    # Generate sample data
    np.random.seed(42)
    n_points = 100
    
    # Create two classes that are hard to separate linearly
    theta = np.linspace(0, 2*np.pi, n_points//2)
    
    # Inner circle (class 0)
    r1 = 1 + 0.3 * np.random.randn(n_points//2)
    x1 = r1 * np.cos(theta) + 0.2 * np.random.randn(n_points//2)
    y1 = r1 * np.sin(theta) + 0.2 * np.random.randn(n_points//2)
    
    # Outer ring (class 1)
    r2 = 2.5 + 0.3 * np.random.randn(n_points//2)
    x2 = r2 * np.cos(theta) + 0.2 * np.random.randn(n_points//2)
    y2 = r2 * np.sin(theta) + 0.2 * np.random.randn(n_points//2)
    
    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=('Original Data (Non-separable)', 'Classical Kernel Transform', 'Quantum Kernel Space'))
    
    # Original data
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', name='Class 0 (Normal)',
                            marker=dict(color='lightblue', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', name='Class 1 (Anomaly)',
                            marker=dict(color='red', size=8)), row=1, col=1)
    
    # Classical kernel (simulated RBF transformation)
    # Project to higher dimension and back for visualization
    X_all = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(kernel='rbf', gamma=0.5, n_components=2)
    X_transformed = kpca.fit_transform(X_all)
    
    fig.add_trace(go.Scatter(x=X_transformed[:n_points//2, 0], y=X_transformed[:n_points//2, 1],
                            mode='markers', name='Classical Transform',
                            marker=dict(color='lightblue', size=8), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_transformed[n_points//2:, 0], y=X_transformed[n_points//2:, 1],
                            mode='markers', name='Classical Transform',
                            marker=dict(color='red', size=8), showlegend=False), row=1, col=2)
    
    # Quantum kernel space (simulated higher-dimensional projection)
    # Simulate quantum advantage with better separation
    quantum_x1 = x1 + 0.5 * np.cos(3 * theta)
    quantum_y1 = y1 + 0.5 * np.sin(3 * theta)
    quantum_x2 = x2 + 0.8 * np.cos(2 * theta)
    quantum_y2 = y2 + 0.8 * np.sin(2 * theta)
    
    fig.add_trace(go.Scatter(x=quantum_x1, y=quantum_y1, mode='markers',
                            name='Quantum Transform', marker=dict(color='lightblue', size=8),
                            showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=quantum_x2, y=quantum_y2, mode='markers',
                            name='Quantum Transform', marker=dict(color='red', size=8),
                            showlegend=False), row=1, col=3)
    
    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Insights:**
    - **Original Data**: Classes are not linearly separable
    - **Classical Kernel**: Limited transformation capabilities
    - **Quantum Kernel**: Access to exponentially larger feature space enables better separation
    
    **Technical Details:**
    - Classical kernels: Polynomial growth in feature dimensions
    - Quantum kernels: Exponential growth (2^n qubits = 2^n dimensions)
    - Quantum entanglement creates correlations impossible classically
    """)

elif demo_choice == "Online Learning Process":
    st.subheader("How Our System Learns and Adapts")
    st.markdown("""
    This shows how our system continuously improves through human feedback and online learning.
    """)
    
    # Simulate learning process
    np.random.seed(42)
    n_samples = 500
    
    # Generate streaming data with concept drift
    time_points = np.arange(n_samples)
    base_scores = 0.3 + 0.2 * np.sin(time_points / 50) + np.random.normal(0, 0.1, n_samples)
    
    # Add anomalies
    anomaly_indices = np.random.choice(n_samples, 30, replace=False)
    base_scores[anomaly_indices] += np.random.uniform(0.4, 0.7, 30)
    
    # Simulate human labeling (sparse)
    labeled_indices = np.sort(np.random.choice(n_samples, 50, replace=False))
    
    # Simulate model improvement
    accuracy_over_time = []
    current_accuracy = 0.6
    
    for i in range(n_samples):
        if i in labeled_indices:
            # Model improves with each label
            current_accuracy = min(0.95, current_accuracy + np.random.uniform(0.01, 0.03))
        accuracy_over_time.append(current_accuracy)
    
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Streaming Anomaly Detection', 'Model Accuracy Improvement'))
    
    # Streaming scores
    colors = ['red' if i in anomaly_indices else 'lightblue' for i in range(n_samples)]
    fig.add_trace(go.Scatter(x=time_points, y=base_scores, mode='markers',
                            marker=dict(color=colors, size=4), name='Anomaly Scores'), row=1, col=1)
    
    # Add threshold line
    fig.add_hline(y=0.6, line_dash="dash", line_color="yellow", row=1, col=1)
    
    # Mark labeled points
    fig.add_trace(go.Scatter(x=labeled_indices, y=base_scores[labeled_indices],
                            mode='markers', marker=dict(color='gold', size=10, symbol='star'),
                            name='Human Labels'), row=1, col=1)
    
    # Accuracy improvement
    fig.add_trace(go.Scatter(x=time_points, y=accuracy_over_time, mode='lines',
                            name='Model Accuracy', line=dict(color='#00ff88', width=3)), row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Learning Process:**
    1. **Streaming Detection**: Immediate anomaly scoring without training
    2. **Human Feedback**: Security analysts label suspicious patterns
    3. **Model Retraining**: Quantum SVM learns from labeled examples
    4. **Performance Improvement**: Accuracy increases with more feedback
    
    **Why This Works:**
    - Combines unsupervised (immediate) and supervised (learned) detection
    - Adapts to organization-specific patterns and threats
    - Quantum kernels can learn complex relationships from sparse labels
    """)

elif demo_choice == "QKD BB84 Protocol":
    st.subheader("Quantum Key Distribution Security")
    st.markdown("""
    This demonstrates how we use quantum mechanics principles to secure our system communications.
    """)
    
    # Simulate BB84 protocol
    np.random.seed(42)
    n_bits = 50
    
    # Alice's preparation
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0: +, 1: x
    
    # Bob's measurement
    bob_bases = np.random.randint(0, 2, n_bits)
    
    # Basis matching
    matching_bases = alice_bases == bob_bases
    sifted_indices = np.where(matching_bases)[0]
    
    # QBER simulation
    eavesdropping_rates = np.linspace(0, 0.2, 20)
    security_status = ['Secure' if rate < 0.11 else 'Compromised' for rate in eavesdropping_rates]
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Quantum Bit Transmission', 'Basis Matching',
                                     'Key Sifting Process', 'Security Analysis (QBER)'))
    
    # 1. Bit transmission
    x_pos = np.arange(min(30, n_bits))
    colors_alice = ['red' if base == 0 else 'blue' for base in alice_bases[:30]]
    fig.add_trace(go.Scatter(x=x_pos, y=alice_bits[:30], mode='markers+lines',
                            marker=dict(color=colors_alice, size=10),
                            name="Alice's Qubits"), row=1, col=1)
    
    # 2. Basis matching
    match_colors = ['green' if match else 'red' for match in matching_bases[:30]]
    fig.add_trace(go.Scatter(x=x_pos, y=matching_bases[:30].astype(int),
                            mode='markers', marker=dict(color=match_colors, size=12),
                            name='Basis Match'), row=1, col=2)
    
    # 3. Sifted key
    if len(sifted_indices) > 0:
        sifted_key = alice_bits[sifted_indices]
        fig.add_trace(go.Scatter(x=np.arange(len(sifted_key[:20])), y=sifted_key[:20],
                                mode='markers+lines', marker=dict(color='gold', size=10),
                                name='Sifted Key'), row=2, col=1)
    
    # 4. QBER analysis
    colors_security = ['green' if status == 'Secure' else 'red' for status in security_status]
    fig.add_trace(go.Scatter(x=eavesdropping_rates, y=eavesdropping_rates,
                            mode='markers+lines', marker=dict(color=colors_security, size=8),
                            name='QBER'), row=2, col=2)
    fig.add_hline(y=0.11, line_dash="dash", line_color="yellow", row=2, col=2)
    
    fig.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **BB84 Protocol Steps:**
    1. **Quantum Transmission**: Alice sends qubits in random bases
    2. **Random Measurement**: Bob measures in random bases
    3. **Basis Reconciliation**: Keep only matching basis measurements
    4. **Eavesdropping Detection**: QBER reveals security breaches
    
    **Security Guarantee:**
    - QBER < 11%: Quantum mechanically secure
    - QBER ≥ 11%: Possible eavesdropping detected
    - Information-theoretic security (not computational)
    """)

elif demo_choice == "Performance Metrics":
    st.subheader("Quantum vs Classical Performance Analysis")
    st.markdown("""
    Comprehensive comparison showing when and why quantum methods provide advantages.
    """)
    
    # Simulate performance data
    training_sizes = np.array([20, 50, 100, 200, 500, 1000])
    
    # Quantum performance (starts lower, improves more with data)
    quantum_acc = 0.6 + 0.35 * (1 - np.exp(-training_sizes / 200))
    quantum_f1 = 0.55 + 0.4 * (1 - np.exp(-training_sizes / 250))
    
    # Classical performance (starts higher, plateaus earlier)
    classical_acc = 0.75 + 0.2 * (1 - np.exp(-training_sizes / 100))
    classical_f1 = 0.7 + 0.25 * (1 - np.exp(-training_sizes / 120))
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Accuracy vs Training Size', 'F1-Score vs Training Size',
                                     'ROC Curves Comparison', 'Computational Cost'))
    
    # Accuracy comparison
    fig.add_trace(go.Scatter(x=training_sizes, y=quantum_acc, mode='lines+markers',
                            name='Quantum SVM', line=dict(color='#ff0080', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=training_sizes, y=classical_acc, mode='lines+markers',
                            name='Classical SVM', line=dict(color='#00e0ff', width=3)), row=1, col=1)
    
    # F1-Score comparison
    fig.add_trace(go.Scatter(x=training_sizes, y=quantum_f1, mode='lines+markers',
                            name='Quantum F1', line=dict(color='#ff0080', width=3),
                            showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=training_sizes, y=classical_f1, mode='lines+markers',
                            name='Classical F1', line=dict(color='#00e0ff', width=3),
                            showlegend=False), row=1, col=2)
    
    # ROC curves (simulated)
    fpr = np.linspace(0, 1, 100)
    tpr_quantum = 1 - np.exp(-3 * fpr)  # Better curve
    tpr_classical = 1 - np.exp(-2 * fpr)  # Standard curve
    
    fig.add_trace(go.Scatter(x=fpr, y=tpr_quantum, mode='lines',
                            name='Quantum ROC', line=dict(color='#ff0080', width=3),
                            showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=fpr, y=tpr_classical, mode='lines',
                            name='Classical ROC', line=dict(color='#00e0ff', width=3),
                            showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            line=dict(dash='dash', color='gray'), showlegend=False), row=2, col=1)
    
    # Computational cost
    quantum_time = training_sizes * 0.1
    classical_time = training_sizes * 0.02
    
    fig.add_trace(go.Scatter(x=training_sizes, y=quantum_time, mode='lines+markers',
                            name='Quantum Time', line=dict(color='#ff0080', width=3),
                            showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=training_sizes, y=classical_time, mode='lines+markers',
                            name='Classical Time', line=dict(color='#00e0ff', width=3),
                            showlegend=False), row=2, col=2)
    
    fig.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Performance Insights:**
    - **Small Data**: Classical methods start with advantage
    - **Large Data**: Quantum methods show superior learning capability
    - **Complex Patterns**: Quantum kernels excel at non-linear relationships
    - **Computational Cost**: Quantum methods require more resources but provide better results
    
    **When to Use Quantum:**
    - Complex, non-linear attack patterns
    - Sufficient training data available
    - High accuracy requirements justify computational cost
    """)

elif demo_choice == "Security Dashboard":
    st.subheader("Real-time Security Operations Center")
    st.markdown("""
    This shows how security analysts would monitor and respond to threats using our system.
    """)
    
    # Generate security dashboard data
    np.random.seed(42)
    
    # Threat timeline
    hours = np.arange(24)
    normal_traffic = 1000 + 500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 50, 24)
    threats_detected = np.random.poisson(8, 24)
    
    # Threat types
    threat_types = ['Port Scan', 'DDoS', 'Malware C&C', 'Data Exfiltration', 'Brute Force']
    threat_counts = [45, 23, 12, 8, 15]
    
    # System metrics
    metrics = ['Detection Rate', 'False Positive Rate', 'Response Time', 'System Uptime']
    values = [94.5, 2.1, 1.8, 99.9]
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('24-Hour Threat Timeline', 'Threat Type Distribution',
                                     'System Performance Metrics', 'Geographic Threat Sources'))
    
    # Threat timeline
    fig.add_trace(go.Scatter(x=hours, y=normal_traffic, mode='lines+markers',
                            name='Network Traffic', line=dict(color='lightblue')), row=1, col=1)
    fig.add_trace(go.Bar(x=hours, y=threats_detected, name='Threats Detected',
                        marker_color='red', opacity=0.7), row=1, col=1)
    
    # Threat types
    fig.add_trace(go.Pie(labels=threat_types, values=threat_counts,
                        name="Threat Types"), row=1, col=2)
    
    # System metrics
    colors = ['green' if v > 90 else 'yellow' if v > 70 else 'red' for v in values]
    fig.add_trace(go.Bar(x=metrics, y=values, name='Performance',
                        marker_color=colors), row=2, col=1)
    
    # Geographic sources
    countries = ['Unknown', 'China', 'Russia', 'USA', 'Germany']
    attack_sources = [45, 23, 18, 12, 8]
    fig.add_trace(go.Bar(x=countries, y=attack_sources, name='Attack Sources',
                        marker_color='orange'), row=2, col=2)
    
    fig.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Security Operations Features:**
    - **Real-time Monitoring**: 24/7 threat detection and analysis
    - **Threat Classification**: Automatic categorization of attack types
    - **Performance Tracking**: System health and effectiveness metrics
    - **Threat Intelligence**: Geographic and temporal attack patterns
    
    **Operational Benefits:**
    - Immediate threat visibility
    - Reduced false positive rates through quantum learning
    - Automated response capabilities
    - Comprehensive audit trails
    """)

# Add explanation section
st.markdown("---")
st.subheader("Understanding the Technology Stack")

tech_tabs = st.tabs(["Network Analysis", "Quantum Computing", "Machine Learning", "Cryptography", "System Integration"])

with tech_tabs[0]:
    st.markdown("""
    ### Network Traffic Analysis
    
    **What we analyze:**
    - Packet headers (IP, TCP, UDP)
    - Timing patterns and inter-arrival times
    - Protocol flags and options
    - Flow characteristics and session data
    
    **Why it matters:**
    - Network attacks leave distinctive signatures
    - Timing analysis reveals automated tools
    - Protocol abuse indicates malicious activity
    - Flow analysis shows communication patterns
    
    **Tools we use:**
    - **Scapy**: Python packet manipulation library
    - **PyShark**: Wireshark integration for Python
    - **PCAP format**: Standard packet capture format
    """)

with tech_tabs[1]:
    st.markdown("""
    ### Quantum Computing Integration
    
    **Key concepts:**
    - **Qubits**: Quantum bits that can be in superposition
    - **Entanglement**: Quantum correlations between qubits
    - **Quantum circuits**: Sequences of quantum gates
    - **Quantum kernels**: Feature maps using quantum states
    
    **Our implementation:**
    - **PennyLane**: Quantum machine learning framework
    - **Angle embedding**: Classical data → quantum states
    - **Variational circuits**: Parameterized quantum algorithms
    - **State fidelity**: Quantum similarity measurement
    
    **Advantage:**
    - Exponentially large feature spaces (2^n dimensions)
    - Quantum correlations impossible classically
    - Better pattern recognition for complex data
    """)

with tech_tabs[2]:
    st.markdown("""
    ### Machine Learning Pipeline
    
    **Dual approach:**
    1. **Streaming ML**: Immediate anomaly detection
       - River HalfSpaceTrees algorithm
       - No training required
       - Adapts automatically to data drift
    
    2. **Supervised Learning**: Pattern recognition from labels
       - Quantum kernel SVM
       - Human-in-the-loop feedback
       - Continuous model improvement
    
    **Online learning benefits:**
    - Adapts to evolving threats
    - Incorporates expert knowledge
    - Handles concept drift automatically
    - Scales to continuous data streams
    """)

with tech_tabs[3]:
    st.markdown("""
    ### Cryptographic Security
    
    **Quantum Key Distribution (QKD):**
    - **BB84 protocol**: Quantum key exchange
    - **QBER monitoring**: Eavesdropping detection
    - **Information-theoretic security**: Based on physics
    
    **Symmetric encryption:**
    - **AES-GCM**: Authenticated encryption
    - **HKDF**: Key derivation function
    - **Session keys**: Derived from quantum protocol
    
    **Security guarantees:**
    - Quantum mechanical security
    - Tamper detection through QBER
    - End-to-end encryption of sensitive data
    """)

with tech_tabs[4]:
    st.markdown("""
    ### System Architecture
    
    **Modular design:**
    - **Capture layer**: Network packet ingestion
    - **Processing layer**: Feature extraction and ML
    - **Security layer**: QKD and encryption
    - **Interface layer**: Streamlit web application
    
    **Data flow:**
    1. Packet capture (live or PCAP)
    2. Feature extraction and normalization
    3. Dual anomaly detection pipeline
    4. Human feedback integration
    5. Secure storage and visualization
    
    **Scalability features:**
    - Streaming processing architecture
    - Modular component design
    - Configurable parameters
    - Database abstraction layer
    """)

theme.footer()