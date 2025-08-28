import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from pennylane import numpy as pnp

# Set style for better visualizations
plt.style.use('dark_background')
sns.set_palette("husl")

class ProjectVisualizationDemo:
    """
    Complete visualization suite to demonstrate what's happening in our project
    """
    
    def __init__(self):
        self.colors = {
            'quantum': '#ff0080',
            'classical': '#00e0ff', 
            'normal': '#00ff88',
            'anomaly': '#ff4444',
            'secure': '#ffd700'
        }
    
    def demo_1_network_packet_analysis(self):
        """
        Show how we extract features from network packets
        """
        # Simulate packet data
        np.random.seed(42)
        time_stamps = np.cumsum(np.random.exponential(0.1, 1000))
        packet_sizes = np.random.lognormal(6, 1.5, 1000).astype(int)
        tcp_flags = np.random.choice(['SYN', 'ACK', 'FIN', 'RST', 'PSH'], 1000)
        src_ports = np.random.choice(range(1024, 65535), 1000)
        dst_ports = np.random.choice([80, 443, 22, 53, 25, 993], 1000, p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Packet Timing Analysis', 'Packet Size Distribution', 
                          'TCP Flags Distribution', 'Port Entropy Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Timing analysis
        inter_arrival = np.diff(time_stamps)
        fig.add_trace(go.Scatter(x=time_stamps[1:], y=inter_arrival, 
                                mode='lines', name='Inter-arrival Time',
                                line=dict(color=self.colors['normal'])), row=1, col=1)
        
        # 2. Packet size distribution
        fig.add_trace(go.Histogram(x=packet_sizes, nbinsx=50, name='Packet Sizes',
                                  marker_color=self.colors['classical']), row=1, col=2)
        
        # 3. TCP flags
        flag_counts = {flag: list(tcp_flags).count(flag) for flag in ['SYN', 'ACK', 'FIN', 'RST', 'PSH']}
        fig.add_trace(go.Bar(x=list(flag_counts.keys()), y=list(flag_counts.values()),
                            name='TCP Flags', marker_color=self.colors['quantum']), row=2, col=1)
        
        # 4. Port entropy over time (sliding window)
        window_size = 100
        entropies = []
        for i in range(window_size, len(dst_ports)):
            window_ports = dst_ports[i-window_size:i]
            unique, counts = np.unique(window_ports, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
        
        fig.add_trace(go.Scatter(x=time_stamps[window_size:], y=entropies,
                                mode='lines', name='Port Entropy',
                                line=dict(color=self.colors['anomaly'])), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Network Packet Feature Extraction Demo",
                         showlegend=True, template="plotly_dark")
        return fig
    
    def demo_2_quantum_vs_classical_kernels(self):
        """
        Visualize the difference between quantum and classical kernel transformations
        """
        # Generate sample data
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        
        # Classical RBF kernel transformation (approximate visualization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Simulate quantum kernel transformation
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev)
        def quantum_embedding(x):
            qml.AngleEmbedding(x, wires=[0, 1])
            qml.BasicEntanglerLayers([[1.0, 1.0]], wires=[0, 1])
            return qml.state()
        
        # Create visualization
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Original Data', 'Classical RBF Kernel Space',
                                         'Quantum Circuit', 'Quantum Kernel Similarity'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}]])
        
        # 1. Original data
        colors = ['red' if label == 0 else 'blue' for label in y]
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                marker=dict(color=colors), name='Original Data'), row=1, col=1)
        
        # 2. Classical kernel space (PCA approximation of RBF)
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(kernel='rbf', gamma=1.0, n_components=2)
        X_kpca = kpca.fit_transform(X_scaled)
        fig.add_trace(go.Scatter(x=X_kpca[:, 0], y=X_kpca[:, 1], mode='markers',
                                marker=dict(color=colors), name='RBF Kernel'), row=1, col=2)
        
        # 3. Quantum circuit diagram (text representation)
        circuit_text = """
        |0⟩ ──RY(x₁)──●──────── |ψ⟩
                     │
        |0⟩ ──RY(x₂)──X──RY(θ)── |ψ⟩
        
        Quantum Feature Map:
        • AngleEmbedding: x → RY(x)
        • Entangling Layer: CNOT + RY
        • Output: |ψ(x)⟩ ∈ ℂ⁴
        """
        fig.add_annotation(text=circuit_text, x=0.5, y=0.5, xref="x3", yref="y3",
                          showarrow=False, font=dict(family="monospace", size=12))
        
        # 4. Quantum kernel similarity matrix
        sample_indices = np.random.choice(len(X), 20, replace=False)
        X_sample = X_scaled[sample_indices]
        
        # Compute quantum kernel matrix (simplified)
        kernel_matrix = np.zeros((len(X_sample), len(X_sample)))
        for i in range(len(X_sample)):
            for j in range(len(X_sample)):
                # Simulate quantum kernel computation
                state_i = quantum_embedding(X_sample[i])
                state_j = quantum_embedding(X_sample[j])
                # Fidelity between quantum states
                kernel_matrix[i, j] = np.abs(np.vdot(state_i, state_j))**2
        
        fig.add_trace(go.Heatmap(z=kernel_matrix, colorscale='Viridis',
                                name='Quantum Kernel Matrix'), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Quantum vs Classical Kernel Comparison",
                         template="plotly_dark")
        return fig
    
    def demo_3_online_learning_process(self):
        """
        Show how the system learns and adapts over time
        """
        # Simulate streaming data with concept drift
        np.random.seed(42)
        n_samples = 1000
        
        # Generate data with changing patterns
        time_points = np.arange(n_samples)
        
        # Normal behavior (first half)
        normal_scores_1 = np.random.normal(0.2, 0.1, n_samples//2)
        # Concept drift (second half) - new normal behavior
        normal_scores_2 = np.random.normal(0.4, 0.15, n_samples//2)
        
        # Inject anomalies
        anomaly_indices = np.random.choice(n_samples, 50, replace=False)
        anomaly_scores = np.random.uniform(0.8, 1.0, 50)
        
        # Combine scores
        all_scores = np.concatenate([normal_scores_1, normal_scores_2])
        all_scores[anomaly_indices] = anomaly_scores
        
        # Simulate human labeling (sparse)
        labeled_indices = np.random.choice(n_samples, 100, replace=False)
        labels = np.where(np.isin(labeled_indices, anomaly_indices), 1, 0)
        
        # Create visualization
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Streaming Anomaly Scores', 
                                         'Human Labeling Process',
                                         'Model Performance Over Time'),
                           vertical_spacing=0.08)
        
        # 1. Streaming scores
        colors = ['red' if i in anomaly_indices else 'lightblue' for i in range(n_samples)]
        fig.add_trace(go.Scatter(x=time_points, y=all_scores, mode='markers',
                                marker=dict(color=colors, size=4),
                                name='Anomaly Scores'), row=1, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="yellow", 
                     annotation_text="Threshold", row=1, col=1)
        
        # 2. Human labeling
        fig.add_trace(go.Scatter(x=labeled_indices, y=all_scores[labeled_indices],
                                mode='markers', marker=dict(color=labels, colorscale='RdYlBu',
                                size=8, symbol='diamond'), name='Human Labels'), row=2, col=1)
        
        # 3. Model performance simulation
        # Simulate improving accuracy as more labels are provided
        cumulative_labels = np.cumsum(np.ones(len(labeled_indices)))
        accuracy_improvement = 0.5 + 0.4 * (1 - np.exp(-cumulative_labels / 20))
        
        fig.add_trace(go.Scatter(x=labeled_indices, y=accuracy_improvement,
                                mode='lines+markers', name='Model Accuracy',
                                line=dict(color=self.colors['quantum'])), row=3, col=1)
        
        fig.update_layout(height=900, title_text="Online Learning and Human-in-the-Loop Process",
                         template="plotly_dark")
        return fig
    
    def demo_4_qkd_bb84_protocol(self):
        """
        Visualize the BB84 Quantum Key Distribution protocol
        """
        np.random.seed(42)
        n_bits = 100
        
        # Alice's preparation
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)  # 0: +, 1: x
        
        # Bob's measurement
        bob_bases = np.random.randint(0, 2, n_bits)
        
        # Simulate measurement results
        bob_bits = []
        for i in range(n_bits):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - Bob gets Alice's bit
                bob_bits.append(alice_bits[i])
            else:
                # Different basis - random result
                bob_bits.append(np.random.randint(0, 2))
        bob_bits = np.array(bob_bits)
        
        # Basis matching
        matching_bases = alice_bases == bob_bases
        sifted_key = alice_bits[matching_bases]
        
        # Error estimation (simulate eavesdropping)
        eavesdropping_rate = 0.05
        errors = np.random.random(len(sifted_key)) < eavesdropping_rate
        qber = np.mean(errors)
        
        # Create visualization
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Bit and Basis Preparation', 'Basis Matching Process',
                                         'Key Sifting Results', 'QBER Analysis'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}]])
        
        # 1. Alice's preparation
        x_pos = np.arange(min(50, n_bits))
        fig.add_trace(go.Scatter(x=x_pos, y=alice_bits[:50], mode='markers+lines',
                                marker=dict(color=alice_bases[:50], colorscale='RdBu',
                                          size=8), name="Alice's Bits"), row=1, col=1)
        
        # 2. Basis matching
        match_colors = ['green' if match else 'red' for match in matching_bases[:50]]
        fig.add_trace(go.Scatter(x=x_pos, y=matching_bases[:50].astype(int),
                                mode='markers', marker=dict(color=match_colors, size=10),
                                name='Basis Match'), row=1, col=2)
        
        # 3. Sifted key
        sifted_x = np.arange(len(sifted_key[:30]))
        fig.add_trace(go.Scatter(x=sifted_x, y=sifted_key[:30], mode='markers+lines',
                                marker=dict(color=self.colors['secure'], size=8),
                                name='Sifted Key'), row=2, col=1)
        
        # 4. QBER analysis
        qber_history = np.random.uniform(0.01, 0.15, 20)
        qber_history[-1] = qber
        fig.add_trace(go.Scatter(x=np.arange(20), y=qber_history, mode='lines+markers',
                                name='QBER', line=dict(color=self.colors['anomaly'])), row=2, col=2)
        fig.add_hline(y=0.11, line_dash="dash", line_color="yellow",
                     annotation_text="Security Threshold", row=2, col=2)
        
        fig.update_layout(height=800, title_text="BB84 Quantum Key Distribution Protocol",
                         template="plotly_dark")
        return fig
    
    def demo_5_system_performance_metrics(self):
        """
        Show comprehensive performance comparison between quantum and classical approaches
        """
        # Simulate performance data
        np.random.seed(42)
        
        # Metrics over time (as more training data is added)
        training_sizes = np.array([20, 50, 100, 200, 500, 1000])
        
        # Quantum SVM performance (starts lower, improves more)
        quantum_accuracy = 0.6 + 0.35 * (1 - np.exp(-training_sizes / 200)) + np.random.normal(0, 0.02, len(training_sizes))
        quantum_f1 = 0.55 + 0.4 * (1 - np.exp(-training_sizes / 250)) + np.random.normal(0, 0.02, len(training_sizes))
        quantum_auc = 0.65 + 0.3 * (1 - np.exp(-training_sizes / 180)) + np.random.normal(0, 0.02, len(training_sizes))
        
        # Classical SVM performance (starts higher, plateaus earlier)
        classical_accuracy = 0.75 + 0.2 * (1 - np.exp(-training_sizes / 100)) + np.random.normal(0, 0.02, len(training_sizes))
        classical_f1 = 0.7 + 0.25 * (1 - np.exp(-training_sizes / 120)) + np.random.normal(0, 0.02, len(training_sizes))
        classical_auc = 0.72 + 0.23 * (1 - np.exp(-training_sizes / 110)) + np.random.normal(0, 0.02, len(training_sizes))
        
        # Create comprehensive performance dashboard
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Accuracy Comparison', 'F1-Score Comparison',
                                         'ROC-AUC Comparison', 'Training Time Analysis'))
        
        # Accuracy
        fig.add_trace(go.Scatter(x=training_sizes, y=quantum_accuracy, mode='lines+markers',
                                name='Quantum SVM', line=dict(color=self.colors['quantum'])), row=1, col=1)
        fig.add_trace(go.Scatter(x=training_sizes, y=classical_accuracy, mode='lines+markers',
                                name='Classical SVM', line=dict(color=self.colors['classical'])), row=1, col=1)
        
        # F1-Score
        fig.add_trace(go.Scatter(x=training_sizes, y=quantum_f1, mode='lines+markers',
                                name='Quantum F1', line=dict(color=self.colors['quantum']),
                                showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=training_sizes, y=classical_f1, mode='lines+markers',
                                name='Classical F1', line=dict(color=self.colors['classical']),
                                showlegend=False), row=1, col=2)
        
        # AUC
        fig.add_trace(go.Scatter(x=training_sizes, y=quantum_auc, mode='lines+markers',
                                name='Quantum AUC', line=dict(color=self.colors['quantum']),
                                showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=training_sizes, y=classical_auc, mode='lines+markers',
                                name='Classical AUC', line=dict(color=self.colors['classical']),
                                showlegend=False), row=2, col=1)
        
        # Training time (quantum is slower but more capable)
        quantum_time = training_sizes * 0.1 + np.random.normal(0, 0.5, len(training_sizes))
        classical_time = training_sizes * 0.02 + np.random.normal(0, 0.1, len(training_sizes))
        
        fig.add_trace(go.Scatter(x=training_sizes, y=quantum_time, mode='lines+markers',
                                name='Quantum Time', line=dict(color=self.colors['quantum']),
                                showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=training_sizes, y=classical_time, mode='lines+markers',
                                name='Classical Time', line=dict(color=self.colors['classical']),
                                showlegend=False), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Quantum vs Classical Performance Analysis",
                         template="plotly_dark")
        return fig
    
    def demo_6_security_dashboard(self):
        """
        Create a comprehensive security monitoring dashboard
        """
        # Simulate network security data
        np.random.seed(42)
        
        # Time series data
        hours = np.arange(24)
        normal_traffic = 1000 + 500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 50, 24)
        anomaly_traffic = np.random.poisson(10, 24)
        
        # Top threats
        threat_types = ['Port Scan', 'DDoS', 'Malware C&C', 'Data Exfiltration', 'Brute Force']
        threat_counts = [45, 23, 12, 8, 15]
        
        # Geographic data (simulated)
        countries = ['USA', 'China', 'Russia', 'Germany', 'Brazil', 'India']
        attack_counts = [120, 89, 67, 34, 28, 19]
        
        # Create security dashboard
        fig = make_subplots(rows=2, cols=3,
                           subplot_titles=('24h Traffic Analysis', 'Threat Type Distribution',
                                         'Geographic Attack Sources', 'Anomaly Score Timeline',
                                         'QKD Security Status', 'System Health Metrics'),
                           specs=[[{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]])
        
        # 1. Traffic analysis
        fig.add_trace(go.Scatter(x=hours, y=normal_traffic, mode='lines+markers',
                                name='Normal Traffic', line=dict(color=self.colors['normal'])), row=1, col=1)
        fig.add_trace(go.Bar(x=hours, y=anomaly_traffic, name='Anomalies',
                            marker_color=self.colors['anomaly'], opacity=0.7), row=1, col=1, secondary_y=True)
        
        # 2. Threat types
        fig.add_trace(go.Pie(labels=threat_types, values=threat_counts, name="Threats",
                            marker_colors=[self.colors['anomaly']] * len(threat_types)), row=1, col=2)
        
        # 3. Geographic sources
        fig.add_trace(go.Bar(x=countries, y=attack_counts, name='Attack Sources',
                            marker_color=self.colors['classical']), row=1, col=3)
        
        # 4. Anomaly timeline
        timeline = np.arange(100)
        anomaly_scores = 0.2 + 0.1 * np.sin(timeline / 10) + np.random.exponential(0.1, 100)
        anomaly_scores[np.random.choice(100, 10)] += np.random.uniform(0.5, 0.8, 10)
        
        fig.add_trace(go.Scatter(x=timeline, y=anomaly_scores, mode='lines',
                                name='Anomaly Scores', line=dict(color=self.colors['quantum'])), row=2, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="yellow", row=2, col=1)
        
        # 5. QKD status
        qkd_metrics = ['Key Exchange Rate', 'QBER', 'Channel Integrity', 'Encryption Status']
        qkd_values = [98.5, 2.1, 99.8, 100.0]
        qkd_colors = ['green' if v > 95 else 'yellow' if v > 80 else 'red' for v in qkd_values]
        
        fig.add_trace(go.Bar(x=qkd_metrics, y=qkd_values, name='QKD Status',
                            marker_color=qkd_colors), row=2, col=2)
        
        # 6. System health
        health_metrics = ['CPU Usage', 'Memory Usage', 'Network I/O', 'Detection Rate']
        health_values = [45, 67, 23, 94]
        
        fig.add_trace(go.Bar(x=health_metrics, y=health_values, name='System Health',
                            marker_color=self.colors['secure']), row=2, col=3)
        
        fig.update_layout(height=900, title_text="Comprehensive Security Operations Dashboard",
                         template="plotly_dark")
        return fig

def generate_all_demos():
    """
    Generate all demonstration visualizations
    """
    demo = ProjectVisualizationDemo()
    
    demos = {
        "Network Packet Analysis": demo.demo_1_network_packet_analysis(),
        "Quantum vs Classical Kernels": demo.demo_2_quantum_vs_classical_kernels(),
        "Online Learning Process": demo.demo_3_online_learning_process(),
        "QKD BB84 Protocol": demo.demo_4_qkd_bb84_protocol(),
        "Performance Metrics": demo.demo_5_system_performance_metrics(),
        "Security Dashboard": demo.demo_6_security_dashboard()
    }
    
    return demos

if __name__ == "__main__":
    demos = generate_all_demos()
    for name, fig in demos.items():
        fig.show()
        print(f"Generated: {name}")
