"""
LIGO RL Visualization Dashboard

A comprehensive Streamlit dashboard for visualizing:
- Training progress (loss curves, rewards)
- Interferometer diagram (beam paths, mirror positions)
- Noise spectra (PSD plots)
- 3D OMC model
- Real-time telemetry

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pathlib import Path
import json

# Page config
st.set_page_config(
    page_title="LIGO RL Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .status-locked { color: #00ff00; }
    .status-unlocked { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔭 LIGO RL Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📈 Training", "🔬 Interferometer", "📊 Noise Spectra", "🎮 3D Model", "📡 Telemetry"]
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_mock_training_data(n_episodes=100):
    """Generate mock training data for visualization."""
    episodes = np.arange(n_episodes)
    rewards = 50 + 30 * (1 - np.exp(-episodes / 30)) + np.random.randn(n_episodes) * 5
    losses = 1.0 * np.exp(-episodes / 40) + np.random.randn(n_episodes) * 0.05
    entropy = 0.5 * np.exp(-episodes / 50) + 0.1
    return {
        'episodes': episodes,
        'rewards': rewards,
        'losses': losses,
        'entropy': entropy
    }

def generate_noise_psd(frequencies):
    """Generate mock noise PSD."""
    seismic = 1e-10 * (1 / (frequencies + 0.1)) ** 2
    thermal = 1e-12 * (1 / np.sqrt(frequencies + 1))
    quantum = 1e-14 * np.ones_like(frequencies)
    total = np.sqrt(seismic**2 + thermal**2 + quantum**2)
    return {
        'seismic': seismic,
        'thermal': thermal,
        'quantum': quantum,
        'total': total
    }

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

if page == "🏠 Overview":
    st.title("LIGO RL Control System")
    st.markdown("### Real-Time Monitoring Dashboard")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lock Status", "LOCKED", delta="Stable")
    with col2:
        st.metric("Alignment RMS", "23.4 nrad", delta="-2.1")
    with col3:
        st.metric("Episode Reward", "78.3", delta="+5.2")
    with col4:
        st.metric("Training Step", "45,230", delta="+1000/s")
    
    st.markdown("---")
    
    # Two-column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("📈 Training Progress")
        data = generate_mock_training_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['episodes'], y=data['rewards'],
            mode='lines', name='Reward',
            line=dict(color='#00ff88', width=2)
        ))
        fig.update_layout(
            template='plotly_dark',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Episode",
            yaxis_title="Reward"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.subheader("📊 Noise Spectrum")
        freqs = np.logspace(0, 2, 100)
        noise = generate_noise_psd(freqs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs, y=noise['total'],
            mode='lines', name='Total',
            line=dict(color='#ff6b6b', width=2)
        ))
        fig.update_layout(
            template='plotly_dark',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Frequency (Hz)",
            yaxis_title="ASD (1/√Hz)",
            xaxis_type="log",
            yaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interferometer mini-view
    st.subheader("🔬 Interferometer Status")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                       LASER                                 │
    │                         │                                   │
    │                         ▼                                   │
    │    ┌───────────────────────────────────────┐               │
    │    │           Input Mode Cleaner          │               │
    │    └───────────────────────────────────────┘               │
    │                         │                                   │
    │                    ┌────┴────┐                             │
    │                    │   BS    │ ← Beam Splitter             │
    │                    └────┬────┘                             │
    │              ┌──────────┴──────────┐                       │
    │              ▼                     ▼                       │
    │         ┌────────┐            ┌────────┐                   │
    │         │  ITM   │            │  ITM   │                   │
    │         │   X    │            │   Y    │                   │
    │         └────────┘            └────────┘                   │
    │              │                     │                       │
    │         4 km │                4 km │                       │
    │              ▼                     ▼                       │
    │         ┌────────┐            ┌────────┐                   │
    │         │  ETM   │            │  ETM   │                   │
    │         │   X    │            │   Y    │                   │
    │         └────────┘            └────────┘                   │
    │                                                            │
    │         ┌───────────────────────────────────┐              │
    │         │    Output Mode Cleaner (OMC)     │ ← RL Target  │
    │         └───────────────────────────────────┘              │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

# =============================================================================
# PAGE: TRAINING
# =============================================================================

elif page == "📈 Training":
    st.title("Training Progress")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_episodes = st.slider("Episodes to show", 10, 500, 100)
    with col2:
        smoothing = st.slider("Smoothing", 1, 20, 5)
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    data = generate_mock_training_data(n_episodes)
    
    # Apply smoothing
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        data['rewards'] = np.convolve(data['rewards'], kernel, mode='same')
        data['losses'] = np.convolve(data['losses'], kernel, mode='same')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Rewards", "Actor Loss", "Critic Loss", "Policy Entropy")
    )
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=data['episodes'], y=data['rewards'], 
                   mode='lines', name='Reward', line=dict(color='#00ff88')),
        row=1, col=1
    )
    
    # Actor Loss
    fig.add_trace(
        go.Scatter(x=data['episodes'], y=data['losses'], 
                   mode='lines', name='Actor Loss', line=dict(color='#ff6b6b')),
        row=1, col=2
    )
    
    # Critic Loss
    fig.add_trace(
        go.Scatter(x=data['episodes'], y=data['losses'] * 0.8 + np.random.randn(n_episodes) * 0.02,
                   mode='lines', name='Critic Loss', line=dict(color='#4ecdc4')),
        row=2, col=1
    )
    
    # Entropy
    fig.add_trace(
        go.Scatter(x=data['episodes'], y=data['entropy'],
                   mode='lines', name='Entropy', line=dict(color='#ffe66d')),
        row=2, col=2
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Training Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Reward", f"{np.max(data['rewards']):.2f}")
    with col2:
        st.metric("Mean Reward (last 10)", f"{np.mean(data['rewards'][-10:]):.2f}")
    with col3:
        st.metric("Final Loss", f"{data['losses'][-1]:.4f}")
    with col4:
        st.metric("Episodes", n_episodes)

# =============================================================================
# PAGE: INTERFEROMETER
# =============================================================================

elif page == "🔬 Interferometer":
    st.title("Interferometer Diagram")
    
    # OMC alignment controls
    st.sidebar.subheader("OMC Alignment")
    omc_pitch1 = st.sidebar.slider("OMC Mirror 1 Pitch (nrad)", -100, 100, 0)
    omc_yaw1 = st.sidebar.slider("OMC Mirror 1 Yaw (nrad)", -100, 100, 0)
    omc_pitch2 = st.sidebar.slider("OMC Mirror 2 Pitch (nrad)", -100, 100, 0)
    omc_yaw2 = st.sidebar.slider("OMC Mirror 2 Yaw (nrad)", -100, 100, 0)
    
    # Calculate beam path with misalignment
    alignment_rms = np.sqrt((omc_pitch1**2 + omc_yaw1**2 + omc_pitch2**2 + omc_yaw2**2) / 4)
    lock_status = "LOCKED" if alignment_rms < 50 else "UNLOCKED"
    status_color = "#00ff00" if lock_status == "LOCKED" else "#ff0000"
    
    # Status display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Lock Status", lock_status)
    with col2:
        st.metric("Alignment RMS", f"{alignment_rms:.1f} nrad")
    
    # Create interactive interferometer diagram
    fig = go.Figure()
    
    # OMC cavity outline
    fig.add_shape(type="rect", x0=2, y0=1, x1=8, y1=3,
                  line=dict(color="cyan", width=2), fillcolor="rgba(0,255,255,0.1)")
    
    # Mirrors
    mirror_color = status_color
    
    # OMC Mirror 1 (input)
    m1_x, m1_y = 2.5, 2
    fig.add_trace(go.Scatter(
        x=[m1_x], y=[m1_y], mode='markers+text',
        marker=dict(size=30, color=mirror_color, symbol='square'),
        text=['M1'], textposition='bottom center', name='Mirror 1'
    ))
    
    # OMC Mirror 2 (output)
    m2_x, m2_y = 7.5, 2
    fig.add_trace(go.Scatter(
        x=[m2_x], y=[m2_y], mode='markers+text',
        marker=dict(size=30, color=mirror_color, symbol='square'),
        text=['M2'], textposition='bottom center', name='Mirror 2'
    ))
    
    # Beam path (affected by alignment)
    beam_offset = (omc_pitch1 + omc_pitch2) / 500  # Scale for visualization
    fig.add_trace(go.Scatter(
        x=[0, m1_x, m2_x, 10],
        y=[2, 2 + beam_offset * 0.5, 2 + beam_offset, 2 + beam_offset * 1.2],
        mode='lines',
        line=dict(color='red', width=3),
        name='Laser Beam'
    ))
    
    # Photodetector
    fig.add_trace(go.Scatter(
        x=[9], y=[2 + beam_offset * 1.2], mode='markers+text',
        marker=dict(size=40, color='yellow', symbol='triangle-right'),
        text=['PD'], textposition='bottom center', name='Photodetector'
    ))
    
    # Labels
    fig.add_annotation(x=5, y=3.5, text="Output Mode Cleaner (OMC)",
                       showarrow=False, font=dict(size=16, color='white'))
    fig.add_annotation(x=0, y=2, text="From IFO →", showarrow=False,
                       font=dict(size=12, color='gray'))
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis=dict(visible=False, range=[-1, 11]),
        yaxis=dict(visible=False, range=[0, 4], scaleanchor="x"),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time alignment visualization
    st.subheader("Alignment State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pitch-Yaw plot for Mirror 1
        fig_m1 = go.Figure()
        fig_m1.add_trace(go.Scatter(
            x=[omc_yaw1], y=[omc_pitch1], mode='markers',
            marker=dict(size=20, color='cyan'),
            name='Current'
        ))
        fig_m1.add_shape(type="circle", x0=-50, y0=-50, x1=50, y1=50,
                         line=dict(color="green", dash="dash"))
        fig_m1.add_shape(type="circle", x0=-100, y0=-100, x1=100, y1=100,
                         line=dict(color="red", dash="dash"))
        fig_m1.update_layout(
            template='plotly_dark',
            height=300,
            title="Mirror 1 (Pitch vs Yaw)",
            xaxis_title="Yaw (nrad)",
            yaxis_title="Pitch (nrad)",
            xaxis=dict(range=[-120, 120]),
            yaxis=dict(range=[-120, 120], scaleanchor="x")
        )
        st.plotly_chart(fig_m1, use_container_width=True)
    
    with col2:
        # Pitch-Yaw plot for Mirror 2
        fig_m2 = go.Figure()
        fig_m2.add_trace(go.Scatter(
            x=[omc_yaw2], y=[omc_pitch2], mode='markers',
            marker=dict(size=20, color='magenta'),
            name='Current'
        ))
        fig_m2.add_shape(type="circle", x0=-50, y0=-50, x1=50, y1=50,
                         line=dict(color="green", dash="dash"))
        fig_m2.add_shape(type="circle", x0=-100, y0=-100, x1=100, y1=100,
                         line=dict(color="red", dash="dash"))
        fig_m2.update_layout(
            template='plotly_dark',
            height=300,
            title="Mirror 2 (Pitch vs Yaw)",
            xaxis_title="Yaw (nrad)",
            yaxis_title="Pitch (nrad)",
            xaxis=dict(range=[-120, 120]),
            yaxis=dict(range=[-120, 120], scaleanchor="x")
        )
        st.plotly_chart(fig_m2, use_container_width=True)

# =============================================================================
# PAGE: NOISE SPECTRA
# =============================================================================

elif page == "📊 Noise Spectra":
    st.title("Noise Power Spectral Density")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        freq_min = st.slider("Min Frequency (Hz)", 1, 10, 1)
        freq_max = st.slider("Max Frequency (Hz)", 50, 500, 100)
    with col2:
        show_seismic = st.checkbox("Seismic", value=True)
        show_thermal = st.checkbox("Thermal", value=True)
        show_quantum = st.checkbox("Quantum", value=True)
        show_control = st.checkbox("Control Noise", value=True)
    
    # Generate data
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), 200)
    noise = generate_noise_psd(freqs)
    
    # Control noise (from RL policy)
    control_noise = 1e-13 * np.exp(-freqs / 30)
    
    # Create figure
    fig = go.Figure()
    
    if show_seismic:
        fig.add_trace(go.Scatter(
            x=freqs, y=noise['seismic'],
            mode='lines', name='Seismic',
            line=dict(color='#ff6b6b', width=2)
        ))
    
    if show_thermal:
        fig.add_trace(go.Scatter(
            x=freqs, y=noise['thermal'],
            mode='lines', name='Thermal',
            line=dict(color='#4ecdc4', width=2)
        ))
    
    if show_quantum:
        fig.add_trace(go.Scatter(
            x=freqs, y=noise['quantum'],
            mode='lines', name='Quantum',
            line=dict(color='#ffe66d', width=2)
        ))
    
    if show_control:
        fig.add_trace(go.Scatter(
            x=freqs, y=control_noise,
            mode='lines', name='Control (RL Policy)',
            line=dict(color='#00ff88', width=2, dash='dash')
        ))
    
    # Total
    fig.add_trace(go.Scatter(
        x=freqs, y=noise['total'],
        mode='lines', name='Total',
        line=dict(color='white', width=3)
    ))
    
    # Observation band highlight
    fig.add_vrect(x0=10, x1=30, fillcolor="rgba(255,255,0,0.1)",
                  layer="below", line_width=0,
                  annotation_text="Observation Band", annotation_position="top")
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude Spectral Density (1/√Hz)",
        xaxis_type="log",
        yaxis_type="log",
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time domain
    st.subheader("Time Domain Signal")
    
    t = np.linspace(0, 1, 1000)
    signal = (
        0.1 * np.sin(2 * np.pi * 0.5 * t) +  # Seismic
        0.05 * np.sin(2 * np.pi * 15 * t) +   # In observation band
        0.01 * np.random.randn(len(t))        # White noise
    )
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=t * 1000, y=signal,
        mode='lines', name='Signal',
        line=dict(color='#00ff88', width=1)
    ))
    fig_time.update_layout(
        template='plotly_dark',
        height=250,
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude"
    )
    st.plotly_chart(fig_time, use_container_width=True)

# =============================================================================
# PAGE: 3D MODEL
# =============================================================================

elif page == "🎮 3D Model":
    st.title("3D Output Mode Cleaner Model")
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    with col1:
        camera_x = st.slider("Camera X", -5, 5, 2)
    with col2:
        camera_y = st.slider("Camera Y", -5, 5, 2)
    with col3:
        camera_z = st.slider("Camera Z", 1, 10, 3)
    
    # Create 3D figure
    fig = go.Figure()
    
    # Optical table (base)
    table_x = np.array([0, 10, 10, 0, 0])
    table_y = np.array([0, 0, 6, 6, 0])
    table_z = np.array([0, 0, 0, 0, 0])
    
    fig.add_trace(go.Mesh3d(
        x=[0, 10, 10, 0, 0, 10, 10, 0],
        y=[0, 0, 6, 6, 0, 0, 6, 6],
        z=[0, 0, 0, 0, -0.2, -0.2, -0.2, -0.2],
        color='gray',
        opacity=0.5,
        name='Optical Table'
    ))
    
    # OMC Mirror 1 (cylinder approximation)
    theta = np.linspace(0, 2*np.pi, 20)
    m1_x = 2 + 0.3 * np.cos(theta)
    m1_y = 3 + 0.3 * np.sin(theta)
    m1_z = np.ones_like(theta) * 0.5
    
    fig.add_trace(go.Scatter3d(
        x=m1_x, y=m1_y, z=m1_z,
        mode='lines',
        line=dict(color='cyan', width=10),
        name='OMC Mirror 1'
    ))
    
    # OMC Mirror 2
    m2_x = 8 + 0.3 * np.cos(theta)
    m2_y = 3 + 0.3 * np.sin(theta)
    m2_z = np.ones_like(theta) * 0.5
    
    fig.add_trace(go.Scatter3d(
        x=m2_x, y=m2_y, z=m2_z,
        mode='lines',
        line=dict(color='magenta', width=10),
        name='OMC Mirror 2'
    ))
    
    # Laser beam
    fig.add_trace(go.Scatter3d(
        x=[0, 2, 8, 10],
        y=[3, 3, 3, 3],
        z=[0.5, 0.5, 0.5, 0.5],
        mode='lines',
        line=dict(color='red', width=5),
        name='Laser Beam'
    ))
    
    # Photodetector
    fig.add_trace(go.Scatter3d(
        x=[10], y=[3], z=[0.5],
        mode='markers',
        marker=dict(size=15, color='yellow', symbol='diamond'),
        name='Photodetector'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            camera=dict(
                eye=dict(x=camera_x, y=camera_y, z=camera_z)
            ),
            aspectmode='data'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Use the sliders above to rotate the camera, or drag the 3D model directly!")

# =============================================================================
# PAGE: TELEMETRY
# =============================================================================

elif page == "📡 Telemetry":
    st.title("Real-Time Telemetry")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (every 2s)", value=False)
    
    if auto_refresh:
        placeholder = st.empty()
        
        # Simulate real-time updates
        for i in range(100):
            with placeholder.container():
                # Generate random telemetry
                alignment_rms = 20 + np.random.randn() * 5
                power = 0.95 + np.random.randn() * 0.02
                step = 1000 + i * 10
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Alignment RMS", f"{alignment_rms:.1f} nrad")
                with col2:
                    st.metric("Power", f"{power:.3f}")
                with col3:
                    st.metric("Step", step)
                with col4:
                    st.metric("Lock", "LOCKED" if alignment_rms < 50 else "UNLOCKED")
                
                # Time series (last 50 points)
                t = np.arange(50)
                alignment_history = 20 + np.cumsum(np.random.randn(50)) * 0.5
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t, y=alignment_history,
                    mode='lines', name='Alignment RMS',
                    line=dict(color='#00ff88')
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    xaxis_title="Time (steps)",
                    yaxis_title="Alignment RMS (nrad)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(2)
    else:
        st.info("Enable auto-refresh to see real-time telemetry updates.")
        
        # Static telemetry display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Alignment RMS", "23.4 nrad")
        with col2:
            st.metric("Power", "0.953")
        with col3:
            st.metric("Training Step", "45,230")
        with col4:
            st.metric("Lock Status", "LOCKED")
        
        # Historical data
        st.subheader("Episode History")
        
        episodes = np.arange(100)
        rewards = 50 + 30 * (1 - np.exp(-episodes / 30)) + np.random.randn(100) * 5
        alignment = 100 * np.exp(-episodes / 30) + 20 + np.random.randn(100) * 3
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Episode Reward", "Alignment RMS"))
        
        fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='lines',
                                  line=dict(color='#00ff88')), row=1, col=1)
        fig.add_trace(go.Scatter(x=episodes, y=alignment, mode='lines',
                                  line=dict(color='#ff6b6b')), row=2, col=1)
        
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**LIGO RL** | Physics-Based Reinforcement Learning for Gravitational Wave Detectors")
