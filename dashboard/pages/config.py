"""
Configuration Page - Modular System Settings

Allows users to:
- Select RL algorithm (SAC, PPO, DreamerV3, etc.)
- Tune hyperparameters
- Choose control target (OMC, IMC, θ_CHP, etc.)
- Configure environment settings
"""

import streamlit as st
import yaml
from pathlib import Path


def render_config_page():
    """Render the configuration page."""
    
    st.title("⚙️ System Configuration")
    st.markdown("Configure the RL training system - all parameters are modular and adjustable.")
    
    # Tabs for different config sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Algorithm", "🎯 Control Target", "🌍 Environment", "📊 Training"
    ])
    
    # ==========================================================================
    # TAB 1: ALGORITHM SELECTION
    # ==========================================================================
    with tab1:
        st.header("Reinforcement Learning Algorithm")
        
        algorithm = st.selectbox(
            "Select Algorithm",
            ["SAC (Recommended)", "PPO", "TD3", "DreamerV3 (Model-Based)", "A2C"],
            help="Choose the RL algorithm for training"
        )
        
        st.markdown("---")
        
        if "SAC" in algorithm:
            st.subheader("SAC Hyperparameters")
            st.markdown("*Soft Actor-Critic - Good for continuous control*")
            
            col1, col2 = st.columns(2)
            with col1:
                sac_lr = st.number_input("Learning Rate", value=3e-4, format="%.1e",
                                         help="Learning rate for actor and critic")
                sac_gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99,
                                      help="How much to value future rewards")
                sac_tau = st.slider("Soft Update (τ)", 0.001, 0.1, 0.005,
                                    help="Target network update rate")
            with col2:
                sac_buffer = st.number_input("Replay Buffer Size", value=1000000,
                                             help="Number of transitions to store")
                sac_batch = st.slider("Batch Size", 32, 512, 256,
                                      help="Training batch size")
                sac_entropy = st.slider("Entropy Coefficient", 0.0, 1.0, 0.2,
                                        help="Encourages exploration")
        
        elif "PPO" in algorithm:
            st.subheader("PPO Hyperparameters")
            st.markdown("*Proximal Policy Optimization - Stable and reliable*")
            
            col1, col2 = st.columns(2)
            with col1:
                ppo_lr = st.number_input("Learning Rate", value=3e-4, format="%.1e")
                ppo_gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99)
                ppo_clip = st.slider("Clip Range", 0.1, 0.4, 0.2,
                                     help="PPO clipping parameter")
            with col2:
                ppo_epochs = st.slider("Epochs per Update", 1, 20, 10)
                ppo_batch = st.slider("Batch Size", 32, 256, 64)
                ppo_gae = st.slider("GAE Lambda", 0.9, 1.0, 0.95)
        
        elif "DreamerV3" in algorithm:
            st.subheader("DreamerV3 Hyperparameters")
            st.markdown("*Model-Based RL - Learns a world model for imagination*")
            
            col1, col2 = st.columns(2)
            with col1:
                dreamer_lr = st.number_input("Model Learning Rate", value=1e-4, format="%.1e")
                dreamer_horizon = st.slider("Imagination Horizon", 5, 50, 15,
                                            help="Steps to imagine into future")
                dreamer_deter = st.slider("Deterministic State Size", 128, 1024, 512)
            with col2:
                dreamer_stoch = st.slider("Stochastic State Size", 16, 64, 32)
                dreamer_kl = st.slider("KL Scale", 0.01, 1.0, 0.1)
                dreamer_discount = st.slider("Discount", 0.99, 0.999, 0.997)
            
            st.info("⚠️ DreamerV3 requires more VRAM (~6GB+). Your RTX 3050 (4GB) may need reduced batch size.")
        
        elif "TD3" in algorithm:
            st.subheader("TD3 Hyperparameters")
            st.markdown("*Twin Delayed DDPG - Robust continuous control*")
            
            col1, col2 = st.columns(2)
            with col1:
                td3_lr = st.number_input("Learning Rate", value=3e-4, format="%.1e")
                td3_gamma = st.slider("Discount Factor", 0.9, 0.999, 0.99)
                td3_tau = st.slider("Soft Update (τ)", 0.001, 0.01, 0.005)
            with col2:
                td3_policy_delay = st.slider("Policy Delay", 1, 5, 2)
                td3_noise = st.slider("Target Noise", 0.1, 0.5, 0.2)
                td3_noise_clip = st.slider("Noise Clip", 0.1, 1.0, 0.5)
    
    # ==========================================================================
    # TAB 2: CONTROL TARGET
    # ==========================================================================
    with tab2:
        st.header("Control Target Selection")
        st.markdown("Choose which LIGO subsystem to control")
        
        target = st.selectbox(
            "Control Target",
            [
                "Output Mode Cleaner (OMC) - Alignment",
                "Output Mode Cleaner (OMC) - Squeezing",
                "Input Mode Cleaner (IMC)",
                "Common Hard Pitch (θ_CHP)",
                "Common Soft Pitch (θ_CSP)",
                "Differential Hard Pitch (θ_DHP)",
                "Custom (Advanced)"
            ],
            help="Select which control loop to optimize"
        )
        
        st.markdown("---")
        
        if "OMC" in target and "Alignment" in target:
            st.subheader("OMC Alignment Control")
            st.markdown("""
            **Target**: Optimize Output Mode Cleaner mirror alignment
            
            **Degrees of Freedom**: 4 (pitch/yaw for 2 mirrors)
            
            **Objective**: Maximize transmitted power, minimize mode mismatch
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                omc_dof = st.multiselect(
                    "Active DOF",
                    ["M1 Pitch", "M1 Yaw", "M2 Pitch", "M2 Yaw"],
                    default=["M1 Pitch", "M1 Yaw", "M2 Pitch", "M2 Yaw"]
                )
            with col2:
                omc_max_action = st.slider("Max Action (nrad/step)", 10, 200, 100)
        
        elif "Squeezing" in target:
            st.subheader("Squeezing Optimization")
            st.markdown("""
            **Target**: Optimize squeezed light injection angle
            
            **Objective**: Minimize quantum noise at target frequency
            
            ⚠️ *Experimental - Not in original Deep Loop Shaping paper*
            """)
            
            squeezing_freq = st.slider("Target Frequency (Hz)", 10, 100, 50)
            squeezing_db = st.slider("Squeezing Level (dB)", 0, 15, 10)
        
        elif "θ_CHP" in target:
            st.subheader("Common Hard Pitch (θ_CHP)")
            st.markdown("""
            **Target**: Same as Deep Loop Shaping paper
            
            **Challenge**: Unstable opto-mechanical mode at ~2.3 Hz
            
            **Objective**: Reduce control noise in 10-30 Hz band
            """)
            
            st.warning("This is the same target as the DeepMind paper. Consider OMC for novel research.")
    
    # ==========================================================================
    # TAB 3: ENVIRONMENT
    # ==========================================================================
    with tab3:
        st.header("Environment Configuration")
        
        st.subheader("Simulation Settings")
        col1, col2 = st.columns(2)
        with col1:
            sim_mode = st.selectbox(
                "Simulation Mode",
                ["Mock (Fast, No Finesse)", "Finesse Linear", "Finesse Full (Slow)"],
                help="Trade-off between speed and physics accuracy"
            )
            control_rate = st.selectbox(
                "Control Rate",
                [256, 512, 1024, 2048],
                help="Policy execution rate in Hz"
            )
        with col2:
            episode_length = st.slider("Episode Length (steps)", 256, 4096, 1024)
            max_misalignment = st.slider("Max Misalignment (nrad)", 100, 1000, 500,
                                         help="Episode terminates if exceeded")
        
        st.markdown("---")
        
        st.subheader("Domain Randomization")
        st.markdown("*Randomize parameters for robust policy learning*")
        
        dr_enabled = st.checkbox("Enable Domain Randomization", value=True)
        
        if dr_enabled:
            col1, col2 = st.columns(2)
            with col1:
                dr_seismic = st.slider("Seismic Noise Range", 0.5, 3.0, (0.5, 3.0))
                dr_pole = st.slider("Pole Frequency Range", 0.8, 1.2, (0.8, 1.2))
            with col2:
                dr_offset = st.slider("Initial Offset Range (nrad)", 0, 100, 50)
                dr_delay = st.slider("Control Delay (samples)", 0, 4, (0, 3))
        
        st.markdown("---")
        
        st.subheader("Noise Sources")
        col1, col2, col3 = st.columns(3)
        with col1:
            noise_seismic = st.checkbox("Seismic Noise", value=True)
            seismic_amp = st.slider("Amplitude", 0.1, 5.0, 1.0) if noise_seismic else None
        with col2:
            noise_thermal = st.checkbox("Thermal Noise", value=True)
            thermal_amp = st.slider("Amplitude ", 0.1, 2.0, 0.5) if noise_thermal else None
        with col3:
            noise_quantum = st.checkbox("Quantum Noise", value=True)
            quantum_amp = st.slider("Amplitude  ", 0.01, 0.5, 0.1) if noise_quantum else None
    
    # ==========================================================================
    # TAB 4: TRAINING
    # ==========================================================================
    with tab4:
        st.header("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            total_steps = st.number_input("Total Training Steps", value=100000, step=10000)
            eval_freq = st.number_input("Evaluation Frequency", value=1000, step=100)
            save_freq = st.number_input("Checkpoint Frequency", value=10000, step=1000)
        with col2:
            seed = st.number_input("Random Seed", value=42)
            n_envs = st.slider("Parallel Environments", 1, 8, 1,
                               help="More = faster but uses more RAM")
            device = st.selectbox("Device", ["auto", "cuda", "cpu"])
        
        st.markdown("---")
        
        st.subheader("Reward Weights")
        st.markdown("*Adjust the relative importance of reward components*")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            w_power = st.slider("Power Reward", 0.0, 2.0, 1.0)
        with col2:
            w_alignment = st.slider("Alignment Reward", 0.0, 2.0, 1.0)
        with col3:
            w_control = st.slider("Control Penalty", 0.0, 2.0, 0.5)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            w_frequency = st.slider("Frequency Band Penalty", 0.0, 2.0, 0.5)
        with col2:
            w_stability = st.slider("Stability Bonus", 0.0, 2.0, 0.2)
        with col3:
            w_physics = st.slider("Physics Reward (Finesse)", 0.0, 2.0, 0.5)
    
    # ==========================================================================
    # SAVE / LOAD CONFIG
    # ==========================================================================
    st.markdown("---")
    st.header("💾 Save / Load Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config_name = st.text_input("Config Name", value="my_experiment")
        if st.button("💾 Save Config", type="primary"):
            # Collect all config
            config = {
                'algorithm': algorithm,
                'target': target,
                'training': {
                    'total_steps': total_steps,
                    'eval_freq': eval_freq,
                    'save_freq': save_freq,
                    'seed': seed,
                    'n_envs': n_envs,
                    'device': device
                },
                'environment': {
                    'episode_length': episode_length,
                    'control_rate': control_rate,
                    'max_misalignment': max_misalignment,
                    'domain_randomization': dr_enabled
                },
                'rewards': {
                    'power': w_power,
                    'alignment': w_alignment,
                    'control': w_control,
                    'frequency': w_frequency,
                    'stability': w_stability,
                    'physics': w_physics
                }
            }
            
            # Save to file
            config_path = Path(f"configs/{config_name}.yaml")
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            st.success(f"Saved to {config_path}")
    
    with col2:
        existing_configs = list(Path("configs").glob("*.yaml")) if Path("configs").exists() else []
        config_files = [f.stem for f in existing_configs]
        
        if config_files:
            load_config = st.selectbox("Load Config", config_files)
            if st.button("📂 Load"):
                st.info(f"Would load: configs/{load_config}.yaml")
        else:
            st.info("No saved configs yet")
    
    with col3:
        if st.button("🚀 Start Training", type="primary"):
            st.success("Training would start with current config!")
            st.balloons()
    
    return {
        'algorithm': algorithm,
        'target': target
    }


if __name__ == "__main__":
    render_config_page()
