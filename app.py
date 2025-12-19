import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageEnhance
from stable_baselines3 import PPO
from environment import LightingControlEnv
import os
import streamlit as st

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Smart Lighting RL Demo", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 { padding-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("💡 Smart Lighting Control System")

# ==============================
# Load model & environment
# ==============================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_lighting_month1_tuned.zip")
    return PPO.load(model_path)

model = load_model()

dataset_path = os.path.join(
    os.path.dirname(__file__),  # app.py 所在目录
    "data",                     # 你把 CSV 放在 data/ 文件夹
    "dataset_env.csv"
)

env = LightingControlEnv(
    dataset_path=dataset_path,
    allowed_months=list(range(2, 13))
)



# ==============================
# Load light bulb image
# ==============================
@st.cache_data
def load_bulb():
    bulb_path = os.path.join(os.path.dirname(__file__), "assets", "light_bulb.jpg")
    return Image.open(bulb_path).convert("RGBA")

def render_light(brightness: float):
    """根据亮度渲染灯泡效果"""
    enhancer = ImageEnhance.Brightness(load_bulb())
    return enhancer.enhance(0.2 + brightness * 1.8)

# ==============================
# Image resource lists
# ==============================

# 只保留文件名
TIME_IMAGE_NAMES = ["morning.png", "afternoon.png", "evening.png", "night.png"]
TIME_LABELS = ["🌅 Morning", "☀️ Afternoon", "🌆 Evening", "🌙 Night"]

WEATHER_IMAGE_NAMES = ["clear.png", "cloudy.png", "foggy.png", "rain.png"]
WEATHER_LABELS = ["☀️ Clear", "☁️ Cloudy", "🌫️ Foggy", "🌧️ Rain"]

BEHAVIOUR_IMAGE_NAMES = ["walking.png", "sleep.png", "eating.png", "studying.png"]
BEHAVIOUR_LABELS = ["🚶 Walking", "😴 Sleeping", "🍽️ Eating", "📚 Studying"]

# ==============================
# 绝对路径拼接，Cloud 安全
# ==============================
def load_image_list(folder: str, file_names: list):
    """返回图片的绝对路径列表"""
    folder_path = os.path.join(os.path.dirname(__file__), "assets", folder)
    return [os.path.join(folder_path, f) for f in file_names]

TIME_IMAGES = load_image_list("time", TIME_IMAGE_NAMES)
WEATHER_IMAGES = load_image_list("weather", WEATHER_IMAGE_NAMES)
BEHAVIOUR_IMAGES = load_image_list("behaviour", BEHAVIOUR_IMAGE_NAMES)

# ==============================
# Sidebar
# ==============================
st.sidebar.header("🎮 Demo Control")

st.sidebar.markdown("**🤖 Algorithm:** PPO")

play_speed = st.sidebar.slider(
    "⏱️ Playback speed (sec per step)", 
    min_value=0.05, 
    max_value=2.0, 
    value=0.5,
    step=0.1
)

sample_rate = st.sidebar.slider(
    "📊 Chart Update Rate (every N steps)",
    min_value=1,
    max_value=50,
    value=10,
    step=1,
    help="Show every Nth step in real-time chart to reduce clutter"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Performance Metrics Explained")
st.sidebar.info("""
**Total Reward**: Cumulative reward for the month (higher = better)

**Energy Savings**: % of electricity saved vs. always-on lights (target: 24-60%)

**Comfort Score**: % of time lighting meets occupant needs (0-100%)
""")

# ==============================
# Play simulation
# ==============================
if st.button("▶️ Play Simulation (Feb–Dec)", type="primary"):

    history_months = []
    history_total_rewards = []
    history_energy_savings = []
    history_comfort_scores = []
    
    month_names = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    placeholder_header = st.empty()
    placeholder_main = st.empty()

    for episode in range(11):
        state, _ = env.reset()
        step_count = 0
        
        sampled_steps = []
        sampled_cumulative_rewards = []
        
        episode_reward = 0
        episode_comfort_hits = 0
        episode_comfort_attempts = 0
        
        current_month_name = month_names[episode]

        while True:
            action, _ = model.predict(state, deterministic=True)
            brightness = float(np.clip(action[0], 0.0, 1.0))

            state, reward, done, _, info = env.step(action)
            if done:
                episode_reward = float(reward)
            else:
                episode_reward += reward

            if int(state[6]) == 1:
                episode_comfort_attempts += 1
                comfort_reward_val = info.get("step_comfort_rewards", [0])[-1] if info.get("step_comfort_rewards") else 0
                if comfort_reward_val > 0:
                    episode_comfort_hits += 1

            if step_count % sample_rate == 0:
                sampled_steps.append(step_count)
                sampled_cumulative_rewards.append(episode_reward)

            time_of_day = min(int(state[0]), len(TIME_IMAGES)-1)
            weather = min(int(state[1]), len(WEATHER_IMAGES)-1)
            day = int(state[2])
            month = int(state[3])
            hour = int(state[4])
            minute = int(state[5])
            motion = int(state[6])
            behaviour = min(int(state[7]), len(BEHAVIOUR_IMAGES)-1)

            with placeholder_header.container():
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='color: white; margin: 0;'>
                                📅 Month: {current_month_name} | Day: {day} | Time: {hour:02d}:{minute:02d}
                            </h3>
                            <h3 style='color: white; margin: 0;'>
                                Episode: {episode + 1}/11 | Step: {step_count+1}
                            </h3>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with placeholder_main.container():
                left_col, right_col = st.columns([3, 2])

                with left_col:
                    st.markdown(f"### 📈 Current Month Performance: {current_month_name}")
                    
                    # 仅显示 Total Reward
                    st.metric("Total Reward", f"{episode_reward:.1f}")
                    
                    st.markdown("---")
                    
                    # [修改] 图表布局改为 3 行 1 列
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f"📊 Cumulative Reward - {current_month_name}",
                            "📅 Historical: Total Rewards",
                            "📅 Historical: Energy Savings"
                        ),
                        vertical_spacing=0.1,
                        specs=[[{"secondary_y": False}],
                               [{"secondary_y": False}],
                               [{"secondary_y": False}]]
                    )

                    # Row 1: Cumulative Reward
                    if len(sampled_cumulative_rewards) > 0:
                        fig.add_trace(go.Scatter(
                            x=sampled_steps,
                            y=sampled_cumulative_rewards,
                            mode='lines',
                            line=dict(color='#00b894', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(0,184,148,0.2)',
                            name="Cumulative Reward",
                            showlegend=False
                        ), row=1, col=1)

                    # Row 2: Historical Rewards
                    # Row 2: Historical Rewards
                    if len(history_total_rewards) > 0:
                        # 只显示已完成的月份，不包括当前正在进行的月份
                        fig.add_trace(go.Bar(
                            x=history_months,
                            y=history_total_rewards,
                            marker_color='#6c5ce7',
                            text=[f"{r:.0f}" for r in history_total_rewards],
                            textposition='outside',
                            name="Total Reward",
                            showlegend=False
                        ), row=2, col=1)

                    else:
                        fig.update_yaxes(range=[0, 100], row=2, col=1)
                        fig.add_annotation(
                            text="Waiting for first month...",
                            xref="x2", yref="y2",
                            x=0.5, y=50, showarrow=False,
                            font=dict(size=14, color="gray"),
                            row=2, col=1
                        )

                    # Row 3: Energy Savings
                    if len(history_energy_savings) > 0:
                        colors = ['green' if 24 <= e <= 60 else 'orange' for e in history_energy_savings]
                        fig.add_trace(go.Bar(
                            x=history_months,
                            y=history_energy_savings,
                            marker_color=colors,
                            text=[f"{e:.1f}%" for e in history_energy_savings],
                            textposition='outside',
                            name="Energy Saved",
                            showlegend=False
                        ), row=3, col=1)
                        
                        fig.add_hline(y=24, line_dash="dot", line_color="green", 
                                    row=3, col=1, opacity=0.3)
                        fig.add_hline(y=60, line_dash="dot", line_color="red", 
                                    row=3, col=1, opacity=0.3)
                    else:
                        fig.update_yaxes(range=[0, 100], row=3, col=1)
                        fig.add_annotation(
                            text="Waiting for first month...",
                            xref="x3", yref="y3",
                            x=0.5, y=50, showarrow=False,
                            font=dict(size=14, color="gray"),
                            row=3, col=1
                        )

                    # Axis Labels
                    fig.update_xaxes(title_text="Steps", row=1, col=1)
                    fig.update_xaxes(title_text="Month", row=2, col=1)
                    fig.update_xaxes(title_text="Month", row=3, col=1)
                    
                    fig.update_yaxes(title_text="Reward", row=1, col=1)
                    fig.update_yaxes(title_text="Total Reward", row=2, col=1)
                    fig.update_yaxes(title_text="Savings (%)", range=[0, 100], row=3, col=1)
                    
                    # 增加高度以适应3行图表
                    fig.update_layout(
                        height=900, 
                        template="plotly_white",
                        margin=dict(l=60, r=40, t=60, b=60),
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"perf_{episode}_{step_count}")

                with right_col:
                    st.markdown("### 🏠 Current State")
                    state_col1, state_col2, state_col3 = st.columns(3)
                    
                    with state_col1:
                        st.image(f"assets/time/{TIME_IMAGES[time_of_day]}", width=80)
                        st.caption(TIME_LABELS[time_of_day])
                    
                    with state_col2:
                        st.image(f"assets/weather/{WEATHER_IMAGES[weather]}", width=80)
                        st.caption(WEATHER_LABELS[weather])
                    
                    with state_col3:
                        if motion == 1:
                            st.image(f"assets/behaviour/{BEHAVIOUR_IMAGES[behaviour]}", width=80)
                            st.caption(BEHAVIOUR_LABELS[behaviour])
                        else:
                            st.markdown("<div style='height:80px; display:flex; align-items:center; justify-content:center; font-size:40px;'>🚪</div>", unsafe_allow_html=True)
                            st.caption("Empty")
                    
                    st.markdown("---")
                    
                    st.markdown("### 💡 Current Lighting")
                    light_col1, light_col2, light_col3 = st.columns([1,2,1])
                    with light_col2:
                        st.image(render_light(brightness), width=120)
                    
                    st.markdown(f"**Brightness Level:** {brightness:.1%}")
                    st.progress(brightness)
                    
                    st.markdown("---")
                    
                    st.markdown("### 📊 Step Details")
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("Step Reward", f"{reward:.2f}")
                    with stat_col2:
                        st.metric("Steps", step_count + 1)

            time.sleep(play_speed)
            step_count += 1

            if done:
                history_months.append(current_month_name)
                history_total_rewards.append(episode_reward)
                history_energy_savings.append(info.get('energy_saving_percentage', 0))
                
                comfort_score = (episode_comfort_hits / episode_comfort_attempts * 100) if episode_comfort_attempts > 0 else 0
                history_comfort_scores.append(comfort_score)
                break

    st.success("✅ Simulation completed for all months (Feb-Dec)!")
    st.balloons()
    
    st.markdown("---")
    st.markdown("## 🎯 Overall Performance Summary (Feb-Dec)")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        avg_reward = np.mean(history_total_rewards)
        st.metric("Average Monthly Reward", f"{avg_reward:.1f}", 
                 help="Average total reward across all months")
    with summary_col2:
        avg_energy = np.mean(history_energy_savings)
        target_met = "✅ Target Met" if 24 <= avg_energy <= 60 else "⚠️ Outside Target"
        st.metric("Average Energy Savings", f"{avg_energy:.1f}%", 
                 delta=target_met, help="Target range: 24-60%")
    with summary_col3:
        avg_comfort = np.mean(history_comfort_scores)
        st.metric("Average Comfort Score", f"{avg_comfort:.1f}%",
                 help="How often lighting matched occupant needs")
    
    st.markdown("### 📋 Detailed Monthly Breakdown")
    summary_df = pd.DataFrame({
        "Month": history_months,
        "Total Reward": [f"{r:.1f}" for r in history_total_rewards],
        "Energy Savings (%)": [f"{e:.1f}" for e in history_energy_savings],
        "Comfort Score (%)": [f"{c:.1f}" for c in history_comfort_scores],
        "Target Met": ["✅" if 24 <= e <= 60 else "❌" for e in history_energy_savings]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)