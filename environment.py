import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import os

class LightingControlEnv(gym.Env):
    def __init__(self, dataset_path, allowed_months):
        super().__init__()

        # ===== Load dataset =====
        self.data = pd.read_csv(dataset_path)
        self.allowed_months = allowed_months
        self.n_steps = len(self.data)
        self.month_indices = self.data[self.data["month"].isin(allowed_months)].index
        self.current_idx_pointer = 0

        # ===== Lighting constants =====
        self.room_area = 28
        self.power_per_light = 25
        self.luminous_efficacy = 105

        self.num_lights = int(np.ceil(self.room_area / 10))
        self.lux_per_light = (self.power_per_light * self.luminous_efficacy) / 10
        self.max_lux = self.lux_per_light * self.num_lights

        # ===== Action space =====
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ===== Observation space =====
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1, 1, 0, 0, 0, 0]),
            high=np.array([3, 3, 31, 12, 23, 59, 1, 4]),
            dtype=np.float32
        )

        # ===== Energy tracking =====
        self.full_brightness_hours = 0.0
        self.dimming_brightness_hours = 0.0

        # ===== Brightness tracking =====
        self.brightness_time_sum = 0.0
        self.total_on_hours = 0.0

        # ===== Step-level storage =====
        self.step_action_rewards = []
        self.step_comfort_rewards = []
        self.step_total_rewards = []

        # ===== Episode-level totals =====
        self.total_action_reward = 0.0
        self.total_comfort_reward = 0.0
        self.episode_total_reward = 0.0

        # ===== Episode-level energy =====
        self.energy_saving_percentage = 0.0
        self.energy_saving_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.current_idx_pointer >= len(self.month_indices):
            self.current_idx_pointer = 0  # reset for testing loop

        self.current_step = self.month_indices[self.current_idx_pointer]
        self.episode_month = self.data.iloc[self.current_step]["month"]

        self.full_brightness_hours = 0.0
        self.dimming_brightness_hours = 0.0
        self.brightness_time_sum = 0.0
        self.total_on_hours = 0.0

        self.step_action_rewards.clear()
        self.step_comfort_rewards.clear()
        self.step_total_rewards.clear()

        self.total_action_reward = 0.0
        self.total_comfort_reward = 0.0
        self.episode_total_reward = 0.0

        self.energy_saving_percentage = 0.0
        self.energy_saving_reward = 0.0

        self.episode_month = self.data.iloc[self.current_step]["month"]
        return self._get_state(), {}

    def _get_state(self):
        idx = min(self.current_step, self.n_steps - 1)
        row = self.data.iloc[idx]
        return np.array([
            row["time_of_day"],
            row["weather_condition"],
            row["day"],
            row["month"],
            row["hour"],
            row["minute"],
            row["motion_detected"],
            row["behaviour"]
        ], dtype=np.float32)

    def step(self, action):
        action = np.clip(float(action[0]), 0.0, 1.0)
        row = self.data.iloc[self.current_step]

        time_of_day = row["time_of_day"]
        weather = row["weather_condition"]
        motion = row["motion_detected"]
        behaviour = row["behaviour"]

        action_reward = 0
        comfort_reward = 0
        step_reward = 0
        light_on = action > 0.0

        # ===============================
        # 1️⃣ Action reward
        # ===============================
        if time_of_day in [0, 1]:  # morning / afternoon
            if weather in [0, 1]:  # clear / cloudy
                action_reward += 1 if not light_on else -1
            else:  # foggy / rainy
                if motion == 0:
                    action_reward += 1 if not light_on else -1
                else:
                    action_reward += -1 if not light_on else 1
        else:  # evening / night
            if motion == 0:
                action_reward += 1 if not light_on else -1
            else:
                action_reward += -1 if not light_on else 1

        # ===============================
        # 2️⃣ Comfort reward
        # ===============================
        current_lux = action * self.max_lux

        def in_range(min_lux, max_lux):
            return min_lux <= current_lux <= max_lux

        if behaviour == 1:  # walking
            if in_range(100, 150):
                comfort_reward = 1
        elif behaviour == 2:  # sleeping
            if in_range(0, 10):
                comfort_reward = 1
        elif behaviour == 3:  # eating
            if in_range(200, 500):
                comfort_reward = 1
        elif behaviour == 4:  # studying
            if in_range(400, 750):
                comfort_reward = 1

        step_reward = action_reward + comfort_reward

        # ===== Store step-level rewards =====
        self.step_action_rewards.append(action_reward)
        self.step_comfort_rewards.append(comfort_reward)
        self.step_total_rewards.append(step_reward)

        # ===== Accumulate episode rewards =====
        self.total_action_reward += action_reward
        self.total_comfort_reward += comfort_reward
        self.episode_total_reward += step_reward

        # ===============================
        # Dynamic step duration (43–44 min)
        # ===============================
        if self.current_step < self.n_steps - 1:
            curr = self.data.iloc[self.current_step]
            nxt = self.data.iloc[self.current_step + 1]

            curr_time = curr["hour"] * 60 + curr["minute"]
            next_time = nxt["hour"] * 60 + nxt["minute"]

            # Handle cross-midnight
            if next_time >= curr_time:
                duration_minutes = next_time - curr_time
            else:
                duration_minutes = (24 * 60 - curr_time) + next_time

            # Dataset resolution safeguard
            duration_minutes = np.clip(duration_minutes, 43, 44)
        else:
            duration_minutes = 44  # fallback for last step

        step_hours = duration_minutes / 60

        # ===============================
        # Energy tracking
        # ===============================
        if action == 1.0:
            self.full_brightness_hours += step_hours
        elif 0.0 < action < 1.0:
            self.dimming_brightness_hours += step_hours

        # ===== Brightness accumulation (for episode-level energy saving) =====
        if action > 0.0:
            self.brightness_time_sum += action * step_hours
            self.total_on_hours += step_hours

        # ===============================
        # Step forward (month-aware)
        # ===============================
        self.current_idx_pointer += 1

        if self.current_idx_pointer >= len(self.month_indices):
            done = True
        else:
            self.current_step = self.month_indices[self.current_idx_pointer]
            next_month = self.data.iloc[self.current_step]["month"]
            done = next_month != self.episode_month


        # ===============================
        # 3️⃣ Energy saving reward (episode end)
        # ===============================
        if done:
            total_days = self.data[self.data["month"] == self.episode_month]["day"].nunique()
            
            if self.total_on_hours > 0:
                avg_brightness = self.brightness_time_sum / self.total_on_hours
            else:
                avg_brightness = 0.0

            total_energy_saved = (
                (1 - (1.0 - avg_brightness))
                * total_days
                * self.power_per_light
                * self.num_lights
                * self.dimming_brightness_hours
            ) / (
                self.power_per_light
                * self.num_lights
                * (self.full_brightness_hours + self.dimming_brightness_hours + 1e-6)
                * total_days
            ) * 100

            self.energy_saving_percentage = total_energy_saved
            self.energy_saving_reward = 1 if 24 <= total_energy_saved <= 60 else -1

            self.episode_total_reward += self.energy_saving_reward

            terminated = True
            truncated = False
            info = {
                # Step-level
                "step_action_rewards": self.step_action_rewards,
                "step_comfort_rewards": self.step_comfort_rewards,
                "step_total_rewards": self.step_total_rewards,

                # Episode-level
                "total_action_reward": self.total_action_reward,
                "total_comfort_reward": self.total_comfort_reward,
                "energy_saving_percentage": self.energy_saving_percentage,
                "energy_saving_reward": self.energy_saving_reward,
                "episode_total_reward": self.episode_total_reward,

                # Meta
                "month": self.episode_month
            }

            return self._get_state(), self.episode_total_reward, True, False, info

        terminated = done
        truncated = False
        return self._get_state(), step_reward, terminated, truncated, {}