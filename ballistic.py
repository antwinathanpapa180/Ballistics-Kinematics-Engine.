

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import random

class MasterpieceBallistics:
    def __init__(self, root):
        self.root = root
        self.root.title("Defense System Effectiveness Analyzer")
       
        # Core Physics
        self.G, self.RHO, self.C_D = 9.81, 1.225, 0.295
        self.MASS, self.AREA = 0.0095, np.pi * (0.00782 / 2)**2

        self.setup_ui()
        self.run_simulation() # Synchronized name

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
       
        controls = ttk.Frame(self.root, padding="15")
        controls.pack(side=tk.LEFT, fill=tk.Y)

        # Basic Controls
        ttk.Label(controls, text="Muzzle Velocity (m/s)", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.v_slider = ttk.Scale(controls, from_=300, to_=1200, orient=tk.HORIZONTAL, command=lambda x: self.run_simulation())
        self.v_slider.set(850); self.v_slider.pack(fill='x', pady=5)

        ttk.Label(controls, text="Launch Angle (deg)", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.a_slider = ttk.Scale(controls, from_=0, to_=45, orient=tk.HORIZONTAL, command=lambda x: self.run_simulation())
        self.a_slider.set(5.0); self.a_slider.pack(fill='x', pady=5)

        ttk.Label(controls, text="Headwind (-) / Tailwind (+)", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.w_slider = ttk.Scale(controls, from_=-30, to_=30, orient=tk.HORIZONTAL, command=lambda x: self.run_simulation())
        self.w_slider.set(0); self.w_slider.pack(fill='x', pady=5)

        ttk.Separator(controls, orient='horizontal').pack(fill='x', pady=10)

        # Target Settings
        ttk.Label(controls, text="Target Distance (m)", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.target_entry = ttk.Entry(controls, justify='center')
        self.target_entry.insert(0, "800")
        self.target_entry.pack(fill='x', pady=2)

        ttk.Label(controls, text="Kill Zone Size (m)", font=('Arial', 9, 'bold')).pack(anchor='w')
        self.kill_zone_entry = ttk.Entry(controls, justify='center')
        self.kill_zone_entry.insert(0, "2.0")
        self.kill_zone_entry.pack(fill='x', pady=2)

        ttk.Separator(controls, orient='horizontal').pack(fill='x', pady=10)

        # Mode Selection
        self.sim_mode = tk.StringVar(value="Monte Carlo")
        ttk.Radiobutton(controls, text="Precision Mode", variable=self.sim_mode, value="Single", command=self.run_simulation).pack(anchor='w')
        ttk.Radiobutton(controls, text="Weapon Test (100 Rds)", variable=self.sim_mode, value="Monte Carlo", command=self.run_simulation).pack(anchor='w')

        # Analysis Buttons
        ttk.Button(controls, text="🎯 AUTO-AIM", command=self.auto_aim).pack(pady=10, fill='x')

        # Advanced Metrics Display
        self.stats_frame = ttk.LabelFrame(controls, text=" Mission Stats ", padding="10")
        self.stats_frame.pack(fill='x', pady=15)
       
        self.prob_label = ttk.Label(self.stats_frame, text="Hit Probability: --%", font=('Arial', 10, 'bold'))
        self.prob_label.pack()
       
        self.prec_label = ttk.Label(self.stats_frame, text="Status: Ready", foreground="green")
        self.prec_label.pack()

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def get_derivatives(self, state, wind_x):
        x, y, vx, vy = state
        v_rel_x = vx - wind_x
        v_mag_rel = np.sqrt(v_rel_x**2 + vy**2)
        accel_drag = -(0.5 * self.RHO * v_mag_rel**2 * self.C_D * self.AREA) / self.MASS
        return np.array([vx, vy, (v_rel_x/v_mag_rel)*accel_drag, (vy/v_mag_rel)*accel_drag - self.G])

    def full_sim(self, v0, angle_deg, wind_x):
        angle = np.radians(angle_deg)
        state = np.array([0.0, 0.0, v0 * np.cos(angle), v0 * np.sin(angle)])
        dt = 0.08
        traj = [state]
        while state[1] >= 0 and state[0] < 6000:
            k1 = self.get_derivatives(state, wind_x)
            k2 = self.get_derivatives(state + k1 * dt / 2, wind_x)
            k3 = self.get_derivatives(state + k2 * dt / 2, wind_x)
            k4 = self.get_derivatives(state + k3 * dt, wind_x)
            state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(state)
        return np.array(traj)

    def run_simulation(self):
        self.ax.clear()
        v_base = self.v_slider.get()
        a_base = self.a_slider.get()
        wind_x = self.w_slider.get()
        target_x = float(self.target_entry.get())
        kill_zone = float(self.kill_zone_entry.get())

        if self.sim_mode.get() == "Single":
            traj = self.full_sim(v_base, a_base, wind_x)
            impact_x = traj[-1, 0]
            color = '#2ecc71' if abs(impact_x - target_x) <= kill_zone else '#e67e22'
            self.ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.5)
            self.prob_label.config(text="Mode: Single Precision")
            self.prec_label.config(text="Target Locked" if color == '#2ecc71' else "Adjust Aim", foreground="blue")
        else:
            hits = 0
            for _ in range(100):
                v_rand = random.gauss(v_base, 4.0)
                a_rand = random.gauss(a_base, 0.08)
                traj = self.full_sim(v_rand, a_rand, wind_x)
                impact_x = traj[-1, 0]
                hit = abs(impact_x - target_x) <= kill_zone
                if hit: hits += 1
                self.ax.plot(traj[:, 0], traj[:, 1], color='#2ecc71' if hit else '#e74c3c', alpha=0.15, lw=0.6)
           
            hit_prob = hits
            self.prob_label.config(text=f"Hit Probability: {hit_prob}%")
            self.update_status(hit_prob)

        self.ax.plot(target_x, 0, 'ko', markersize=12)
        self.ax.plot(target_x, 0, 'wo', markersize=8)
        self.ax.set_ylim(bottom=0)
        self.ax.set_title(f"Range Analysis: {target_x}m")
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def update_status(self, prob):
        if prob > 80: self.prec_label.config(text="Status: ELITE PRECISION", foreground="#27ae60")
        elif prob > 50: self.prec_label.config(text="Status: COMBAT EFFECTIVE", foreground="#f39c12")
        else: self.prec_label.config(text="Status: POOR ACCURACY", foreground="#c0392b")

    def auto_aim(self):
        try:
            target_x = float(self.target_entry.get())
            v0 = self.v_slider.get()
            wind = self.w_slider.get()
            a0, a1 = 1.0, 20.0
            for _ in range(15):
                f0 = self.full_sim(v0, a0, wind)[-1, 0] - target_x
                f1 = self.full_sim(v0, a1, wind)[-1, 0] - target_x
                if abs(f1 - f0) < 1e-4: break
                a_next = a1 - f1 * (a1 - a0) / (f1 - f0)
                a0, a1 = a1, a_next
            self.a_slider.set(max(0, min(45, a1)))
            self.run_simulation()
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MasterpieceBallistics(root)
    root.mainloop()