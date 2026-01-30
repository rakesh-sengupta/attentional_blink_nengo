"""
Biophysical Attentional Blink Model (Final Submission Version)
Target Journal: Journal of Computational Neuroscience

- Includes: Standard vs. Instant modes, High-fidelity tuning, Raster & Summary plots.
"""

import nengo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nengo.processes import Process
import scipy.stats as stats

# --- 1. BIOPHYSICAL CONSTANTS ---
DT = 0.001
N_NEURONS = 200
N_TRIALS = 100  # High trial count for smooth figures

# Synaptic Time Constants
TAU_AMPA = 0.005
TAU_GABA_A = 0.010
TAU_GABA_B = 0.150

# --- 2. NEURON CONFIGURATION (High Gain) ---
PyramidalType = nengo.AdaptiveLIF(
    tau_rc=0.02,
    tau_ref=0.002,
    tau_n=0.1,
    inc_n=0.1,
    min_voltage=-1
)
InterneuronType = nengo.LIF(tau_rc=0.015, tau_ref=0.001)

# --- 3. DELAY PROCESS ---
MEAN_LOOP_DELAY = 0.280

class DistributedAxonalDelay(Process):
    def __init__(self, n_neurons, mean_delay, std_delay=0.01, **kwargs):
        self.n_neurons = n_neurons
        self.delays = np.maximum(np.random.normal(mean_delay, std_delay, n_neurons), DT)
        self.max_steps = int(np.ceil(np.max(self.delays) / DT))
        self.delay_steps = (self.delays / DT).astype(int)
        super().__init__(default_size_in=n_neurons, default_size_out=n_neurons, **kwargs)

    def make_step(self, shape_in, shape_out, dt, rng, state):
        buffer = state['buffer']
        write_idx = state['write_idx']
        d_steps = self.delay_steps
        m_steps = self.max_steps
        n_neu = self.n_neurons

        def step_fn(t, x):
            w = int(write_idx[0])
            buffer[w] = x
            read_indices = (w - d_steps) % m_steps
            output = buffer[read_indices, np.arange(n_neu)]
            write_idx[0] = (w + 1) % m_steps
            return output
        return step_fn

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        return {
            'buffer': np.zeros((self.max_steps, self.n_neurons)),
            'write_idx': np.array([0], dtype=int)
        }

# --- 4. NETWORK CONSTRUCTION ---
class BiophysicalABModel:
    def __init__(self, mode="Standard", noise_scale=0.0, seed=42):
        self.mode = mode  # Store the mode (Standard vs Instant)
        self.noise_scale = noise_scale
        self.seed = seed

    def build_network(self, rsvp_func):
        model = nengo.Network(label="Biophysical Cortico-Thalamic Loop")
        with model:
            stim = nengo.Node(rsvp_func)

            # V4
            v4 = nengo.Ensemble(N_NEURONS, 3, neuron_type=PyramidalType, radius=1.0,
                                intercepts=nengo.dists.Uniform(-0.5, 0.5), label="V4")
            nengo.Connection(stim, v4, synapse=TAU_AMPA)

            # dlPFC
            pfc = nengo.Ensemble(N_NEURONS, 1, neuron_type=PyramidalType, radius=1.0,
                                 intercepts=nengo.dists.Uniform(0.0, 0.5), label="dlPFC")
            target_vec = np.array([1.0, 0.0, 0.0])
            nengo.Connection(v4, pfc, function=lambda x: np.dot(x, target_vec) * 2.0, synapse=TAU_AMPA)

            # TRN
            trn = nengo.Ensemble(N_NEURONS, 1, neuron_type=InterneuronType, radius=1.0, label="TRN")

            # --- CRITICAL: Mode Switching Logic ---
            if self.mode == "Standard":
                # Standard Mode: Use the DistributedAxonalDelay process
                tract_process = DistributedAxonalDelay(N_NEURONS, mean_delay=MEAN_LOOP_DELAY)
                tract_node = nengo.Node(tract_process, size_in=N_NEURONS, size_out=N_NEURONS)
                nengo.Connection(pfc.neurons, tract_node, synapse=None)
                nengo.Connection(tract_node, trn.neurons, synapse=TAU_AMPA)
            else:
                # Instant Mode: Direct connection (Control condition)
                nengo.Connection(pfc.neurons, trn.neurons, synapse=TAU_AMPA)

            # VWM
            vwm = nengo.Ensemble(N_NEURONS, 3, neuron_type=PyramidalType, radius=1.0, label="VWM")
            nengo.Connection(v4, vwm, synapse=TAU_AMPA)

            # Dual Inhibition
            nengo.Connection(trn, vwm.neurons, transform=np.full((N_NEURONS, 1), -15.0), synapse=TAU_GABA_A)
            nengo.Connection(trn, vwm.neurons, transform=np.full((N_NEURONS, 1), -5.0), synapse=TAU_GABA_B)

            # Probes
            self.p_vwm = nengo.Probe(vwm, synapse=0.05)
            self.p_spikes_pfc = nengo.Probe(pfc.neurons, 'output')
            self.p_spikes_trn = nengo.Probe(trn.neurons, 'output')

        return model

    def run_trial(self, lag):
        t1_onset = 0.5
        t2_onset = t1_onset + (lag * 0.1)

        def rsvp(t):
            if 0.5 <= t < 0.6: return np.array([2.0, 0, 0])
            if t2_onset <= t < t2_onset + 0.1: return np.array([0, 2.0, 0])
            return np.array([0, 0, 0])

        model = self.build_network(rsvp)
        sim = nengo.Simulator(model, dt=DT, progress_bar=False)
        sim.run(t2_onset + 0.4)

        data = sim.data[self.p_vwm]
        idx_start = int((t2_onset + 0.05)/DT)
        peak = np.max(data[idx_start:, 1])
        return peak, sim

# --- 5. FIGURE GENERATORS ---

def plot_mechanism(results_dict=None):
    print(f"\nGenerating Figure 3 (Raster from 1 random trial, Trace avg of {N_TRIALS})...")
    lags = [1, 3]
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2)
    model = BiophysicalABModel(mode="Standard", seed=42)
    
    for i, lag in enumerate(lags):
        vwm_traces = []
        example_sim = None
        example_pfc_probe = None
        example_trn_probe = None
        
        print(f"  > Simulating Lag {lag}...")
        for trial in range(N_TRIALS):
            _, sim = model.run_trial(lag)
            vwm_traces.append(sim.data[model.p_vwm][:, 1])
            
            if trial == 0:
                example_sim = sim
                example_pfc_probe = model.p_spikes_pfc
                example_trn_probe = model.p_spikes_trn
                
        mean_vwm = np.mean(vwm_traces, axis=0)
        t = example_sim.trange()
        
        # 1. dlPFC Spikes
        ax1 = plt.subplot(gs[0, i])
        s_pfc = example_sim.data[example_pfc_probe]
        active = np.where(np.sum(s_pfc, axis=0) > 2)[0]
        display = active[:50]
        if len(display) > 0:
            raster = [t[np.where(s_pfc[:, n] > 0)] for n in display]
            ax1.eventplot(raster, colors='black', linewidths=0.8)
        ax1.set_title(f"Lag {lag}: {'Sparing' if lag==1 else 'Blink'}")
        ax1.set_ylabel("dlPFC Spikes")
        ax1.set_xlim(0.4, 1.2)
        
        # 2. TRN Spikes
        ax2 = plt.subplot(gs[1, i])
        s_trn = example_sim.data[example_trn_probe]
        active = np.where(np.sum(s_trn, axis=0) > 2)[0]
        display = active[:50]
        if len(display) > 0:
            raster = [t[np.where(s_trn[:, n] > 0)] for n in display]
            ax2.eventplot(raster, colors='red', linewidths=0.8)
        ax2.set_ylabel("TRN Spikes")
        ax2.set_xlim(0.4, 1.2)
        t2_on = 0.5 + (lag * 0.1)
        ax2.axvspan(t2_on, t2_on+0.1, color='blue', alpha=0.1, label='T2 Stim')

        # 3. VWM Activity
        ax3 = plt.subplot(gs[2, i])
        ax3.plot(t, mean_vwm, 'b-', lw=2, label=f'Mean (n={N_TRIALS})')
        ax3.axhline(0.30, color='k', linestyle='--', alpha=0.5)
        ax3.set_ylabel("VWM Activity")
        ax3.set_xlabel("Time (s)")
        ax3.set_xlim(0.4, 1.2)
        ax3.set_ylim(0, max(0.6, np.max(mean_vwm)*1.2))

    plt.tight_layout()
    plt.savefig('mechanism_raster_averaged.png', dpi=300)
    print("Saved mechanism_raster_averaged.png")

def generate_summary_plot():
    print("\nGenerating Figure 2 (Standard vs. Instant Comparison)...")
    lags = [1, 2, 3, 4, 6, 8]
    
    # 1. Standard Model
    print("  > Running Standard Model...")
    model_std = BiophysicalABModel(mode="Standard")
    means_std, sems_std = [], []
    for lag in lags:
        peaks = [model_std.run_trial(lag)[0] for _ in range(N_TRIALS)]
        means_std.append(np.mean(peaks))
        sems_std.append(np.std(peaks, ddof=1) / np.sqrt(N_TRIALS))

    # 2. Instant Control
    print("  > Running Instant Suppression Control...")
    model_inst = BiophysicalABModel(mode="Instant") 
    means_inst, sems_inst = [], []
    for lag in lags:
        peaks = [model_inst.run_trial(lag)[0] for _ in range(N_TRIALS)]
        means_inst.append(np.mean(peaks))
        sems_inst.append(np.std(peaks, ddof=1) / np.sqrt(N_TRIALS))

    # 3. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(lags, means_std, 'o-', color='black', label='Standard (Delayed)', lw=2)
    plt.fill_between(lags, np.array(means_std)-np.array(sems_std), np.array(means_std)+np.array(sems_std), color='black', alpha=0.1)
    
    plt.plot(lags, means_inst, 's--', color='gray', label='Control (Instant)', lw=2)
    plt.fill_between(lags, np.array(means_inst)-np.array(sems_inst), np.array(means_inst)+np.array(sems_inst), color='gray', alpha=0.1)

    plt.xlabel('T1-T2 Lag')
    plt.ylabel('T2 Memory Strength')
    plt.title('Lag-1 Sparing Requires Transport Delay')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('mechanism_final.png', dpi=300)
    print("Saved mechanism_final.png")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Generate Raster (Fig 3)
    plot_mechanism()
    
    # 2. Generate Summary (Fig 2)
    generate_summary_plot()
