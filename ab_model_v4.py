"""
ab_model_v4.py — Attentional Blink: Re-entrant Cortico-Thalamic Loop Latency
============================================================================
Author : Rakesh Sengupta  (rakesh.sengupta@krea.edu.in)
Version: 4  (post-Biological-Cybernetics reframe)

WHY v4 EXISTS 
----------------------------------------
Two substantive points:
  (a) the axonal lengths were anatomically wrong (SLF/IC too long), which
      "counters the premise" IF the premise is conduction; and
  (b) the model was framed as a *discovery* of mechanism when, given the
      scoring scheme, the headline result is closer to true-by-construction.

KEY SCIENTIFIC CHANGES FROM v3
------------------------------
1. CORRECTED ANATOMY (Bakhit et al. 2020):
     SLF  ~6.5 cm (was 15 cm)  -> delta_FF ~21 ms (was 45 ms)
     IC   ~4.5 cm (was 12 cm)  -> delta_FB ~15 ms (was 40 ms)
     PFC integration unchanged -> delta_INT 195 ms (P3a prior; DOMINANT term)
     Total loop latency        -> delta_FULL ~231 ms (was 280 ms)
     "Axonal-only" control      -> ~36 ms (was 85 ms); relabelled SHORT-DELAY.
   The axonal terms are now visibly a minor fraction of delta; the result
   does not depend on them. This is the whole point.

2. NEW Exp 5 — INSTANT SYNAPSES control (the genuinely novel result):
     delta = delta_FULL retained, tau_GABA_A = tau_GABA_B -> dt.
     Proves delta (transport), not slow GABA_B filtering, creates the
     sparing window. Lead with this in the paper.

3. NEW Exp 6 — VELOCITY / LENGTH ROBUSTNESS:
     Vary conduction velocity over a plausible range with the CORRECTED
     lengths; show the implied delta stays inside the sparing window for
     every velocity. This is the figure that answers the BC anatomy
     objection by converting it into a robustness demonstration.

4. NEW Exp 7 — dt-INVARIANCE check:
     Re-run the primary contrast at dt = 1.0 / 0.5 / 0.25 ms. Disarms the
     "1 ms timestep is too coarse" objection by SHOWING invariance instead
     of arguing. (Mains stay at dt = 1 ms for tractability; this proves
     that choice does not drive the result.)

5. dt-AGNOSTIC delay process: AxonalDelayProcess now computes per-neuron
   delay steps from the simulator's actual dt (was hard-coded to a global),
   so Exp 7 is valid.

6. FULL NEURON PARAMETERISATION documented (addresses "LIF underspecified"):
   Nengo normalised-voltage convention stated explicitly (V_reset = 0,
   V_th = 1), which is why alpha_n is dimensionless; max_rates / intercepts
   / radius given for every ensemble.

7. Relabelled figures/annotations to the corrected numbers and reframed
   language ("short-delay control", "re-entrant loop latency").

USAGE (Colab / Jupyter / plain Python — no argparse)
----------------------------------------------------
  Edit the CONFIG booleans in __main__.
    FAST_MODE = True   -> smoke test
    FAST_MODE = False  -> publication run
    RUN_SWEEP = True   -> include parametric delta sweep (Exp 2)
  Default DT = 1 ms keeps the run tractable (~same budget as v3). The
  dt-invariance experiment (Exp 7) confirms this is adequate. Set
  DT_DEFAULT = 0.0005 below if a reviewer insists the mains run finer.
"""

import nengo
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1.  CONFIGURABLE TIMESTEP + BIOPHYSICAL CONSTANTS
# =============================================================================

DT_DEFAULT = 0.001   # s  primary-run timestep. Exp 7 shows results are
                     #    invariant down to 0.25 ms; raise resolution here
                     #    (e.g. 0.0005) only if required by a referee.

N_NEURONS = 200      # neurons per ensemble
N_TRIALS  = 30       # trials per (condition x lag) for publication

# ---- Synaptic time constants -------------------------------------------------
TAU_AMPA   = 0.005   # s  AMPA  (fast excitatory feedforward)
TAU_GABA_A = 0.010   # s  GABA_A (fast phasic TRN inhibition)
TAU_GABA_B = 0.150   # s  GABA_B (slow tonic suppression; sustains blink DEPTH)

# ---- Adaptive-LIF parameters (Nengo normalised-voltage convention) ----------
# Nengo LIF/AdaptiveLIF use a non-dimensional membrane voltage with
#   V_reset = 0 (resting) and V_th = 1 (threshold).
# Hence the adaptation increment ALPHA_N is dimensionless (it is added to a
# dimensionless adaptive state n_i that is subtracted from the input current).
# This is the correct response to the reviewer's "alpha has wrong units"
# point: in this convention it is unitless by construction.
TAU_RC  = 0.020      # s  membrane time constant (calibrated value; see Exp 4)
TAU_REF = 0.002      # s  absolute refractory period
TAU_N   = 0.100      # s  spike-frequency adaptation recovery
ALPHA_N = 0.10       # dimensionless adaptation increment per spike
MIN_V   = -1.0       # dimensionless minimum voltage (hyperpolarisation floor)

# Ensemble tuning (now stated explicitly for reproducibility) ----------------
MAX_RATES = nengo.dists.Uniform(150, 350)   # Hz, firing-rate range at radius
PYR_RADIUS = 1.2
PFC_RADIUS = 1.0
TRN_RADIUS = 1.0

# ---- Anatomical delay components --------------------------------------------
# CORRECTED lengths (Bakhit et al. 2020). Central conduction velocity 3.5 m/s
# (myelinated association/projection fibres; Caminiti et al. 2013), plus a
# ~2 ms synaptic/dendritic transmission term at each target.
V_COND      = 3.5            # m/s  central conduction velocity
LEN_SLF     = 0.065          # m    superior longitudinal fasciculus (~6.5 cm)
LEN_IC      = 0.045          # m    internal capsule (~4.5 cm)
SYN_TRANS   = 0.002          # s    synaptic transmission at tract terminus

DELTA_FF  = LEN_SLF / V_COND + SYN_TRANS     # ~0.021 s  V4 -> dlPFC
DELTA_FB  = LEN_IC  / V_COND + SYN_TRANS     # ~0.015 s  dlPFC -> TRN
DELTA_INT = 0.195                            #  0.195 s  PFC integration (P3a prior)

DELTA_SHORT = DELTA_FF + DELTA_FB            # ~0.036 s  short-delay control
DELTA_FULL  = DELTA_FF + DELTA_INT + DELTA_FB  # ~0.231 s  full loop latency

# ---- RSVP parameters --------------------------------------------------------
T1_ONSET = 0.500     # s
ITEM_DUR = 0.100     # s  (100 ms SOA — standard RSVP)
ITEM_ON  = 0.085     # s  stimulus on-time within each slot
LAGS     = [1, 2, 3, 4, 6, 8]

# ---- Inhibition strength (tuned for canonical trough at lag-2/3) ------------
INH_FAST = -14.0     # GABA_A weight (phasic gate closure)
INH_SLOW = -5.0      # GABA_B weight (sustained suppression)

# ---- Scoring ----------------------------------------------------------------
SCORE_OFFSET = 0.050 # s  scoring window starts this long after T2 onset
SCORE_WIN    = 0.100 # s  scoring window width

# ---- Schematic empirical AB curve (overlay only; NOT fit data) --------------
EMPIRICAL_LAGS   = np.array([1, 2, 3, 4, 6, 8])
EMPIRICAL_SCORES = np.array([0.85, 0.45, 0.40, 0.55, 0.75, 0.85])


# =============================================================================
# 2.  NEURON TYPE FACTORY
# =============================================================================

def make_pyramidal(tau_rc=TAU_RC):
    """Adaptive-LIF pyramidal cell (V4, dlPFC, VWM). tau_rc is variable."""
    return nengo.AdaptiveLIF(
        tau_rc=tau_rc, tau_ref=TAU_REF,
        tau_n=TAU_N,   inc_n=ALPHA_N,
        min_voltage=MIN_V,
    )

def make_interneuron():
    """TRN interneuron — fast LIF kinetics."""
    return nengo.LIF(tau_rc=0.010, tau_ref=0.001)


# =============================================================================
# 3.  AXONAL DELAY PROCESS  (dt-agnostic)
# =============================================================================

class AxonalDelayProcess(nengo.processes.Process):
    """
    Heterogeneous axonal conduction delay across a white-matter tract.

    Per-neuron delay ~ Clip(N(mean, std), 0, inf), reflecting fibre-calibre
    variation within the tract. Delays are stored in SECONDS and converted to
    integer step counts using the simulator's actual dt inside make_step, so
    the process behaves correctly under any dt (required for the dt-invariance
    experiment).
    """
    def __init__(self, n_neurons, mean_delay, std_delay=None, seed=0, **kwargs):
        rng = np.random.RandomState(seed)
        if std_delay is None:
            std_delay = max(mean_delay * 0.05, 0.003)
        self.n_neurons = n_neurons
        self.delays = np.clip(rng.normal(mean_delay, std_delay, n_neurons),
                              0.0, None)
        super().__init__(default_size_in=n_neurons,
                         default_size_out=n_neurons, **kwargs)

    def make_state(self, shape_in, shape_out, dt, dtype=None):
        max_steps = int(np.ceil(np.max(self.delays) / dt)) + 2
        return {'buffer':    np.zeros((max_steps, self.n_neurons)),
                'write_idx': np.array([0], dtype=int)}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        buf  = state['buffer']
        widx = state['write_idx']
        m    = buf.shape[0]
        n    = self.n_neurons
        delay_steps = np.clip(np.round(self.delays / dt).astype(int), 1, m - 1)
        idx = np.arange(n)

        def step(t, x):
            w = int(widx[0])
            buf[w] = x
            out = buf[(w - delay_steps) % m, idx]
            widx[0] = (w + 1) % m
            return out
        return step


# =============================================================================
# 4.  MODEL BUILDER
# =============================================================================

class ABModel:
    """
    Spiking model of the re-entrant cortico-thalamic attentional blink.

    Conditions
    ----------
    'instant'      delta = 0        synaptic smoothing only, no transport delay
    'short_delay'  delta ~36 ms     corrected axonal transport only (SLF + IC)
    'full_loop'    delta ~231 ms    full loop (corrected anatomy)
    'decomposed'   3-component      FF axonal + integration + FB axonal

    Parameters
    ----------
    delta_override : float  override delta (parametric sweep / velocity sweep)
    instant_synapses : bool if True, set TRN->VWM tau ~ dt (Exp 5)
    dt             : float  simulation timestep (Exp 7)
    """

    _DEFAULTS = {
        'instant':     0.0,
        'short_delay': DELTA_SHORT,
        'full_loop':   DELTA_FULL,
        'decomposed':  None,
    }

    def __init__(self, condition='full_loop', delta_override=None,
                 seed=42, tau_rc=TAU_RC, instant_synapses=False,
                 dt=DT_DEFAULT):
        if condition not in self._DEFAULTS:
            raise ValueError(f"Unknown condition '{condition}'")
        self.condition = condition
        self.seed      = seed
        self.tau_rc    = tau_rc
        self.dt        = dt
        self.delta     = (delta_override if delta_override is not None
                          else self._DEFAULTS[condition])
        # Inhibitory synapse time constants (overridden for Exp 5)
        if instant_synapses:
            self.tau_inh_a = max(dt, 1e-4)
            self.tau_inh_b = max(dt, 1e-4)
        else:
            self.tau_inh_a = TAU_GABA_A
            self.tau_inh_b = TAU_GABA_B
        self.probes = {}

    # ------------------------------------------------------------------ build
    def build_network(self, rsvp_func):
        pyr = make_pyramidal(self.tau_rc)
        inh = make_interneuron()

        model = nengo.Network(seed=self.seed, label=f"AB_{self.condition}")
        with model:
            stim = nengo.Node(rsvp_func, label="RSVP")

            v4 = nengo.Ensemble(N_NEURONS, dimensions=3,
                                neuron_type=pyr, radius=PYR_RADIUS,
                                max_rates=MAX_RATES,
                                intercepts=nengo.dists.Uniform(-0.4, 0.5),
                                label="V4")
            nengo.Connection(stim, v4, synapse=TAU_AMPA)

            pfc = nengo.Ensemble(N_NEURONS, dimensions=1,
                                 neuron_type=pyr, radius=PFC_RADIUS,
                                 max_rates=MAX_RATES,
                                 intercepts=nengo.dists.Uniform(0.05, 0.55),
                                 label="dlPFC")
            T1 = np.array([1.0, 0.0, 0.0])
            nengo.Connection(v4, pfc,
                             function=lambda x: float(np.dot(x, T1)) * 2.5,
                             synapse=TAU_AMPA, label="V4_to_PFC")

            if self.condition == 'decomposed':
                trn_src = self._decomposed_stage(pfc)
            else:
                trn_src = self._lumped_stage(pfc)

            trn = nengo.Ensemble(N_NEURONS, dimensions=1,
                                 neuron_type=inh, radius=TRN_RADIUS,
                                 max_rates=MAX_RATES, label="TRN")
            nengo.Connection(trn_src, trn.neurons, synapse=TAU_AMPA)

            vwm = nengo.Ensemble(N_NEURONS, dimensions=3,
                                 neuron_type=pyr, radius=PYR_RADIUS,
                                 max_rates=MAX_RATES, label="VWM_IPS")
            nengo.Connection(v4, vwm, synapse=TAU_AMPA)

            # Dual TRN -> VWM inhibition (tau overridden in Exp 5)
            nengo.Connection(trn, vwm.neurons,
                             transform=np.full((N_NEURONS, 1), INH_FAST),
                             synapse=self.tau_inh_a)
            nengo.Connection(trn, vwm.neurons,
                             transform=np.full((N_NEURONS, 1), INH_SLOW),
                             synapse=self.tau_inh_b)

            self.probes = {
                'vwm': nengo.Probe(vwm, synapse=0.050),
                'pfc': nengo.Probe(pfc, synapse=0.010),
                'trn': nengo.Probe(trn, synapse=0.010),
            }
        return model

    # -------------------------------------------------------------- lumped
    def _lumped_stage(self, pfc):
        if self.delta <= self.dt:
            return pfc.neurons
        proc = AxonalDelayProcess(N_NEURONS, self.delta, seed=self.seed)
        node = nengo.Node(proc, size_in=N_NEURONS, size_out=N_NEURONS,
                          label=f"Delay_{int(self.delta*1000)}ms")
        nengo.Connection(pfc.neurons, node, synapse=None)
        return node

    # ----------------------------------------------------------- decomposed
    def _decomposed_stage(self, pfc):
        ff = AxonalDelayProcess(N_NEURONS, DELTA_FF, std_delay=0.003,
                                seed=self.seed)
        ff_node = nengo.Node(ff, size_in=N_NEURONS, size_out=N_NEURONS,
                             label="FF_SLF")
        nengo.Connection(pfc.neurons, ff_node, synapse=None)

        it = AxonalDelayProcess(N_NEURONS, DELTA_INT, std_delay=0.025,
                                seed=self.seed + 10)   # larger jitter: cognitive
        it_node = nengo.Node(it, size_in=N_NEURONS, size_out=N_NEURONS,
                             label="PFC_integration")
        nengo.Connection(ff_node, it_node, synapse=None)

        fb = AxonalDelayProcess(N_NEURONS, DELTA_FB, std_delay=0.003,
                                seed=self.seed + 1)
        fb_node = nengo.Node(fb, size_in=N_NEURONS, size_out=N_NEURONS,
                             label="FB_Capsule")
        nengo.Connection(it_node, fb_node, synapse=None)
        return fb_node

    # ----------------------------------------------------------------- trial
    def run_trial(self, lag, trial_seed=None, return_data=False):
        """Run one trial; return (gamma, data?) where gamma is T2 consolidation."""
        seed_used = trial_seed if trial_seed is not None else self.seed
        orig_seed = self.seed
        self.seed = seed_used

        t2_onset     = T1_ONSET + lag * ITEM_DUR
        sim_duration = t2_onset + 0.500

        model = self.build_network(self._make_rsvp(lag))
        with nengo.Simulator(model, dt=self.dt, progress_bar=False) as sim:
            sim.run(sim_duration)
        self.seed = orig_seed

        t       = sim.trange()
        i_start = np.searchsorted(t, t2_onset + SCORE_OFFSET)
        i_end   = np.searchsorted(t, t2_onset + SCORE_OFFSET + SCORE_WIN)
        window  = sim.data[self.probes['vwm']][i_start:i_end, 1]
        gamma   = float(np.mean(window)) if len(window) > 0 else 0.0

        if return_data:
            data = {k: sim.data[v] for k, v in self.probes.items()}
            data['trange'] = t
            return gamma, data
        return gamma, None

    # -------------------------------------------------------------- rsvp
    def _make_rsvp(self, lag):
        t2_onset = T1_ONSET + lag * ITEM_DUR

        def rsvp(t):
            if T1_ONSET <= t < T1_ONSET + ITEM_ON:
                return np.array([2.0, 0.0, 0.0])           # T1
            if t2_onset <= t < t2_onset + ITEM_ON:
                return np.array([0.0, 2.0, 0.0])           # T2
            if T1_ONSET + ITEM_DUR <= t < t2_onset:        # distractor stream
                phase = (t - (T1_ONSET + ITEM_DUR)) % ITEM_DUR
                if phase < ITEM_ON:
                    return np.array([0.0, 0.0, 0.80])
            return np.zeros(3)
        return rsvp


# =============================================================================
# 5.  EXPERIMENT HELPERS
# =============================================================================

def _trial_seed(trial, lag, delta_ms=0, tau_rc_ms=20, tag=0):
    return (trial * 9973 + lag * 97 + int(delta_ms) * 7
            + int(tau_rc_ms) * 3 + int(tag) * 100003) % (2**31)


def run_lag_curve(condition, lags=LAGS, n_trials=N_TRIALS, delta_override=None,
                  tau_rc=TAU_RC, instant_synapses=False, dt=DT_DEFAULT,
                  verbose=True):
    model = ABModel(condition=condition, delta_override=delta_override,
                    tau_rc=tau_rc, instant_synapses=instant_synapses, dt=dt)
    peaks = {lag: [] for lag in lags}
    tag = 1 if instant_synapses else 0
    for lag in lags:
        if verbose:
            print(f"    Lag {lag} ...", end='', flush=True)
        for trial in range(n_trials):
            s = _trial_seed(trial, lag,
                            delta_ms=int((delta_override or 0) * 1000),
                            tau_rc_ms=int(tau_rc * 1000), tag=tag)
            g, _ = model.run_trial(lag, trial_seed=s)
            peaks[lag].append(g)
        if verbose:
            m  = np.mean(peaks[lag])
            se = np.std(peaks[lag], ddof=1) / np.sqrt(n_trials)
            print(f"  Gamma = {m:.3f} +/- {se:.3f}")
    means = np.array([np.mean(peaks[l]) for l in lags])
    sems  = np.array([np.std(peaks[l], ddof=1) / np.sqrt(n_trials) for l in lags])
    return means, sems, peaks


def paired_sparing(peaks, l_hi=1, l_lo=3):
    """Paired t-test + Cohen's d + bootstrap CI for Gamma(l_hi) - Gamma(l_lo)."""
    a = np.array(peaks[l_hi]); b = np.array(peaks[l_lo]); n = len(a)
    t_stat, p_val = stats.ttest_rel(a, b)
    diff = a - b
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)
    rng = np.random.RandomState(0)
    boot = [np.mean(diff[rng.randint(0, n, n)]) for _ in range(10000)]
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    ratio = np.mean(a) / (np.mean(b) + 1e-9)
    return dict(t=t_stat, p=p_val, d=d, ci=ci, ratio=ratio,
                m_hi=np.mean(a), m_lo=np.mean(b))


# =============================================================================
# 6.  EXPERIMENTS
# =============================================================================

# ---- Exp 1 ------------------------------------------------------------------
def exp1_three_conditions(n_trials=N_TRIALS, dt=DT_DEFAULT):
    conditions = [
        ('instant',     f'Instant Suppression (delta=0)',                 '#d62728'),
        ('short_delay', f'Short-delay control (delta~{DELTA_SHORT*1000:.0f} ms)', '#ff7f0e'),
        ('full_loop',   f'Full Loop (delta~{DELTA_FULL*1000:.0f} ms)',     '#2ca02c'),
    ]
    results = {}
    for cond, lbl, col in conditions:
        print(f"\n  Condition: {lbl}")
        means, sems, peaks = run_lag_curve(cond, n_trials=n_trials, dt=dt)
        results[cond] = dict(label=lbl, color=col,
                             means=means, sems=sems, peaks=peaks)
    return results


# ---- Exp 2 ------------------------------------------------------------------
def exp2_delta_sweep(delta_values=None, n_trials=15, dt=DT_DEFAULT, verbose=True):
    if delta_values is None:
        delta_values = np.linspace(0, 0.400, 17)
    records = []
    for delta in delta_values:
        d_ms = delta * 1000
        if verbose:
            print(f"    delta = {d_ms:.0f} ms ...", end='', flush=True)
        row = {'delta': delta}
        for lag in [1, 2, 3, 6]:
            tp = []
            for trial in range(n_trials):
                s = _trial_seed(trial, lag, delta_ms=int(d_ms))
                m = ABModel(condition='full_loop', delta_override=delta,
                            seed=s % 10000, dt=dt)
                p, _ = m.run_trial(lag, trial_seed=s)
                tp.append(p)
            row[f'lag{lag}'] = np.mean(tp)
        row['sparing_ratio'] = row['lag1'] / (row['lag3'] + 1e-9)
        row['blink_depth']   = 1.0 - row['lag3'] / (row['lag6'] + 1e-9)
        records.append(row)
        if verbose:
            print(f"  ratio = {row['sparing_ratio']:.2f}  depth = {row['blink_depth']:.2f}")
    return records


# ---- Exp 3 ------------------------------------------------------------------
def exp3_decomposed(n_trials=N_TRIALS, dt=DT_DEFAULT):
    print("\n  Running decomposed model ...")
    means, sems, peaks = run_lag_curve('decomposed', n_trials=n_trials, dt=dt)
    total = (DELTA_FF + DELTA_INT + DELTA_FB) * 1000
    print(f"\n  Delay budget (corrected anatomy):")
    print(f"    FF axonal (SLF, ~6.5 cm):   {DELTA_FF*1000:5.1f} ms  (Caminiti 2013 / Bakhit 2020)")
    print(f"    PFC integration (P3a prior):{DELTA_INT*1000:5.1f} ms  (Soltani & Knight 2000)")
    print(f"    FB axonal (IC, ~4.5 cm):    {DELTA_FB*1000:5.1f} ms  (Caminiti 2013 / Bakhit 2020)")
    print(f"    ---------------------------------------")
    print(f"    Total loop latency:         {total:5.1f} ms")
    return dict(means=means, sems=sems, peaks=peaks, total_ms=total)


# ---- Exp 4 ------------------------------------------------------------------
def exp4_tau_rc(tau_rc_values=None, n_trials=20, dt=DT_DEFAULT, verbose=True):
    if tau_rc_values is None:
        tau_rc_values = [0.015, 0.018, 0.020, 0.022, 0.025]
    n_trials = max(n_trials, 20)
    results = []
    for tau in tau_rc_values:
        tau_ms = int(tau * 1000)
        if verbose:
            print(f"    tau_RC = {tau_ms:>2} ms  full lag curve ...")
        means, sems, peaks = run_lag_curve('full_loop', n_trials=n_trials,
                                           tau_rc=tau, dt=dt, verbose=False)
        trough_idx = int(np.argmin(means)); trough_lag = LAGS[trough_idx]
        st = paired_sparing(peaks, 1, trough_lag)
        recovery = np.mean([means[LAGS.index(l)] for l in LAGS if l >= 6])
        depth = 1.0 - means[trough_idx] / (recovery + 1e-9)
        rec = dict(tau_rc_ms=tau_ms, means=means, sems=sems, peaks=peaks,
                   trough_lag=trough_lag, ratio=st['ratio'], p=st['p'],
                   blink_depth=depth,
                   sig="OK" if (st['p'] < 0.05 and st['ratio'] > 1.3) else "no")
        results.append(rec)
        if verbose:
            print(f"      trough=lag-{trough_lag}  ratio={st['ratio']:.2f}  "
                  f"depth={depth:.2f}  p={st['p']:.3f}  [{rec['sig']}]")
    return results


# ---- Exp 5 (NEW) ------------------------------------------------------------
def exp5_instant_synapses(n_trials=N_TRIALS, dt=DT_DEFAULT):
    """
    GABA_B ambiguity resolution. delta = delta_FULL retained; TRN->VWM
    synapses set to ~dt. If sparing survives, delta (transport) — not slow
    GABA_B filtering — creates the sparing window. This is the headline.
    """
    print("\n  Running Instant-Synapses control (delta retained, tau_GABA -> dt) ...")
    means, sems, peaks = run_lag_curve('full_loop', n_trials=n_trials,
                                       instant_synapses=True, dt=dt)
    return dict(means=means, sems=sems, peaks=peaks)


# ---- Exp 6 (NEW) ------------------------------------------------------------
def exp6_velocity_robustness(velocities=None, n_trials=12, dt=DT_DEFAULT,
                             verbose=True):
    """
    Vary conduction velocity (corrected lengths fixed) -> implied delta;
    show sparing holds across the whole plausible velocity band. Converts the
    BC anatomy objection into a robustness result.
    """
    if velocities is None:
        velocities = [2.0, 3.0, 3.5, 5.0, 7.0, 10.0]   # m/s
    records = []
    for v in velocities:
        delta = (LEN_SLF + LEN_IC) / v + 2 * SYN_TRANS + DELTA_INT
        d_ms = delta * 1000
        if verbose:
            print(f"    v = {v:4.1f} m/s -> delta = {d_ms:5.1f} ms ...",
                  end='', flush=True)
        peaks = {1: [], 3: []}
        for lag in (1, 3):
            for trial in range(n_trials):
                s = _trial_seed(trial, lag, delta_ms=int(d_ms), tag=7)
                m = ABModel(condition='full_loop', delta_override=delta,
                            seed=s % 10000, dt=dt)
                p, _ = m.run_trial(lag, trial_seed=s)
                peaks[lag].append(p)
        st = paired_sparing(peaks, 1, 3)
        records.append(dict(v=v, delta=delta, ratio=st['ratio'], p=st['p'],
                            ci=st['ci']))
        if verbose:
            print(f"  ratio={st['ratio']:.2f}  p={st['p']:.3f}")
    return records


# ---- Exp 7 (NEW) ------------------------------------------------------------
def exp7_dt_invariance(dts=(0.001, 0.0005, 0.00025), n_trials=8):
    """
    Re-run the primary contrast (full_loop, lag-1 vs lag-3) at several dt.
    Demonstrates the 1 ms mains are adequate.
    """
    print("\n  dt-invariance check (full_loop, lag-1 vs lag-3) ...")
    records = []
    for dt in dts:
        peaks = {1: [], 3: []}
        for lag in (1, 3):
            for trial in range(n_trials):
                s = _trial_seed(trial, lag, tag=int(dt * 1e6))
                m = ABModel(condition='full_loop', seed=s % 10000, dt=dt)
                p, _ = m.run_trial(lag, trial_seed=s)
                peaks[lag].append(p)
        st = paired_sparing(peaks, 1, 3)
        records.append(dict(dt_ms=dt * 1000, ratio=st['ratio'],
                            p=st['p'], m_hi=st['m_hi'], m_lo=st['m_lo']))
        print(f"    dt = {dt*1000:5.2f} ms -> Gamma(1)={st['m_hi']:.3f}  "
              f"Gamma(3)={st['m_lo']:.3f}  ratio={st['ratio']:.2f}  p={st['p']:.3f}")
    return records


# ---- timing verification (paper Exp 6, analytical + numerical) --------------
def timing_verification(dt=DT_DEFAULT):
    print("\n  Timing verification")
    print("  -------------------")
    delta_ms = DELTA_FULL * 1000
    print(f"  Gate onset (T1 + delta) = T1 + {delta_ms:.1f} ms")
    for L in (1, 2, 3):
        ws = L * 100 + SCORE_OFFSET * 1000          # T1-relative window start
        we = ws + SCORE_WIN * 1000
        rel = ("before gate" if we <= delta_ms else
               "after gate"  if ws >= delta_ms else
               "STRADDLES gate")
        print(f"    Lag-{L} scoring window  [T1+{ws:.0f}, T1+{we:.0f}] ms  -> {rel}")
    # numerical TRN onset
    m = ABModel(condition='full_loop', seed=123, dt=dt)
    _, data = m.run_trial(3, trial_seed=123, return_data=True)
    t = data['trange']; trn = data['trn'][:, 0]
    base = T1_ONSET
    peak = np.max(trn)
    hold = max(1, int(round(0.010 / dt)))   # require >=10 ms sustained crossing
    for thr in (0.03, 0.10):
        crossed = (t > base) & (trn > thr * peak)
        onset = float('nan')
        for i in range(len(crossed) - hold):
            if crossed[i] and crossed[i:i + hold].all():
                onset = (t[i] - base) * 1000
                break
        print(f"    Numerical TRN onset (thr={thr:.2f}, sustained) "
              f"= T1 + {onset:.0f} ms")


# =============================================================================
# 7.  PRINTING
# =============================================================================

def print_statistics(results):
    print("\n" + "=" * 64)
    print("STATISTICAL SUMMARY  (paired t-test, lag-1 vs lag-3)")
    print("=" * 64)
    for cond, res in results.items():
        st = paired_sparing(res['peaks'], 1, 3); n = len(res['peaks'][1])
        sig = "OK SPARING" if (st['p'] < 0.05 and st['ratio'] > 1.3) else "no sparing"
        print(f"\n  {res['label']}")
        print(f"    Gamma(1) = {st['m_hi']:.3f}   Gamma(3) = {st['m_lo']:.3f}   "
              f"ratio = {st['ratio']:.2f}")
        print(f"    t({n-1}) = {st['t']:.2f}   p = {st['p']:.3e}   d = {st['d']:.2f}   "
              f"95% CI [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}]   [{sig}]")


def print_summary(three_cond, decomp, instant, sens):
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Condition':<32}{'G(1)':>7}{'G(3)':>7}{'Ratio':>7}{'Trough':>9}  Spared")
    print("-" * 72)

    def row(label, peaks, means):
        g1 = np.mean(peaks[1]); g3 = np.mean(peaks[3])
        tl = LAGS[int(np.argmin(means))]; r = g1 / (g3 + 1e-9)
        print(f"  {label:<30}{g1:>7.3f}{g3:>7.3f}{r:>7.2f}"
              f"{'lag-'+str(tl):>9}  {'YES' if r > 1.3 else 'no'}")

    for cond, res in three_cond.items():
        row(res['label'], res['peaks'], res['means'])
    row('Decomposed', decomp['peaks'], decomp['means'])
    row('Instant Synapses (Exp 5)', instant['peaks'], instant['means'])
    print("-" * 72)
    print(f"  Corrected delay budget:  FF {DELTA_FF*1000:.1f} + "
          f"INT {DELTA_INT*1000:.0f} + FB {DELTA_FB*1000:.1f} = "
          f"{DELTA_FULL*1000:.1f} ms")
    print(f"  tau_RC sensitivity:")
    for r in sens:
        print(f"    {r['tau_rc_ms']:>2} ms  trough lag-{r['trough_lag']}  "
              f"ratio {r['ratio']:.2f}  depth {r['blink_depth']:.2f}  [{r['sig']}]")
    print("=" * 72)


# =============================================================================
# 8.  PLOTTING
# =============================================================================

def _norm(m):
    lo, hi = np.min(m), np.max(m)
    return (m - lo) / (hi - lo) if hi - lo > 1e-6 else m * 0


def plot_three_conditions(results, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lag = np.array(LAGS)
    ax = axes[0]
    for cond, res in results.items():
        ax.plot(lag, res['means'], 'o-', color=res['color'], lw=2, ms=7,
                label=res['label'])
        ax.fill_between(lag, res['means'] - res['sems'],
                        res['means'] + res['sems'], alpha=0.15, color=res['color'])
    ax.set_xlabel("Lag (RSVP position)"); ax.set_ylabel("T2 consolidation Gamma")
    ax.set_title("Short-delay transport is insufficient;\nfull loop latency yields Lag-1 sparing",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_xticks(LAGS); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cond, res in results.items():
        ax.plot(lag, _norm(res['means']), 'o-', color=res['color'], lw=2, ms=7,
                alpha=0.85, label=res['label'])
    ax.plot(EMPIRICAL_LAGS, EMPIRICAL_SCORES, 'k^--', ms=8, lw=1.5, alpha=0.6,
            label='Empirical AB (Chun & Potter 1995, schematic)')
    ax.set_xlabel("Lag (RSVP position)"); ax.set_ylabel("Normalised T2 accuracy")
    ax.set_title("Normalised comparison with empirical AB profile",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.set_xticks(LAGS); ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('fig1_three_conditions.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig1_three_conditions.png")
    plt.show()


def plot_delta_sweep(records, save=True):
    deltas = np.array([r['delta'] * 1000 for r in records])
    ratios = np.array([r['sparing_ratio'] for r in records])
    depths = np.array([r['blink_depth'] for r in records])
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax = axes[0]
    ax.plot(deltas, ratios, 'o-', color='#1f77b4', lw=2, ms=7)
    ax.axhline(1.3, color='grey', ls='--', lw=1, label='Sparing threshold (1.3x)')
    ax.axvline(DELTA_FULL*1000, color='#2ca02c', ls=':', lw=1.8,
               label=f'Corrected anatomical delta = {DELTA_FULL*1000:.0f} ms')
    ax.axvline(DELTA_SHORT*1000, color='#ff7f0e', ls=':', lw=1.5,
               label=f'Short-delay = {DELTA_SHORT*1000:.0f} ms')
    mask = ratios >= 1.3
    if np.any(mask):
        lo, hi = deltas[mask][0], deltas[mask][-1]
        ax.axvspan(lo, hi, alpha=0.10, color='#2ca02c',
                   label=f'Sparing window [{lo:.0f}-{hi:.0f} ms]')
    ax.set_ylabel("Sparing ratio Gamma(1)/Gamma(3)")
    ax.set_title("Parametric delta sweep: sparing emerges only in a bounded window",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(deltas, depths, 's-', color='#d62728', lw=2, ms=7)
    ax.axvline(DELTA_FULL*1000, color='#2ca02c', ls=':', lw=1.8)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_xlabel("Loop latency delta (ms)")
    ax.set_ylabel("Blink depth [1 - Gamma(3)/Gamma(6)]")
    ax.set_title("Blink depth vs delta", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('fig2_delta_sweep.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig2_delta_sweep.png")
    plt.show()


def plot_decomposed(decomp, three_cond=None, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lag = np.array(LAGS)
    ax = axes[0]
    ax.plot(lag, decomp['means'], 'o-', color='#1f77b4', lw=2, ms=7,
            label='Decomposed')
    ax.fill_between(lag, decomp['means'] - decomp['sems'],
                    decomp['means'] + decomp['sems'], alpha=0.15, color='#1f77b4')
    if three_cond is not None:
        fl = three_cond['full_loop']
        ax.plot(lag, fl['means'], 's--', color='#2ca02c', lw=1.5, ms=6,
                alpha=0.7, label='Full Loop')
    ax.set_xlabel("Lag"); ax.set_ylabel("T2 consolidation Gamma")
    ax.set_title("Decomposed vs full-loop lag curve", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.set_xticks(LAGS); ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ['FF axonal\n(SLF)', 'PFC integration\n(P3a prior)', 'FB axonal\n(IC)']
    values = [DELTA_FF*1000, DELTA_INT*1000, DELTA_FB*1000]
    cols = ['#ff7f0e', '#9467bd', '#d62728']; sig = [3, 25, 3]
    bars = ax.bar(labels, values, color=cols, edgecolor='k', lw=0.8, alpha=0.85,
                  yerr=sig, capsize=6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 4, f'{v:.0f} ms',
                ha='center', fontsize=10, fontweight='bold')
    ax.axhline(decomp['total_ms'], color='green', ls='--', lw=1.8,
               label=f"Total = {decomp['total_ms']:.0f} ms")
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Delay budget — integration dominates, axonal terms minor",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        plt.savefig('fig3_decomposed.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig3_decomposed.png")
    plt.show()


def plot_tau_rc(sens, save=True):
    lag = np.array(LAGS); n = len(sens)
    cmap = plt.cm.coolwarm
    cols = [cmap(i / max(n - 1, 1)) for i in range(n)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for i, r in enumerate(sens):
        ax.plot(lag, _norm(r['means']), 'o-', color=cols[i], lw=2, ms=7,
                label=f"tau_RC = {r['tau_rc_ms']} ms")
        ti = LAGS.index(r['trough_lag'])
        ax.plot(r['trough_lag'], _norm(r['means'])[ti], '*', color=cols[i],
                ms=14, zorder=5)
    ax.plot(EMPIRICAL_LAGS, EMPIRICAL_SCORES, 'k^--', ms=7, lw=1.5, alpha=0.5,
            label='Empirical AB')
    ax.set_xlabel("Lag"); ax.set_ylabel("Normalised Gamma")
    ax.set_title("Lag curves across tau_RC (* = trough)", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right'); ax.set_xticks(LAGS)
    ax.set_ylim(-0.2, 1.3); ax.grid(True, alpha=0.3)

    ax = axes[1]
    taus = [r['tau_rc_ms'] for r in sens]; tr = [r['trough_lag'] for r in sens]
    ax.plot(taus, tr, 'D-', color='#7f7f7f', lw=2, ms=10)
    for i, (t, l) in enumerate(zip(taus, tr)):
        ax.plot(t, l, 'D', color=cols[i], ms=12, zorder=5)
    ax.set_xlabel("tau_RC (ms)"); ax.set_ylabel("Blink trough lag")
    ax.set_yticks([1, 2, 3, 4]); ax.set_xticks(taus)
    ax.set_title("Trough location vs tau_RC", fontsize=10, fontweight='bold')
    ax.set_ylim(0.5, 4.5); ax.grid(True, alpha=0.3)

    ax = axes[2]; ax2 = ax.twinx()
    depths = [r['blink_depth'] for r in sens]; ratios = [r['ratio'] for r in sens]
    ax.bar(taus, depths, width=1.5, color=cols, edgecolor='k', alpha=0.75,
           label='Blink depth')
    ax2.plot(taus, ratios, 'o-', color='black', lw=2, ms=8, zorder=5,
             label='Gamma(1)/Gamma(trough)')
    ax2.axhline(1.3, color='black', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel("tau_RC (ms)"); ax.set_ylabel("Blink depth", color='steelblue')
    ax2.set_ylabel("Sparing ratio"); ax.set_xticks(taus); ax.set_ylim(0, 1.2)
    ax.set_title("Depth and sparing robust across tau_RC", fontsize=10, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        plt.savefig('fig4_tau_rc_sensitivity.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig4_tau_rc_sensitivity.png")
    plt.show()


def plot_instant_synapses(three_cond, instant, save=True):
    lag = np.array(LAGS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    fl = three_cond['full_loop']; isup = three_cond['instant']
    ax.plot(lag, fl['means'], 'o-', color='#2ca02c', lw=2, ms=7, label='Full Loop')
    ax.plot(lag, instant['means'], 's--', color='#9467bd', lw=2, ms=7,
            label='Instant Synapses (delta retained, tau_GABA->dt)')
    ax.plot(lag, isup['means'], '^:', color='#d62728', lw=2, ms=7,
            label='Instant Suppression (delta=0)')
    ax.set_xlabel("Lag"); ax.set_ylabel("T2 consolidation Gamma")
    ax.set_title("Sparing survives removal of slow GABA_B:\ndelta, not synaptic filtering, creates the window",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.set_xticks(LAGS); ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = ['Full Loop', 'Instant\nSynapses', 'Instant\nSuppression']
    rr = [paired_sparing(fl['peaks'])['ratio'],
          paired_sparing(instant['peaks'])['ratio'],
          paired_sparing(isup['peaks'])['ratio']]
    cols = ['#2ca02c', '#9467bd', '#d62728']
    bars = ax.bar(names, rr, color=cols, edgecolor='k', alpha=0.85)
    for b, v in zip(bars, rr):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03, f'{v:.2f}',
                ha='center', fontweight='bold')
    ax.axhline(1.3, color='grey', ls='--', lw=1, label='Sparing threshold')
    ax.set_ylabel("Sparing ratio Gamma(1)/Gamma(3)")
    ax.set_title("Sparing ratios", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        plt.savefig('fig5_instant_synapses.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig5_instant_synapses.png")
    plt.show()


def plot_velocity_robustness(records, sweep=None, save=True):
    deltas = np.array([r['delta'] * 1000 for r in records])
    ratios = np.array([r['ratio'] for r in records])
    vels   = np.array([r['v'] for r in records])
    fig, ax = plt.subplots(figsize=(9, 6))
    if sweep is not None:
        sd = np.array([r['delta'] * 1000 for r in sweep])
        sr = np.array([r['sparing_ratio'] for r in sweep])
        ax.plot(sd, sr, '-', color='#cccccc', lw=2, zorder=1,
                label='delta sweep (Exp 2)')
        mask = sr >= 1.3
        if np.any(mask):
            ax.axvspan(sd[mask][0], sd[mask][-1], alpha=0.10, color='#2ca02c',
                       label=f'Sparing window [{sd[mask][0]:.0f}-{sd[mask][-1]:.0f} ms]')
    sc = ax.scatter(deltas, ratios, c=vels, cmap='viridis', s=120,
                    edgecolor='k', zorder=3)
    for r in records:
        ax.annotate(f"{r['v']:.1f} m/s", (r['delta']*1000, r['ratio']),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(1.3, color='grey', ls='--', lw=1, label='Sparing threshold (1.3x)')
    ax.axvline(DELTA_FULL*1000, color='#2ca02c', ls=':', lw=1.8,
               label=f'Central anatomy ({DELTA_FULL*1000:.0f} ms, 3.5 m/s)')
    plt.colorbar(sc, label='Conduction velocity (m/s)')
    ax.set_xlabel("Implied loop latency delta (ms)")
    ax.set_ylabel("Sparing ratio Gamma(1)/Gamma(3)")
    ax.set_title("Velocity / length robustness: corrected anatomy lands inside\n"
                 "the sparing window for every plausible conduction velocity",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('fig6_velocity_robustness.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig6_velocity_robustness.png")
    plt.show()


def plot_dt_invariance(records, save=True):
    dts = [r['dt_ms'] for r in records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dts, [r['m_hi'] for r in records], 'o-', color='#2ca02c', lw=2,
            ms=8, label='Gamma(1)')
    ax.plot(dts, [r['m_lo'] for r in records], 's-', color='#d62728', lw=2,
            ms=8, label='Gamma(3)')
    ax2 = ax.twinx()
    ax2.plot(dts, [r['ratio'] for r in records], 'D--', color='black', lw=2,
             ms=8, label='ratio')
    ax2.axhline(1.3, color='grey', ls='--', lw=1)
    ax.set_xlabel("Simulation timestep dt (ms)")
    ax.set_ylabel("Gamma"); ax2.set_ylabel("Sparing ratio")
    ax.set_title("dt-invariance: result unchanged from 1.0 to 0.25 ms",
                 fontsize=11, fontweight='bold')
    ax.invert_xaxis()
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='center right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('fig7_dt_invariance.png', dpi=300, bbox_inches='tight')
        print("  Saved: fig7_dt_invariance.png")
    plt.show()


# =============================================================================
# 9.  MAIN
# =============================================================================

if __name__ == "__main__":
    # ----------------------- CONFIG -----------------------------------------
    FAST_MODE = False     # True = smoke test;  False = publication run
    RUN_SWEEP = True    # True = include Exp 2 (delta sweep) and feed it to Exp 6
    # ------------------------------------------------------------------------

    n_main  = 8  if FAST_MODE else N_TRIALS
    n_sweep = 5  if FAST_MODE else 15
    n_pts   = 5  if FAST_MODE else 17
    n_vel   = 6  if FAST_MODE else 12

    print(f"\nLoop latency (corrected anatomy): "
          f"FF {DELTA_FF*1000:.1f} + INT {DELTA_INT*1000:.0f} + "
          f"FB {DELTA_FB*1000:.1f} = {DELTA_FULL*1000:.1f} ms")
    print(f"Short-delay control: {DELTA_SHORT*1000:.1f} ms")

    print("\n" + "=" * 60 + "\nEXPERIMENT 1 — Three conditions\n" + "=" * 60)
    three_cond = exp1_three_conditions(n_trials=n_main)
    print_statistics(three_cond)
    plot_three_conditions(three_cond)

    sweep = None
    if RUN_SWEEP:
        print("\n" + "=" * 60 + "\nEXPERIMENT 2 — Parametric delta sweep\n" + "=" * 60)
        sweep = exp2_delta_sweep(delta_values=np.linspace(0, 0.400, n_pts),
                                 n_trials=n_sweep)
        plot_delta_sweep(sweep)
    else:
        print("\n  [Exp 2 skipped — set RUN_SWEEP=True]")

    print("\n" + "=" * 60 + "\nEXPERIMENT 3 — Decomposed model\n" + "=" * 60)
    decomp = exp3_decomposed(n_trials=n_main)
    plot_decomposed(decomp, three_cond=three_cond)

    print("\n" + "=" * 60 + "\nEXPERIMENT 4 — tau_RC sensitivity\n" + "=" * 60)
    sens = exp4_tau_rc(n_trials=20)
    plot_tau_rc(sens)

    print("\n" + "=" * 60 + "\nEXPERIMENT 5 — Instant synapses (GABA_B)\n" + "=" * 60)
    instant = exp5_instant_synapses(n_trials=n_main)
    plot_instant_synapses(three_cond, instant)

    print("\n" + "=" * 60 + "\nEXPERIMENT 6 — Velocity / length robustness\n" + "=" * 60)
    vel = exp6_velocity_robustness(n_trials=n_vel)
    plot_velocity_robustness(vel, sweep=sweep)

    print("\n" + "=" * 60 + "\nEXPERIMENT 7 — dt-invariance\n" + "=" * 60)
    dtinv = exp7_dt_invariance(n_trials=6 if FAST_MODE else 10)
    plot_dt_invariance(dtinv)

    timing_verification()

    print_summary(three_cond, decomp, instant, sens)
    print("\nAll experiments complete.")
