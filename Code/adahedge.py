"""
================================================================================
  Adaptive Hedge
  Machine Learning (Nicolò Cesa-Bianchi)
  University of Milan, A.Y. 2025/26

  Reference paper:
    "Adaptive Hedge" -- Tim van Erven, Peter D. Grünwald, Wouter M. Koolen,
    Steven de Rooij. NeurIPS 2011.
    https://arxiv.org/abs/1110.1877
================================================================================
"""

import os
import numpy as np

# Configure matplotlib to generate plots without opening a GUI window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
# Fixing the global seed guarantees that every run of this script produces the
# EXACT same synthetic loss sequences and therefore the EXACT same plots.
GLOBAL_RANDOM_SEED = 777
numpy_random_generator = np.random.default_rng(GLOBAL_RANDOM_SEED)

# ── Output paths ───────────────────────────────────────────────────────────────
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
PLOTS_OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "Plots")
REPORTS_OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "Reports")

# Ensure output directories exist
os.makedirs(PLOTS_OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(REPORTS_OUTPUT_DIRECTORY, exist_ok=True)

print("="*72)
print("  Adaptive Hedge Project")
print(f"  Random seed : {GLOBAL_RANDOM_SEED}")
print(f"  Plots saved : {PLOTS_OUTPUT_DIRECTORY}")
print("="*72)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 -- SYNTHETIC LOSS GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_stochastic_regime_losses(
    total_number_of_rounds: int,
    total_number_of_experts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a loss matrix for the STOCHASTIC regime.

    Setup:
        - Expert 0  : Bernoulli(0.3) -- best expert
        - Experts 1 to (K-1) : Bernoulli(0.5) -- sub-optimal experts
    The gap 0.2 between the best expert and all others is large enough that, 
    by the Law of Large Numbers, the best expert will be clearly superior 
    very quickly.

    Args:
        total_number_of_rounds  : T, the number of online learning rounds.
        total_number_of_experts : K, the number of available experts.
        rng                     : A seeded numpy Generator for reproducibility.

    Returns:
        loss_matrix : shape (T, K), dtype float64.  Each entry is in {0, 1}.
    """
    # Allocate the full matrix at once for efficiency
    loss_matrix = np.zeros(
        (total_number_of_rounds, total_number_of_experts), dtype=np.float64
    )

    # Expert 0: Bernoulli(0.3) -- the best expert, loses with probability 0.3
    loss_matrix[:, 0] = rng.binomial(
        n=1, p=0.3, size=total_number_of_rounds
    ).astype(np.float64)

    # Experts 1 ... K-1: Bernoulli(0.5) -- all sub-optimal
    for expert_index in range(1, total_number_of_experts):
        loss_matrix[:, expert_index] = rng.binomial(
            n=1, p=0.5, size=total_number_of_rounds
        ).astype(np.float64)

    return loss_matrix


def generate_adversarial_regime_losses(
    total_number_of_rounds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a loss matrix for the ADVERSARIAL regime (2 experts, best swaps).

    Setup:
        - t ≤ T/2 : Expert 0 ~ Bernoulli(0.3), Expert 1 ~ Bernoulli(0.7)
        - t > T/2 : Expert 0 ~ Bernoulli(0.7), Expert 1 ~ Bernoulli(0.3)

    The identity of the best expert changes at the halfway mark, so any 
    algorithm that has "converged" to Expert 0 in the first half will suffer
    high losses in the second half. The key insight is that the TOTAL best 
    expert in hindsight (over all T rounds) has mean loss ≈ 
    (0.3 · T/2 + 0.7 · T/2) / T = 0.5, so no expert is substantially better
    than 0.5 in hindsight -- yet AdaHedge should still achieve O(√(T ln K)) 
    regret (Theorem 5).

    Args:
        total_number_of_rounds  : T, the number of rounds.
        rng                     : A seeded numpy Generator.

    Returns:
        loss_matrix : shape (T, 2), dtype float64.
    """
    total_number_of_experts = 2
    loss_matrix = np.zeros(
        (total_number_of_rounds, total_number_of_experts), dtype=np.float64
    )

    # Compute the halfway boundary (integer division to get an exact round)
    halfway_round_boundary = total_number_of_rounds // 2

    # ── FIRST HALF: Expert 0 is good (p=0.3), Expert 1 is bad (p=0.7) ──────
    loss_matrix[:halfway_round_boundary, 0] = rng.binomial(
        n=1, p=0.3, size=halfway_round_boundary
    ).astype(np.float64)
    loss_matrix[:halfway_round_boundary, 1] = rng.binomial(
        n=1, p=0.7, size=halfway_round_boundary
    ).astype(np.float64)

    # ── SECOND HALF: Roles reversed -- Expert 1 is now good (p=0.3) ──────────
    remaining_rounds = total_number_of_rounds - halfway_round_boundary
    loss_matrix[halfway_round_boundary:, 0] = rng.binomial(
        n=1, p=0.7, size=remaining_rounds
    ).astype(np.float64)
    loss_matrix[halfway_round_boundary:, 1] = rng.binomial(
        n=1, p=0.3, size=remaining_rounds
    ).astype(np.float64)

    return loss_matrix


def generate_low_gap_stochastic_losses(
    total_number_of_rounds: int,
    total_number_of_experts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a loss matrix for the LOW-GAP STOCHASTIC regime.
    
    Setup:
        - Expert 0  : Bernoulli(0.49)  -- best expert, barely better
        - Experts 1...(K-1) : Bernoulli(0.50) -- sub-optimal by gap Δ = 0.01
    The gap 0.01 is smaller than in the stochastic regime.  By Theorem 9,
    AdaHedge still achieves O(1) regret (with a much larger constant), because
    the cumulative loss gap grows linearly at rate Δ · T.  A fixed-η Hedge with
    η tuned for the worst case (η = √(2 ln K / T)) will fail to exploit
    this gap because its learning rate is far too small to discriminate
    between experts whose mean losses differ by only 0.01.

    Args:
        total_number_of_rounds  : T.
        total_number_of_experts : K.
        rng                     : A seeded numpy Generator.

    Returns:
        loss_matrix : shape (T, K), dtype float64.
    """
    loss_matrix = np.zeros(
        (total_number_of_rounds, total_number_of_experts), dtype=np.float64
    )

    # Expert 0: Bernoulli(0.49) -- narrowly best
    loss_matrix[:, 0] = rng.binomial(
        n=1, p=0.49, size=total_number_of_rounds
    ).astype(np.float64)

    # Experts 1 ... K-1: Bernoulli(0.50)
    for expert_index in range(1, total_number_of_experts):
        loss_matrix[:, expert_index] = rng.binomial(
            n=1, p=0.50, size=total_number_of_rounds
        ).astype(np.float64)

    return loss_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 -- HEDGE WITH FIXED LEARNING RATE
# ══════════════════════════════════════════════════════════════════════════════

def run_hedge_fixed_learning_rate(
    loss_matrix: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Run the standard Hedge algorithm with a fixed learning rate η.

    Algorithm:
        Initialise: w_1^k = 1/K for all k  (uniform prior)
        For t = 1, 2, ..., T:
            1. Output the probability vector w_t.
            2. Observe the loss vector ℓ_t ∈ [0,1]^K.
            3. Update cumulative losses: L_t^k = L_{t-1}^k + ℓ_t^k
            4. Recompute weights: w_{t+1}^k ∝ exp(−η · L_t^k)

    IMPORTANT: We compute weights from scratch (from cumulative losses)
    rather than incrementally (multiplying by exp(−η ℓ_t^k)) to avoid
    accumulated floating-point drift over thousands of rounds.

    Args:
        loss_matrix   : shape (T, K) array of per-round per-expert losses in [0,1].
        learning_rate : η > 0, the fixed learning rate.

    Returns:
        A dict with keys:
            'cumulative_loss_of_learner'      : shape (T,) -- learner's cum. loss per round
            'cumulative_loss_of_best_expert'  : shape (T,) -- best expert's cum. loss
            'cumulative_regret_of_learner'    : shape (T,) -- regret at each round
            'expert_probability_allocation_weights' : shape (T+1, K) -- weight history
    """
    total_number_of_rounds, total_number_of_experts = loss_matrix.shape

    # ── Allocate output arrays ────────────────────────────────────────────────
    # We store cumulative (prefix-sum) quantities, indexed by round t (1-indexed
    # in the paper, 0-indexed here for numpy convenience).
    accumulated_loss_of_individual_experts = np.zeros(
        total_number_of_experts, dtype=np.float64
    )
    cumulative_loss_of_learner_over_time = np.zeros(
        total_number_of_rounds, dtype=np.float64
    )
    cumulative_loss_of_best_expert_over_time = np.zeros(
        total_number_of_rounds, dtype=np.float64
    )
    # Store the full weight trajectory for optional analysis
    expert_probability_allocation_weight_history = np.zeros(
        (total_number_of_rounds + 1, total_number_of_experts), dtype=np.float64
    )

    # ── Initialise weights: uniform prior  ────────────────────────────────────
    initial_uniform_probability = 1.0 / total_number_of_experts
    current_expert_probability_weights = np.full(
        total_number_of_experts, initial_uniform_probability, dtype=np.float64
    )
    expert_probability_allocation_weight_history[0] = current_expert_probability_weights

    # Running sum of the learner's cumulative loss (used to compute regret)
    running_cumulative_learner_loss = 0.0

    # ── Main learning loop ────────────────────────────────────────────────────
    for round_index in range(total_number_of_rounds):

        # Step 1: The learner outputs w_t and incurs expected loss w_t · ℓ_t
        per_round_learner_loss = np.dot(
            current_expert_probability_weights, loss_matrix[round_index]
        )
        running_cumulative_learner_loss += per_round_learner_loss
        cumulative_loss_of_learner_over_time[round_index] = running_cumulative_learner_loss

        # Step 2: Update cumulative expert losses  L_t^k = L_{t-1}^k + ℓ_t^k
        accumulated_loss_of_individual_experts += loss_matrix[round_index]

        # Step 3: Compute new weights using the log-sum-exp trick for stability.
        #
        #   Raw exponent for each expert: x_k = −η · L_t^k
        #   Without stability: w ∝ exp(x)  ← may overflow/underflow
        #   With stability:    w ∝ exp(x − max(x))  ← all terms ≤ 1 OK
        #
        #   The normalisation constant cancels when dividing, so the trick is
        #   mathematically exact (not an approximation).
        raw_log_weight_unnormalised = (
            -learning_rate * accumulated_loss_of_individual_experts
        )
        # Shift by max to prevent overflow (log-sum-exp trick)
        log_weight_stabilisation_shift = np.max(raw_log_weight_unnormalised)
        stabilised_unnormalised_weights = np.exp(
            raw_log_weight_unnormalised - log_weight_stabilisation_shift
        )
        # Normalise to get a valid probability distribution
        normalisation_constant = np.sum(stabilised_unnormalised_weights)
        current_expert_probability_weights = (
            stabilised_unnormalised_weights / normalisation_constant
        )
        expert_probability_allocation_weight_history[round_index + 1] = (
            current_expert_probability_weights
        )

    # ── Compute regret ────────────────────────────────────────────────────────
    # Regret_T = Σ_t w_t · ℓ_t − min_k L_T^k 
    #
    # We build the cumulative best-expert loss as a running minimum over
    # the cumulative per-expert losses.  At each round t, we compare all
    # experts' cumulative losses and take the minimum.
    cumulative_expert_losses_over_time = np.cumsum(loss_matrix, axis=0)
    cumulative_loss_of_best_expert_over_time = np.min(
        cumulative_expert_losses_over_time, axis=1
    )
    cumulative_regret_of_learner_over_time = (
        cumulative_loss_of_learner_over_time
        - cumulative_loss_of_best_expert_over_time
    )

    return {
        "cumulative_loss_of_learner": cumulative_loss_of_learner_over_time,
        "cumulative_loss_of_best_expert": cumulative_loss_of_best_expert_over_time,
        "cumulative_regret_of_learner": cumulative_regret_of_learner_over_time,
        "expert_probability_allocation_weights": (
            expert_probability_allocation_weight_history
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 -- ADAHEDGE (ADAPTIVE LEARNING RATE WITH DOUBLING TRICK)
# ══════════════════════════════════════════════════════════════════════════════

def run_adahedge_algorithm(
    loss_matrix: np.ndarray,
    segment_learning_rate_decay_factor: float = 2.0,
) -> dict:
    """
    Run the AdaHedge algorithm.

    The algorithm divides the T rounds into segments. Within segment i it runs
    Hedge with a fixed learning rate. A new segment starts whenever the 
    accumulated mixability gap reaches the budget.

    IMPLEMENTATION: LOG-PARTITION RATIO FORMULATION (no incremental weights)
        Instead of maintaining w_t incrementally and taking log(w_t) to evaluate
        the mixability gap, we track the CUMULATIVE LOSSES WITHIN THE CURRENT
        SEGMENT (S^k) and compute everything from scratch via log-sum-exp.

    WHY THE INCREMENTAL APPROACH IS BUGGY:
        After ~710 rounds of an expert consistently losing (eta=1), that
        expert's weight underflows to 0.0 in float64.  Clipping to 1e-300
        before taking log substitutes -690 for the correct -inf.  The
        algorithm then thinks the losing expert has weight ~1e-300 instead of
        the correct ~0, causing a premature weight switch ~3000 rounds early.
        This produces the artifact of AdaHedge "beating" the best fixed expert
        (impossible by definition of regret). Regret is negative, which is a 
        clear sign of a bug. Regret = -1194.74

    THE FIX -- KEY IDENTITY:
        delta_t = w_t . ell_t + (1/eta) * log(w_t . exp(-eta ell_t))
                = w_t . ell_t + (1/eta) * [log(Z_t) - log(Z_{t-1})]

        where  Z_{t-1} = sum_k exp(-eta * S^k)              (before round t)
               Z_t     = sum_k exp(-eta * (S^k + ell_t^k))  (after round t)

        Both log-Z values are computed via stable log-sum-exp on the raw
        segment losses.  No log of any individual weight is ever needed.
        log(Z_t) is cached and reused as log(Z_{t-1}) for the next round.

    Args:
        loss_matrix                        : shape (T, K), losses in [0,1].
        segment_learning_rate_decay_factor : phi > 1.  Default = 2 (paper).

    Returns dict with keys:
        'cumulative_loss_of_learner'            : shape (T,)
        'cumulative_loss_of_best_expert'        : shape (T,)
        'cumulative_regret_of_learner'          : shape (T,)
        'expert_probability_allocation_weights' : shape (T+1, K)
        'learning_rate_at_each_round'           : shape (T,)
        'segment_boundary_round_indices'        : list of int
        'total_number_of_segments_used'         : int
    """
    total_number_of_rounds, total_number_of_experts = loss_matrix.shape

    # -- Output arrays ---------------------------------------------------------
    cumulative_loss_of_learner_over_time = np.zeros(
        total_number_of_rounds, dtype=np.float64
    )
    expert_probability_allocation_weight_history = np.zeros(
        (total_number_of_rounds + 1, total_number_of_experts), dtype=np.float64
    )
    learning_rate_at_each_round = np.zeros(
        total_number_of_rounds, dtype=np.float64
    )
    segment_boundary_round_indices = []

    # -- AdaHedge state --------------------------------------------------------
    # Algorithm starts with eta = phi, then immediately divides by phi at the
    # first segment, giving eta_1 = 1.
    current_learning_rate_eta = segment_learning_rate_decay_factor
    accumulated_mixability_gap_within_segment = 0.0
    mixability_gap_budget_limit_for_segment   = 0.0   # set on first reset
    running_cumulative_learner_loss           = 0.0
    eulers_number = np.exp(1.0)

    # -- Per-segment cumulative losses (KEY DATA STRUCTURE) --------------------
    #
    # S^k = sum of ell_s^k for all s in the current segment before round t.
    # Weights at round t:  w_t^k = exp(-eta * S^k) / Z_{t-1}
    # Reset to zeros at each new segment (equivalent to resetting w to 1/K).
    segment_cumulative_losses_per_expert = np.zeros(
        total_number_of_experts, dtype=np.float64
    )

    # Cached log(Z_{t-1}):  at a fresh segment start, S^k=0 for all k,
    # so Z_0 = sum_k exp(0) = K, hence log(Z_0) = log(K).
    log_normalisation_constant_at_previous_round = np.log(
        float(total_number_of_experts)
    )

    # -- Main loop -------------------------------------------------------------
    for round_index in range(total_number_of_rounds):

        # ---- Segment reset check ---------------------------------------------
        # Condition: "if T=1 or Delta >= b(eta)"
        should_start_new_segment = (
            round_index == 0
            or accumulated_mixability_gap_within_segment
               >= mixability_gap_budget_limit_for_segment
        )

        if should_start_new_segment:
            # eta <- eta / phi
            current_learning_rate_eta /= segment_learning_rate_decay_factor

            # Budget b(eta) = (1/(e-1) + 1/eta) * ln(K)
            mixability_gap_budget_limit_for_segment = (
                (1.0 / (eulers_number - 1.0) + 1.0 / current_learning_rate_eta)
                * np.log(total_number_of_experts)
            )

            accumulated_mixability_gap_within_segment = 0.0

            # Reset segment losses (= uniform weights 1/K)
            segment_cumulative_losses_per_expert = np.zeros(
                total_number_of_experts, dtype=np.float64
            )

            # Reset cached log-Z: Z_0 = K => log(K)
            log_normalisation_constant_at_previous_round = np.log(
                float(total_number_of_experts)
            )

            segment_boundary_round_indices.append(round_index)

        learning_rate_at_each_round[round_index] = current_learning_rate_eta

        # ---- Compute weights from scratch via log-sum-exp -------------------
        #
        # w_t^k = exp(-eta * S^k) / Z_{t-1}
        # shift = max(-eta * S^k) keeps all exp() arguments <= 0.
        # No clipping of any weight is ever needed.
        raw_log_weight_per_expert = (
            -current_learning_rate_eta * segment_cumulative_losses_per_expert
        )
        log_shift_weights = np.max(raw_log_weight_per_expert)
        unnormalised_weights = np.exp(
            raw_log_weight_per_expert - log_shift_weights
        )
        current_expert_probability_weights = (
            unnormalised_weights / np.sum(unnormalised_weights)
        )
        expert_probability_allocation_weight_history[round_index] = (
            current_expert_probability_weights
        )

        # ---- Learner loss:  w_t . ell_t -------------------------------------
        per_round_learner_loss = np.dot(
            current_expert_probability_weights, loss_matrix[round_index]
        )
        running_cumulative_learner_loss += per_round_learner_loss
        cumulative_loss_of_learner_over_time[round_index] = (
            running_cumulative_learner_loss
        )

        # ---- Update segment cumulative losses --------------------------------
        # S^k <- S^k + ell_t^k  for all k.
        # Done BEFORE computing log(Z_t) so that Z_t uses the updated losses.
        segment_cumulative_losses_per_expert += loss_matrix[round_index]

        # ---- Mixability gap via log-partition ratio --------------------------
        #
        # delta_t = w_t . ell_t + (1/eta) * log(w_t . exp(-eta ell_t))
        #
        # Key identity (derivation via marginal likelihood factorisation):
        #   log(w_t . exp(-eta ell_t))
        #     = log[ sum_k exp(-eta S^k) * exp(-eta ell_t^k) ] - log(Z_{t-1})
        #     = log[ sum_k exp(-eta * (S^k + ell_t^k)) ] - log(Z_{t-1})
        #     = log(Z_t) - log(Z_{t-1})
        #
        # By Jensen's inequality (log is concave), log(Z_t) <= log(Z_{t-1}),
        # so log(Z_t) - log(Z_{t-1}) <= 0, hence delta_t >= 0 (Lemma 1).
        raw_log_weight_per_expert_updated = (
            -current_learning_rate_eta * segment_cumulative_losses_per_expert
        )
        log_shift_updated = np.max(raw_log_weight_per_expert_updated)
        log_normalisation_constant_at_current_round = (
            log_shift_updated
            + np.log(np.sum(np.exp(
                raw_log_weight_per_expert_updated - log_shift_updated
            )))
        )

        # log(w_t . exp(-eta ell_t)) = log(Z_t) - log(Z_{t-1})
        stable_log_mixture_likelihood = (
            log_normalisation_constant_at_current_round
            - log_normalisation_constant_at_previous_round
        )

        mixability_gap_at_current_round = (
            per_round_learner_loss
            + (1.0 / current_learning_rate_eta) * stable_log_mixture_likelihood
        )
        accumulated_mixability_gap_within_segment += mixability_gap_at_current_round

        # Cache log(Z_t) => becomes log(Z_{t-1}) for the next round
        log_normalisation_constant_at_previous_round = (
            log_normalisation_constant_at_current_round
        )

    # -- Store final weight vector ---------------------------------------------
    raw_final  = -current_learning_rate_eta * segment_cumulative_losses_per_expert
    unnorm_fin = np.exp(raw_final - np.max(raw_final))
    expert_probability_allocation_weight_history[total_number_of_rounds] = (
        unnorm_fin / np.sum(unnorm_fin)
    )

    # -- Compute cumulative regret --------------------------------------------
    cumulative_expert_losses_over_time = np.cumsum(loss_matrix, axis=0)
    cumulative_loss_of_best_expert_over_time = np.min(
        cumulative_expert_losses_over_time, axis=1
    )
    cumulative_regret_of_learner_over_time = (
        cumulative_loss_of_learner_over_time
        - cumulative_loss_of_best_expert_over_time
    )

    total_number_of_segments_used = len(segment_boundary_round_indices)

    return {
        "cumulative_loss_of_learner":            cumulative_loss_of_learner_over_time,
        "cumulative_loss_of_best_expert":        cumulative_loss_of_best_expert_over_time,
        "cumulative_regret_of_learner":          cumulative_regret_of_learner_over_time,
        "expert_probability_allocation_weights": expert_probability_allocation_weight_history,
        "learning_rate_at_each_round":           learning_rate_at_each_round,
        "segment_boundary_round_indices":        segment_boundary_round_indices,
        "total_number_of_segments_used":         total_number_of_segments_used,
    }



# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 -- HELPER: COMPUTE THEORETICAL REGRET BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_adahedge_theoretical_regret_bound(
    cumulative_best_expert_losses: np.ndarray,
    total_number_of_experts: int,
    segment_learning_rate_decay_factor: float = 2.0,
) -> np.ndarray:
    """
    Compute the AdaHedge worst-case theoretical regret bound per round t.

    Full bound from Theorem 5 (van Erven et al. 2011):
        R_T  <=  C(φ) · √(4/(e-1) · L*_T · ln K)  +  (φ/(φ-1)) · (1/(e-1) + 1) · ln K

    The additive O(ln K) correction dominates when L*_t is small (early
    rounds).  Without it, the leading-order square-root term underestimates
    the true bound and AdaHedge's empirical regret can appear to exceed its
    own theoretical guarantee -- a misleading plotting artefact.

    Args:
        cumulative_best_expert_losses  : L*_t for each round t, shape (T,).
        total_number_of_experts        : K.
        segment_learning_rate_decay_factor : φ.

    Returns:
        theoretical_regret_upper_bound : shape (T,), full bound per t.
    """
    phi = segment_learning_rate_decay_factor
    eulers_number = np.exp(1.0)

    # Leading constant C(φ) = φ · √(φ²−1) / (φ−1)  (from Theorem 5)
    adahedge_leading_constant = (
        phi * np.sqrt(phi**2 - 1.0) / (phi - 1.0)
    )

    # Leading-order term: C(φ) · √(4/(e−1) · L*_T · ln(K))
    leading_order_term = adahedge_leading_constant * np.sqrt(
        (4.0 / (eulers_number - 1.0))
        * cumulative_best_expert_losses
        * np.log(total_number_of_experts)
    )

    # Additive correction (lower-order term from Theorem 5):
    #   (φ/(φ-1)) · (1/(e-1) + 1) · ln(K)
    additive_correction = (
        phi / (phi - 1.0)
        * (1.0 / (eulers_number - 1.0) + 1.0)
        * np.log(total_number_of_experts)
    )

    theoretical_regret_upper_bound = leading_order_term + additive_correction

    return theoretical_regret_upper_bound


def compute_fixed_hedge_worst_case_bound(
    total_number_of_rounds: int,
    total_number_of_experts: int,
) -> np.ndarray:
    """
    Compute the worst-case regret bound for Hedge with η = √(2 ln K / T).

    NOTE: This is an ENVELOPE bound -- it evaluates √(t · ln K / 2) at each
    round t, which corresponds to the bound achieved by the BEST possible
    fixed η for that specific horizon t.  Any single fixed-η Hedge run
    (including η*(T)) may exceed this curve at intermediate rounds because
    its η was tuned for the final T, not for each t.  This is expected and
    does NOT indicate a bug.

    Args:
        total_number_of_rounds  : T.
        total_number_of_experts : K.

    Returns:
        worst_case_bound : shape (T,).
    """
    round_indices = np.arange(1, total_number_of_rounds + 1, dtype=np.float64)
    worst_case_bound = np.sqrt(
        round_indices * np.log(total_number_of_experts) / 2.0
    )
    return worst_case_bound


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 -- PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_ADAHEDGE           = "#F72585"   # Hot magenta    — the hero algorithm
COLOR_BEST_EXPERT        = "#06D6A0"   # Emerald teal   — ground-truth lower bound
COLOR_HEDGE_CONSERVATIVE = "#4CC9F0"   # Icy cyan       — η = 0.1 (too small)
COLOR_HEDGE_MODERATE     = "#FF9F1C"   # Warm amber     — η = 0.5
COLOR_HEDGE_AGGRESSIVE   = "#F77F00"   # Vivid orange   — η = 1.0 (too large)
COLOR_HEDGE_WORST_CASE   = "#4361EE"   # Indigo         — η = η*(T) optimal
COLOR_HEDGE_POSTHOC      = "#9B5DE5"   # Soft violet    — η = η*(L*) post-hoc
COLOR_THEORETICAL_BOUND  = "#666680"   # Muted grey     — dashed bound reference
_ACCENT_AMBER            = "#FFBE0B"   # Amber          — segment boundaries

# ── Per-algorithm style maps (shared by all three plot functions) ─────────────
_ALGO_COLOUR = {
    "AdaHedge (φ=2)":            COLOR_ADAHEDGE,
    "Hedge η=0.1":               COLOR_HEDGE_CONSERVATIVE,
    "Hedge η=0.5":               COLOR_HEDGE_MODERATE,
    "Hedge η=1.0":               COLOR_HEDGE_AGGRESSIVE,
    "Hedge η=η*(T) worst-case":  COLOR_HEDGE_WORST_CASE,
    "Hedge η=η*(L*) post-hoc":   COLOR_HEDGE_POSTHOC,
}
_ALGO_LINESTYLE = {
    "AdaHedge (φ=2)":            "-",
    "Hedge η=0.1":               "-.",
    "Hedge η=0.5":               "-.",
    "Hedge η=1.0":               "-.",
    "Hedge η=η*(T) worst-case":  ":",
    "Hedge η=η*(L*) post-hoc":   ":",
}
_ALGO_LINEWIDTH = {
    "AdaHedge (φ=2)":            2.6,   # hero — noticeably thicker
    "Hedge η=0.1":               1.3,
    "Hedge η=0.5":               1.3,
    "Hedge η=1.0":               1.3,
    "Hedge η=η*(T) worst-case":  1.6,
    "Hedge η=η*(L*) post-hoc":   1.6,
}

# ── Dark-theme colour tokens ──────────────────────────────────────────────────
_BG    = "#0A0A14"   # figure background (very dark navy)
_PANEL = "#0F0F1F"   # axes panel (slightly lighter — adds depth)
_SPINE = "#252540"   # spine / border lines
_GRID  = "#1A1A30"   # grid lines
_LABEL = "#D0D0EE"   # axis / tick labels
_TITLE = "#EEEEFF"   # title text

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _PANEL,
    "axes.edgecolor":    _SPINE,
    "axes.labelcolor":   _LABEL,
    "axes.titlecolor":   _TITLE,
    "text.color":        _LABEL,
    "xtick.color":       _LABEL,
    "ytick.color":       _LABEL,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "grid.color":        _GRID,
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "grid.linewidth":    0.6,
    "legend.facecolor":  _BG,
    "legend.edgecolor":  _SPINE,
    "legend.labelcolor": _LABEL,
    "legend.framealpha": 0.90,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "axes.titlepad":     10,
    "lines.linewidth":   1.8,
    "lines.antialiased": True,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.facecolor": _BG,
})


def _style_axis(ax, title: str, xlabel: str, ylabel: str, round_indices):
    """Apply consistent dark-theme finishing touches to an axis."""
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_xlim(round_indices[0], round_indices[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.6)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.6)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, pad=4)
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)


def save_figure_to_plots_directory(figure_handle, filename_without_extension: str):
    full_output_path = os.path.join(
        PLOTS_OUTPUT_DIRECTORY, filename_without_extension + ".png"
    )
    figure_handle.savefig(full_output_path, bbox_inches="tight")
    plt.close(figure_handle)
    print(f"  [SAVED] {full_output_path}")


def plot_cumulative_losses(
    round_indices: np.ndarray,
    results_by_algorithm_name: dict,
    cumulative_best_expert_losses: np.ndarray,
    regime_display_name: str,
    filename_prefix: str,
) -> None:
    """Plot cumulative loss over time for all algorithms + best expert."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        round_indices, cumulative_best_expert_losses,
        color=COLOR_BEST_EXPERT, linewidth=2.0, linestyle="--",
        label="Best Expert in Hindsight (L*_T)", zorder=5,
    )
    for label, result in results_by_algorithm_name.items():
        is_hero = "AdaHedge" in label
        ax.plot(
            round_indices, result["cumulative_loss_of_learner"],
            color=_ALGO_COLOUR.get(label, "#FFFFFF"),
            linestyle=_ALGO_LINESTYLE.get(label, "-"),
            linewidth=_ALGO_LINEWIDTH.get(label, 1.8),
            alpha=1.0 if is_hero else 0.80,
            zorder=10 if is_hero else 3,
            label=label,
        )

    ax.legend(loc="upper left", fontsize=8, borderpad=0.6,
              labelspacing=0.4, handlelength=2.0)
    _style_axis(ax,
                title=f"Cumulative Losses — {regime_display_name}",
                xlabel="Round  t",
                ylabel=r"Cumulative Loss  $\sum_{s=1}^{t}\,\ell_s$",
                round_indices=round_indices)
    fig.tight_layout()
    save_figure_to_plots_directory(fig, f"{filename_prefix}_cumulative_loss")


def plot_cumulative_regrets(
    round_indices: np.ndarray,
    results_by_algorithm_name: dict,
    total_number_of_experts: int,
    regime_display_name: str,
    filename_prefix: str,
) -> None:
    """Plot cumulative regret for all algorithms with the worst-case bound."""
    fig, ax = plt.subplots(figsize=(8, 5))

    T     = len(round_indices)
    bound = compute_fixed_hedge_worst_case_bound(T, total_number_of_experts)
    ax.plot(
        round_indices, bound,
        color=COLOR_THEORETICAL_BOUND, linestyle="--", linewidth=1.4,
        label=r"Worst-case bound $\sqrt{t\,\ln K\,/\,2}$", zorder=2,
    )
    for label, result in results_by_algorithm_name.items():
        is_hero = "AdaHedge" in label
        ax.plot(
            round_indices, result["cumulative_regret_of_learner"],
            color=_ALGO_COLOUR.get(label, "#FFFFFF"),
            linestyle=_ALGO_LINESTYLE.get(label, "-"),
            linewidth=_ALGO_LINEWIDTH.get(label, 1.8),
            alpha=1.0 if is_hero else 0.80,
            zorder=10 if is_hero else 3,
            label=label,
        )

    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.legend(loc="upper left", fontsize=8, borderpad=0.6,
              labelspacing=0.4, handlelength=2.0)
    _style_axis(ax,
                title=f"Cumulative Regret — {regime_display_name}",
                xlabel="Round  t",
                ylabel=r"Cumulative Regret  $R_t$",
                round_indices=round_indices)
    fig.tight_layout()
    save_figure_to_plots_directory(fig, f"{filename_prefix}_cumulative_regret")


def plot_adahedge_learning_rate_decay(
    round_indices: np.ndarray,
    adahedge_result: dict,
    regime_display_name: str,
    filename_prefix: str,
) -> None:
    """Plot the AdaHedge learning rate η_t with segment-transition markers."""
    eta_traj   = adahedge_result["learning_rate_at_each_round"]
    boundaries = adahedge_result["segment_boundary_round_indices"]
    n_segments = adahedge_result["total_number_of_segments_used"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, bnd in enumerate(boundaries):
        ax.axvline(
            x=bnd + 1, color=_ACCENT_AMBER, linestyle="--",
            alpha=0.50, linewidth=0.9,
            label="Segment reset (Δ ≥ b(η))" if i == 0 else None,
        )

    ax.plot(
        round_indices, eta_traj,
        color=COLOR_ADAHEDGE, linewidth=2.2,
        label=f"η_t  (AdaHedge, φ=2)  — {n_segments} segment(s)",
    )

    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8, borderpad=0.6, labelspacing=0.4)
    _style_axis(ax,
                title=f"AdaHedge Learning Rate Decay — {regime_display_name}",
                xlabel="Round  t",
                ylabel=r"Learning Rate  $\eta_t$  (log scale)",
                round_indices=round_indices)
    fig.tight_layout()
    save_figure_to_plots_directory(fig, f"{filename_prefix}_learning_rate_decay")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 -- SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_simulation_for_regime(
    loss_matrix: np.ndarray,
    regime_display_name: str,
    filename_prefix: str,
    segment_learning_rate_decay_factor: float = 2.0,
) -> None:
    """
    Run a complete simulation experiment for a single loss regime.

    Executes AdaHedge and multiple fixed-η Hedge variants, then generates
    all three required plots (cumulative loss, regret, η-decay).

    Args:
        loss_matrix                        : (T, K) loss matrix for this regime.
        regime_display_name                : Name for titles.
        filename_prefix                    : Filename prefix for saved plots.
        segment_learning_rate_decay_factor : φ for AdaHedge (default 2).
    """
    total_number_of_rounds, total_number_of_experts = loss_matrix.shape
    round_indices = np.arange(1, total_number_of_rounds + 1, dtype=np.float64)

    print("\n" + "-"*60)
    print(f"  Regime : {regime_display_name}")
    print(f"  T = {total_number_of_rounds} rounds,  K = {total_number_of_experts} experts")
    print("-"*60)

    # ── 1. Run AdaHedge ───────────────────────────────────────────────────────
    print("  Running AdaHedge ...")
    adahedge_result = run_adahedge_algorithm(
        loss_matrix=loss_matrix,
        segment_learning_rate_decay_factor=segment_learning_rate_decay_factor,
    )
    print(
        f"    -> Segments used: {adahedge_result['total_number_of_segments_used']}"
    )
    print(
        f"    -> Final regret : {adahedge_result['cumulative_regret_of_learner'][-1]:.4f}"
    )

    # ── 2. Run fixed-eta Hedge variants ──────────────────────────────────────
    #
    # Compute L*_T (total cumulative loss of best expert over all T rounds)
    # needed for the post-hoc optimal eta.
    cumulative_expert_losses_final = np.sum(loss_matrix, axis=0)
    best_expert_total_cumulative_loss = np.min(cumulative_expert_losses_final)

    # Worst-case optimal eta (requires knowing T):
    #   eta* = sqrt(8 * ln(K) / T)
    #
    # Derivation: the Hedge regret bound (via Hoeffding on the cumulant
    # generating function) is:
    #   R_Hedge(T) <= ln(K)/eta + eta*T/8
    # Minimising over eta: d/d_eta [ln(K)/eta + eta*T/8] = 0
    #   => -ln(K)/eta^2 + T/8 = 0  => eta* = sqrt(8*ln(K)/T)
    # At this eta*, the bound equals:
    #   2 * sqrt(T*ln(K)/8) = sqrt(T*ln(K)/2)  <- the reference bound in plots
    worst_case_optimal_learning_rate = np.sqrt(
        8.0 * np.log(total_number_of_experts) / total_number_of_rounds
    )

    # Post-hoc optimal eta (requires knowing L*_T in advance, i.e. "cheating"):
    #   eta* = ln(1 + sqrt(2*ln(K) / L*_T))
    #
    # This is the formula from Cesa-Bianchi & Lugosi (2006) that achieves the
    # regret bound sqrt(2*L*_T*ln(K)) + ln(K).
    #
    # Paper footnote (Section 5): "Cesa-Bianchi and Lugosi use
    # eta=ln(1+sqrt(2*ln(K)/L*_T)), but the same bound can be obtained for
    # the simplified expression eta=sqrt(2*ln(K)/L*_T)."
    # We use the original C&L formula as it is the primary theoretical reference.
    #
    # If L*_T = 0 (pathological, all zero losses), default to eta = 1.0.
    if best_expert_total_cumulative_loss > 1e-10:
        posthoc_optimal_learning_rate = np.log(
            1.0 + np.sqrt(
                2.0 * np.log(total_number_of_experts)
                / best_expert_total_cumulative_loss
            )
        )
    else:
        posthoc_optimal_learning_rate = 1.0

    # Clip post-hoc eta at 1.0 -- the paper restricts eta in (0, 1] for analysis.
    posthoc_optimal_learning_rate = min(posthoc_optimal_learning_rate, 1.0)

    fixed_learning_rate_values = {
        "Hedge η=0.1":              0.1,
        "Hedge η=0.5":              0.5,
        "Hedge η=1.0":              1.0,
        "Hedge η=η*(T) worst-case": worst_case_optimal_learning_rate,
        "Hedge η=η*(L*) post-hoc":  posthoc_optimal_learning_rate,
    }

    all_results = {"AdaHedge (φ=2)": adahedge_result}

    for algorithm_label, fixed_eta in fixed_learning_rate_values.items():
        print(f"  Running {algorithm_label.replace('η', 'eta')} (eta = {fixed_eta:.5f}) ...")
        hedge_result = run_hedge_fixed_learning_rate(
            loss_matrix=loss_matrix,
            learning_rate=fixed_eta,
        )
        all_results[algorithm_label] = hedge_result
        print(
            f"    -> Final regret : {hedge_result['cumulative_regret_of_learner'][-1]:.4f}"
        )

    # ── 3. Extract best-expert cumulative losses (from AdaHedge result) ───────
    cumulative_best_expert_losses = adahedge_result["cumulative_loss_of_best_expert"]

    # ── 4. Generate plots ──────────────────────────────────────────────────────
    print("  Generating plots ...")

    plot_cumulative_losses(
        round_indices=round_indices,
        results_by_algorithm_name=all_results,
        cumulative_best_expert_losses=cumulative_best_expert_losses,
        regime_display_name=regime_display_name,
        filename_prefix=filename_prefix,
    )

    plot_cumulative_regrets(
        round_indices=round_indices,
        results_by_algorithm_name=all_results,
        total_number_of_experts=total_number_of_experts,
        regime_display_name=regime_display_name,
        filename_prefix=filename_prefix,
    )

    plot_adahedge_learning_rate_decay(
        round_indices=round_indices,
        adahedge_result=adahedge_result,
        regime_display_name=regime_display_name,
        filename_prefix=filename_prefix,
    )

    print(f"  All good, Done with '{regime_display_name}'")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 -- MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main function.  Generates loss matrices for all three regimes and runs
    the full simulation + plotting pipeline for each.

    Configuration:
        T = 10 000 rounds   (consistent with paper experiments)
        K = 4 experts       (stochastic and low-gap regimes)
        K = 2 experts       (adversarial regime -- swap trick requires 2)
        φ = 2               (as in paper experiments)
    """
    total_number_of_rounds       = 10_000
    number_of_experts_standard   = 4      # For stochastic and low-gap regimes
    phi                          = 2.0   # Segment decay factor

    # ── 1. STOCHASTIC REGIME ─────────────────────────────────────────────────
    print("\n[REGIME 1] Stochastic -- Bernoulli(0.3) vs Bernoulli(0.5)")
    stochastic_loss_matrix = generate_stochastic_regime_losses(
        total_number_of_rounds=total_number_of_rounds,
        total_number_of_experts=number_of_experts_standard,
        rng=numpy_random_generator,
    )
    run_full_simulation_for_regime(
        loss_matrix=stochastic_loss_matrix,
        regime_display_name=(
            "Stochastic  "
            "[Expert 0: Bernoulli(0.3),  Others: Bernoulli(0.5)]"
        ),
        filename_prefix="stochastic",
        segment_learning_rate_decay_factor=phi,
    )

    # ── 2. ADVERSARIAL REGIME ─────────────────────────────────────────────────
    print("\n[REGIME 2] Adversarial -- best expert swaps at t = T/2")
    adversarial_loss_matrix = generate_adversarial_regime_losses(
        total_number_of_rounds=total_number_of_rounds,
        rng=numpy_random_generator,
    )
    run_full_simulation_for_regime(
        loss_matrix=adversarial_loss_matrix,
        regime_display_name=(
            "Adversarial  "
            "[Best expert swaps halfway:  Expert 0->1 at t=T/2]"
        ),
        filename_prefix="adversarial",
        segment_learning_rate_decay_factor=phi,
    )

    # ── 3. LOW-GAP STOCHASTIC REGIME ──────────────────────────────────────────
    print("\n[REGIME 3] Low-Gap Stochastic -- Bernoulli(0.49) vs Bernoulli(0.50)")
    low_gap_loss_matrix = generate_low_gap_stochastic_losses(
        total_number_of_rounds=total_number_of_rounds,
        total_number_of_experts=number_of_experts_standard,
        rng=numpy_random_generator,
    )
    run_full_simulation_for_regime(
        loss_matrix=low_gap_loss_matrix,
        regime_display_name=(
            "Low-Gap Stochastic  "
            "[Expert 0: Bernoulli(0.49),  Others: Bernoulli(0.50)]"
        ),
        filename_prefix="low_gap",
        segment_learning_rate_decay_factor=phi,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  Phase 1 simulation COMPLETE.")
    print(f"  All plots saved to: {PLOTS_OUTPUT_DIRECTORY}")
    print("="*72)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
