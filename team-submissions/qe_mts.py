

import random
import copy

def calculate_energy(s: list[int], N: int):
    '''
    Calculates the energy (E) of a bitstring s for the LABS problem.
    s: List of integers, e.g., [0, 0, 1, 0]
    Mapping: 0 -> -1, 1 -> 1
    '''
    # Safety check: if list is shorter than N, pad with 0s (leading zeros)
    # This fixes the issue if an incomplete list is ever passed
    if len(s) < N:
        s = [0] * (N - len(s)) + s

    # Convert 0/1 to -1/+1
    # 0 becomes -1, 1 becomes 1
    sequence = [2*x - 1 for x in s]

    energy = 0
    # C_k calculation (Autocorrelation)
    for k in range(1, N):
        c_k = 0
        for i in range(N - k):
            c_k += sequence[i] * sequence[i + k]
        energy += c_k**2

    return energy

def combine(s1: list[int], s2: list[int], N: int):
    '''Combines two bitstrings (lists) of the same length via single-point crossover'''
    # Ensure cut_pt allows for at least one bit from each
    if N > 1:
        cut_pt = random.randint(1, N - 1)
        return s1[0:cut_pt] + s2[cut_pt:]
    else:
        return s1 # Fallback for N=1

def mutate(s: list[int], p: float, N: int):
    '''Mutates N-length bitstring IN PLACE'''
    for i in range(N):
        if random.random() < p:
            # XOR with 1 flips the bit (0->1, 1->0)
            s[i] = s[i] ^ 1

def tabu_search(N: int, s: list[int], max_iters=100, tabu_tenure=5):
    '''
    Performs a greedy local search with a Tabu list mechanism.
    '''
    current_s = copy.deepcopy(s)
    best_s = copy.deepcopy(s)
    best_energy = calculate_energy(current_s, N)

    # Tabu list stores INDICES of bits that were recently flipped
    tabu_list = []

    for _ in range(max_iters):
        local_best_neighbor = None
        local_best_energy = float('inf')
        move_index = -1

        # 1. Evaluate all 1-bit flip neighbors
        for i in range(N):
            # Create neighbor efficiently
            # We modify current_s temporarily to check energy, then flip back
            # This is faster than deepcopying N times
            current_s[i] ^= 1 # Flip
            neighbor_energy = calculate_energy(current_s, N)

            # 2. Check Tabu conditions
            is_tabu = i in tabu_list
            is_aspiration = neighbor_energy < best_energy

            if (not is_tabu) or is_aspiration:
                if neighbor_energy < local_best_energy:
                    local_best_energy = neighbor_energy
                    move_index = i
                    # We must copy here to save the configuration
                    local_best_neighbor = copy.deepcopy(current_s)

            current_s[i] ^= 1 # Flip back (Revert)

        # 3. Make the move if a valid neighbor was found
        if local_best_neighbor is not None:
            current_s = local_best_neighbor

            # Update Global Best
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_s = copy.deepcopy(current_s)

            # Update Tabu List
            tabu_list.append(move_index)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    return best_s

def generate_random_bitstring(k, N):
    '''Generates k many N-length bitstrings as LISTS of INTs'''
    population = []
    for _ in range(k):
        # [0, 1, 0, 0, ...]
        bitstring = [random.randint(0, 1) for _ in range(N)]
        population.append(bitstring)
    return population

def MTS(N, k, num_iterations, mutation_rate=0.05, population=[]):
    '''
    Performs MTS algorithm to solve LABS.
    Expects population to be a list of lists: [[0,1,0], [1,1,0], ...]
    '''
    # Initialize if empty
    if not population:
        population = generate_random_bitstring(k, N)
    elif len(population) < k: 
        # Pad population if too small
        additional = generate_random_bitstring(k - len(population), N)
        population.extend(additional)

    # Initialize best tracker
    best_solution = population[0]
    best_energy = calculate_energy(best_solution, N)

    # Find initial best
    for ind in population:
        e = calculate_energy(ind, N)
        if e < best_energy:
            best_energy = e
            best_solution = copy.deepcopy(ind)

    print(f"Initial Best Energy: {best_energy}")

    for it in range(num_iterations):
        ## Step 2: Make child
        child = []
        if it == 0:
            child = copy.deepcopy(random.choice(population))
        else:
            p1 = random.choice(population)
            p2 = random.choice(population)
            child = combine(p1, p2, N)

        ## Step 3: Mutate
        mutate(child, mutation_rate, N)

        ## Step 4: Tabu Search
        improved_child = tabu_search(N, child, max_iters=N*2)
        improved_child_energy = calculate_energy(improved_child, N)

        ## Step 5: Update Population
        if improved_child_energy < best_energy:
            print(f"Iteration {it}: New Best Energy Found: {improved_child_energy}")

            best_energy = improved_child_energy
            best_solution = copy.deepcopy(improved_child)

            replace_idx = random.randint(0, k - 1)
            population[replace_idx] = improved_child

    return best_solution, best_energy





import cudaq

cudaq.set_target("tensornet-mps", option="fp32")  # GPU statevector

#print("Target: ", cudaq.get_target().name)
import numpy as np

# Helper kernel for RZZ gate (not native in CUDA-Q)
@cudaq.kernel
def rzz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """RZZ(theta) = CNOT · RZ(theta) · CNOT"""
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)


@cudaq.kernel
def block1_ryz_rzy(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    Block 1: R_YZ(theta) * R_ZY(theta)
    2 RZZ entangling gates + 4 single-qubit gates
    """
    # R_YZ(theta): Y on q0, Z on q1
    rx(-np.pi/2, q0)
    rzz(q0, q1, theta)
    rx(np.pi/2, q0)

    # R_ZY(theta): Z on q0, Y on q1
    rx(-np.pi/2, q1)
    rzz(q0, q1, theta)
    rx(np.pi/2, q1)


@cudaq.kernel
def block2_4qubit_rotations(q0: cudaq.qubit, q1: cudaq.qubit,
                             q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """
    Block 2: R_YZZZ * R_ZYZZ * R_ZZYZ * R_ZZZY
    10 RZZ entangling gates + 28 single-qubit gates
    """
    # R_YZZZ(theta): Y on q0, Z on q1,q2,q3
    rx(-np.pi/2, q0)
    rzz(q0, q1, theta)
    rzz(q1, q2, -theta)
    rzz(q2, q3, theta)
    rzz(q1, q2, -theta)
    rzz(q0, q1, theta)
    rx(np.pi/2, q0)
    rz(-theta, q3)

    # R_ZYZZ(theta): Z on q0, Y on q1, Z on q2,q3
    rx(-np.pi/2, q1)
    rzz(q0, q1, theta)
    rx(np.pi/2, q1)
    rx(-np.pi/2, q1)
    rzz(q1, q2, -theta)
    rzz(q2, q3, theta)
    rzz(q1, q2, -theta)
    rx(np.pi/2, q1)
    rz(-theta, q3)

    # R_ZZYZ(theta): Z on q0,q1, Y on q2, Z on q3
    rx(-np.pi/2, q2)
    rzz(q1, q2, theta)
    rzz(q2, q3, -theta)
    rzz(q1, q2, theta)
    rx(np.pi/2, q2)
    rz(-theta, q3)

    # R_ZZZY(theta): Z on q0,q1,q2, Y on q3
    rx(-np.pi/2, q3)
    rzz(q2, q3, theta)
    rx(np.pi/2, q3)
    rz(-theta, q3)


@cudaq.kernel
def dcqo_circuit(theta: float):
    """Main circuit combining both blocks"""
    q = cudaq.qvector(4)

    # Apply Block 1 on qubits 0,1
    block1_ryz_rzy(q[0], q[1], theta)

    # Apply Block 2 on all 4 qubits
    block2_4qubit_rotations(q[0], q[1], q[2], q[3], theta)

    mz(q)



def get_interactions(N):
    """
    Generates G2 and G4 interaction lists based on Eq. 15 loop limits.
    Returns standard 0-based indices.
    """
    G2 = []
    G4 = []

    # --- G2: Two-Body Interactions ---
    # Eq 15: i from 1 to N-2
    for i in range(N - 2):
        # Eq 15: k from 1 to floor((N-i)/2)
        # Note: (i+1) converts Python 0-index back to 1-based math for the limit calc
        limit_k = (N - (i + 1)) // 2

        for k in range(1, limit_k + 1):
            G2.append([i, i + k])

    # --- G4: Four-Body Interactions ---
    # Eq 15: i from 1 to N-3
    for i in range(N - 3):
        # Eq 15: t from 1 to floor((N-i-1)/2)
        limit_t = (N - (i + 1) - 1) // 2

        for t in range(1, limit_t + 1):
            # Eq 15: k from t+1 to N-i-t
            limit_k = N - (i + 1) - t

            for k in range(t + 1, limit_k + 1):
                G4.append([i, i + t, i + k, i + k + t])

    return G2, G4


# @cudaq.kernel
# def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], steps: int, dt: float, T: float, thetas: list[float]):

#     reg = cudaq.qvector(N)
#     h(reg) ## this (|+>^{\otimes N}) is the groundstate of H_i

#     # TODO - write the full kernel to apply the trotterized circuit




# T=1               # total time
# n_steps = 20      # number of trotter steps
# dt = T / n_steps
# N = 20
# G2, G4 = get_interactions(N)

# thetas =[]

# for step in range(1, n_steps + 1):
#     t = step * dt
#     theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
#     thetas.append(theta_val)

# # TODO - Sample your kernel to make sure it works

@cudaq.kernel
def trotterized_circuit(N: int, n_steps: int,
                        # Flattened interaction lists (required for CUDA-Q kernels)
                        g2_flat: list[int], num_g2: int,
                        g4_flat: list[int], num_g4: int,
                        thetas: list[float]):

    # 1. Initialize Register
    reg = cudaq.qvector(N)

    # 2. State Preparation: |+>^N
    h(reg)

    # Hamiltonian parameter (h^x_i = -1)
    h_x = -1.0

    # 3. Trotter Evolution Loop
    for step in range(n_steps):
        # FIX: Rename variable to avoid "captured from parent scope" error
        theta_step = thetas[step]

        # --- Block 1: Two-Body Terms ---
        for p in range(num_g2):
            # Stride 2 indexing
            idx_i = g2_flat[2 * p]
            idx_j = g2_flat[2 * p + 1]

            # Calculate angle: 4 * theta(t) * h_x
            angle_2body = 4.0 * theta_step * h_x

            # Call your provided helper
            block1_ryz_rzy(reg[idx_i], reg[idx_j], angle_2body)

        # --- Block 2: Four-Body Terms ---
        for q in range(num_g4):
            # Stride 4 indexing
            q_i   = g4_flat[4 * q]
            q_it  = g4_flat[4 * q + 1]
            q_ik  = g4_flat[4 * q + 2]
            q_ikt = g4_flat[4 * q + 3]

            # Calculate angle: 8 * theta(t) * h_x
            angle_4body = 8.0 * theta_step * h_x

            # Call your provided helper
            block2_4qubit_rotations(reg[q_i], reg[q_it], reg[q_ik], reg[q_ikt], angle_4body)

    # 4. Measurement
    mz(reg)

# TODO - write code here to sample from your CUDA-Q kernel and used the results to seed your MTS population

import time
import random
import copy
import numpy as np

# ============================================================
# A) Build LABS interactions in the format your trotter circuit expects
#    - G2: list[(i,j)]
#    - G4: list[(i,j,k,l)]
# ============================================================

def labs_G2_G4(N: int):
    """
    Returns:
      G4: list of 4-tuples (i, i+tt, i+kk, i+kk+tt)  (your 4-body ZZZZ terms)
      G2: list of 2-tuples (i, i+2k)                 (your 2-body ZZ terms)
    """
    G4 = []
    for ii in range(N - 3):
        for tt in range(1, (N - ii - 1) // 2 + 1):
            for kk in range(tt + 1, N - ii - tt + 1):
                if ii + kk + tt < N:
                    G4.append((ii, ii + tt, ii + kk, ii + kk + tt))

    G2 = []
    for ii in range(N - 2):
        for kk in range(1, (N - ii) // 2 + 1):
            jj = ii + 2 * kk
            if jj < N:
                G2.append((ii, jj))

    return G2, G4


def flatten_G2_G4(G2, G4):
    """
    Flatten tuple lists into the exact style you used:
      g2_flat = [i0, j0, i1, j1, ...]
      g4_flat = [a0, b0, c0, d0, a1, b1, c1, d1, ...]
    """
    g2_flat = []
    for (i, j) in G2:
        g2_flat.extend([int(i), int(j)])

    g4_flat = []
    for (a, b, c, d) in G4:
        g4_flat.extend([int(a), int(b), int(c), int(d)])

    return g2_flat, g4_flat


# ============================================================
# B) Theta generation (you choose a layout that matches your kernel)
# ============================================================

def generate_thetas(
    n_steps: int,
    theta2_max: float = 0.4,
    theta4_max: float = 0.2,
    schedule: str = "linear",
    layout: str = "interleaved",
):
    """
    Produces thetas used by trotterized_circuit.

    schedule:
      - "linear": theta ramps linearly from small -> max
      - "constant": same theta each step

    layout (MUST match your trotterized_circuit):
      - "interleaved": [theta2_0, theta4_0, theta2_1, theta4_1, ...]   length = 2*n_steps
      - "split":       [theta2_0..theta2_{S-1}, theta4_0..theta4_{S-1}] length = 2*n_steps
      - "pair":        [theta2, theta4] (constant, length = 2)          (only for constant schedule)
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")

    if schedule == "linear":
        ts = [(s + 1) / n_steps for s in range(n_steps)]
        theta2 = [theta2_max * t for t in ts]
        theta4 = [theta4_max * t for t in ts]
    elif schedule == "constant":
        theta2 = [theta2_max] * n_steps
        theta4 = [theta4_max] * n_steps
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    if layout == "interleaved":
        thetas = []
        for s in range(n_steps):
            thetas.append(float(theta2[s]))
            thetas.append(float(theta4[s]))
        return thetas

    if layout == "split":
        return [float(x) for x in theta2] + [float(x) for x in theta4]

    if layout == "pair":
        if schedule != "constant":
            raise ValueError("layout='pair' only makes sense with schedule='constant'")
        return [float(theta2_max), float(theta4_max)]

    raise ValueError(f"Unknown layout: {layout}")


# ============================================================
# C) Convert cudaq counts -> population (optionally symmetry dedup)
# ============================================================

def canonicalize_bitstring(bits01: list[int]) -> list[int]:
    """
    LABS symmetries:
      1) bit-flip: s and NOT(s) have same energy
      2) reversal: s and reverse(s) have same energy
    Return lexicographically smallest among {s, NOT(s), REV(s), NOT(REV(s))}
    """
    s = np.array(bits01, dtype=np.int8)
    candidates = [
        s,
        1 - s,
        s[::-1],
        1 - s[::-1],
    ]
    best = min(candidates, key=lambda x: tuple(x.tolist()))
    return best.tolist()


def results_to_population(counts, k_pop: int, N: int, use_symmetry: bool = True) -> list[list[int]]:
    """
    counts: cudaq.sample result (behaves like dict: bitstring -> count)
    """
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])

    pop = []
    seen = set()

    for bitstring, _c in sorted_items:
        bits = [int(b) for b in bitstring.zfill(N)]
        if use_symmetry:
            canon = tuple(canonicalize_bitstring(bits))
            if canon in seen:
                continue
            seen.add(canon)
            pop.append(list(canon))
        else:
            pop.append(bits)

        if len(pop) >= k_pop:
            break

    while len(pop) < k_pop:
        pop.append(copy.deepcopy(random.choice(pop)))

    return pop


# ============================================================
# D) The requested "one function" QE–MTS solve wrapper
# ============================================================

def qe_mts_solve(
    N: int,
    *,
    k: int = 10,
    n_steps: int = 3,
    shots: int = 4096,
    theta2_max: float = 0.4,
    theta4_max: float = 0.2,
    theta_schedule: str = "linear",
    theta_layout: str = "interleaved",
    use_symmetry_population: bool = True,
    mts_iterations: int = 100,
    mutation_rate: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Requires these to exist in your environment:
      - cudaq
      - trotterized_circuit kernel
      - your classical MTS(...) function (same signature you posted)

    Returns a dict with:
      best_solution, best_energy, merit_factor,
      timings (quantum, mts, total),
      initial_population, counts,
      thetas, |G2|, |G4|
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1) Build couplings
    G2, G4 = labs_G2_G4(N)
    g2_flat, g4_flat = flatten_G2_G4(G2, G4)

    # 2) Generate trotter angles
    thetas = generate_thetas(
        n_steps=n_steps,
        theta2_max=theta2_max,
        theta4_max=theta4_max,
        schedule=theta_schedule,
        layout=theta_layout,
    )

    if verbose:
        print(f"[QE–MTS] N={N} | k={k} | n_steps={n_steps} | shots={shots}")
        print(f"  |G2|={len(G2)}  |G4|={len(G4)}")
        print(f"  thetas(len={len(thetas)}): {thetas[:min(8,len(thetas))]}{'...' if len(thetas)>8 else ''}")

    # 3) Quantum sampling
    t0 = time.perf_counter()
    counts = cudaq.sample(
        trotterized_circuit,
        N,
        n_steps,
        g2_flat, len(G2),
        g4_flat, len(G4),
        thetas,
        shots_count=shots
    )
    # 4) Convert to population
    initial_population = results_to_population(
        counts, k_pop=k, N=N, use_symmetry=use_symmetry_population
    )
    t1 = time.perf_counter()

    # 5) Run MTS
    best_solution, best_energy = MTS(
        N=N,
        k=k,
        num_iterations=mts_iterations,
        mutation_rate=mutation_rate,
        population=initial_population
    )
    t2 = time.perf_counter()

    best_energy = int(best_energy)
    merit = (N**2 / (2 * best_energy)) if best_energy > 0 else float("inf")

    out = {
        "N": N,
        "best_solution": best_solution,
        "best_energy": best_energy,
        "merit_factor": merit,
        "quantum_time_s": (t1 - t0),
        "mts_time_s": (t2 - t1),
        "total_time_s": (t2 - t0),
        "counts": counts,
        "initial_population": initial_population,
        "thetas": thetas,
        "n_steps": n_steps,
        "shots": shots,
        "G2_size": len(G2),
        "G4_size": len(G4),
        "theta_schedule": theta_schedule,
        "theta_layout": theta_layout,
    }

    if verbose:
        print(f"  quantum_time={out['quantum_time_s']:.3f}s | mts_time={out['mts_time_s']:.3f}s | total={out['total_time_s']:.3f}s")
        print(f"  best_energy={best_energy} | merit={merit:.4f}")

    return out


# # ============================================================
# # Example usage:
# # ============================================================
# result = qe_mts_solve(
#     30,
#     k=10,
#     n_steps=3,
#     shots=4096,
#     theta2_max=0.4,
#     theta4_max=0.2,
#     theta_schedule="linear",
#     theta_layout="interleaved",   # <-- if this errors / behaves wrong, try "split" or "pair"
#     use_symmetry_population=True,
#     mts_iterations=100,
#     mutation_rate=0.05,
#     seed=0,
#     verbose=True,
# )



"""
Scaling plots for QE–MTS using your NEW qe_mts_solve(N, ...) wrapper
(which dynamically generates G2/G4 + thetas internally).

Produces:
(a) total runtime (QE + MTS) vs N   (cap N<=31)
(b) MTS iterations used vs N

IMPORTANT:
- Your qe_mts_solve currently calls MTS(...) which runs a fixed number of iterations.
  So (b) will be a flat line unless we add an early-stop MTS variant.
- Below I give you BOTH options:
    1) Fixed-iteration plots (b will be constant)
    2) Early-stop plots (b becomes meaningful)
- This code assumes qe_mts_solve exists in your environment already.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import io

# ============================================================
# Optional: Early-stop MTS wrapper (uses your existing primitives)
# ============================================================
def MTS_with_stats(
    N: int,
    k: int,
    num_iterations: int,
    mutation_rate: float,
    population: list[list[int]],
    patience: int | None = 10,    # None disables early stop
    tabu_tenure: int = 5
):
    """
    Same structure as your classical MTS, but returns:
      best_solution, best_energy, iters_used, n_improvements

    Requires: calculate_energy, combine, mutate, tabu_search, generate_random_bitstring
    """
    if population is None or len(population) == 0:
        population = generate_random_bitstring(k, N)
    elif len(population) < k:
        population.extend(generate_random_bitstring(k - len(population), N))

    best_solution = population[0]
    best_energy = calculate_energy(best_solution, N)
    for ind in population:
        e = calculate_energy(ind, N)
        if e < best_energy:
            best_energy = e
            best_solution = copy.deepcopy(ind)

    no_improve = 0
    n_improvements = 0

    for it in range(num_iterations):
        if it == 0:
            child = copy.deepcopy(random.choice(population))
        else:
            p1 = random.choice(population)
            p2 = random.choice(population)
            child = combine(p1, p2, N)

        mutate(child, mutation_rate, N)
        improved_child = tabu_search(N, child, max_iters=N * 2, tabu_tenure=tabu_tenure)
        improved_E = calculate_energy(improved_child, N)

        if improved_E < best_energy:
            best_energy = improved_E
            best_solution = copy.deepcopy(improved_child)
            population[random.randint(0, k - 1)] = improved_child
            n_improvements += 1
            no_improve = 0
        else:
            no_improve += 1

        if patience is not None and no_improve >= patience:
            return best_solution, best_energy, (it + 1), n_improvements

    return best_solution, best_energy, num_iterations, n_improvements


# ============================================================
# A patched QE–MTS runner that reuses qe_mts_solve but optionally
# replaces the MTS call with early-stop MTS_with_stats.
# ============================================================
def qe_mts_solve_with_stats(
    N: int,
    *,
    k: int = 10,
    n_steps: int = 3,
    shots: int = 4096,
    theta2_max: float = 0.4,
    theta4_max: float = 0.2,
    theta_schedule: str = "linear",
    theta_layout: str = "interleaved",
    use_symmetry_population: bool = True,
    mts_iterations: int = 100,
    mutation_rate: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
    # new controls:
    use_early_stop: bool = True,
    patience: int | None = 10,
    silence_mts_prints: bool = True,
):
    """
    Calls qe_mts_solve up through quantum sampling + population generation,
    then runs either:
      - early-stop MTS_with_stats (recommended for iterations-used plot), or
      - original fixed-iteration MTS (iters_used = mts_iterations).
    """
    # --- Run qe_mts_solve but stop it from printing a ton (optional) ---
    if silence_mts_prints:
        sink = io.StringIO()
        ctx = contextlib.redirect_stdout(sink)
    else:
        ctx = contextlib.nullcontext()

    # We want qe_mts_solve's dynamic G2/G4 + theta generation + sampling + population.
    # But qe_mts_solve currently also runs MTS inside it.
    #
    # To avoid duplicating your qe_mts_solve code, we just call it with mts_iterations=0
    # IF your MTS can handle 0. If it can't, set mts_iterations=1 and ignore result.
    #
    # Safer approach: call qe_mts_solve normally and just use its returned initial_population,
    # but that wastes time doing MTS twice. We'll avoid that by directly doing:
    #   - rebuild G2/G4 + thetas + sample + results_to_population
    #
    # So: we reimplement the "quantum half" here using the same helpers qe_mts_solve used.
    # This assumes you have labs_G2_G4, flatten_G2_G4, generate_thetas, results_to_population in scope.
    random.seed(seed)
    np.random.seed(seed)

    # --- Quantum half (same as inside qe_mts_solve) ---
    G2, G4 = labs_G2_G4(N)
    g2_flat, g4_flat = flatten_G2_G4(G2, G4)
    thetas = generate_thetas(
        n_steps=n_steps,
        theta2_max=theta2_max,
        theta4_max=theta4_max,
        schedule=theta_schedule,
        layout=theta_layout,
    )

    t0 = time.perf_counter()
    counts = cudaq.sample(
        trotterized_circuit,
        N,
        n_steps,
        g2_flat, len(G2),
        g4_flat, len(G4),
        thetas,
        shots_count=shots
    )
    initial_population = results_to_population(
        counts, k_pop=k, N=N, use_symmetry=use_symmetry_population
    )
    t1 = time.perf_counter()

    # --- Classical half (MTS) ---
    with ctx:
        if use_early_stop:
            best_sol, best_E, iters_used, n_impr = MTS_with_stats(
                N=N,
                k=k,
                num_iterations=mts_iterations,
                mutation_rate=mutation_rate,
                population=copy.deepcopy(initial_population),
                patience=patience
            )
        else:
            best_sol, best_E = MTS(
                N=N,
                k=k,
                num_iterations=mts_iterations,
                mutation_rate=mutation_rate,
                population=copy.deepcopy(initial_population)
            )
            iters_used = mts_iterations
            n_impr = np.nan

    t2 = time.perf_counter()

    best_E = int(best_E)
    merit = (N**2 / (2 * best_E)) if best_E > 0 else np.inf

    return {
        "N": N,
        "best_solution": best_sol,
        "best_energy": best_E,
        "merit_factor": float(merit),
        "quantum_time_s": (t1 - t0),
        "mts_time_s": (t2 - t1),
        "total_time_s": (t2 - t0),
        "mts_iters_used": int(iters_used),
        "mts_improvements": n_impr,
        "G2_size": len(G2),
        "G4_size": len(G4),
        "thetas_len": len(thetas),
    }


# ============================================================
# Benchmark sweep + aggregation
# ============================================================
def sweep_qe_mts_scaling(
    N_min: int = 2,
    N_max: int = 11,     # cap at 31
    repeats: int = 1,
    *,
    k: int = 10,
    n_steps: int = 3,
    shots: int = 4096,
    theta2_max: float = 0.4,
    theta4_max: float = 0.2,
    theta_schedule: str = "linear",
    theta_layout: str = "interleaved",
    use_symmetry_population: bool = True,
    mts_iterations: int = 100,
    mutation_rate: float = 0.05,
    use_early_stop: bool = True,
    patience: int | None = 10,
    base_seed: int = 0,
):
    rows = []
    for N in range(N_min, N_max + 1):
        for r in range(repeats):
            seed = base_seed + 1000 * r + N
            print(f"\n=== QE–MTS scaling: N={N} (repeat {r+1}/{repeats}) ===")
            try:
                row = qe_mts_solve_with_stats(
                    N,
                    k=k,
                    n_steps=n_steps,
                    shots=shots,
                    theta2_max=theta2_max,
                    theta4_max=theta4_max,
                    theta_schedule=theta_schedule,
                    theta_layout=theta_layout,
                    use_symmetry_population=use_symmetry_population,
                    mts_iterations=mts_iterations,
                    mutation_rate=mutation_rate,
                    seed=seed,
                    verbose=False,
                    use_early_stop=use_early_stop,
                    patience=patience,
                    silence_mts_prints=True
                )
                print(f"  total={row['total_time_s']:.2f}s (qe={row['quantum_time_s']:.2f}s, mts={row['mts_time_s']:.2f}s)"
                      f"  iters_used={row['mts_iters_used']}  bestE={row['best_energy']}  |G2|={row['G2_size']} |G4|={row['G4_size']}")
                rows.append(row)
            except Exception as e:
                print(f"[STOP] Failed at N={N}: {repr(e)}")
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def aggregate_for_plot(df: pd.DataFrame):
    g = df.groupby("N", as_index=False).agg(
        total_mean=("total_time_s", "mean"),
        total_std=("total_time_s", "std"),
        qe_mean=("quantum_time_s", "mean"),
        qe_std=("quantum_time_s", "std"),
        mts_mean=("mts_time_s", "mean"),
        mts_std=("mts_time_s", "std"),
        iters_mean=("mts_iters_used", "mean"),
        iters_std=("mts_iters_used", "std"),
        bestE_mean=("best_energy", "mean"),
        G2_mean=("G2_size", "mean"),
        G4_mean=("G4_size", "mean"),
    )
    for c in ["total_std", "qe_std", "mts_std", "iters_std"]:
        g[c] = g[c].fillna(0.0)
    return g


# ============================================================
# Plotting
# ============================================================
def plot_qe_mts_scaling(agg: pd.DataFrame, save_prefix: str | None = None):
    N = agg["N"].to_numpy()

    # (a) Total QE+MTS runtime vs N
    plt.figure()
    plt.errorbar(N, agg["total_mean"], yerr=agg["total_std"], capsize=3)
    plt.xlabel("Bitstring length N")
    plt.ylabel("Total runtime (QE + MTS) [s]")
    plt.title("QE–MTS total runtime vs N (dynamic G2/G4)")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_total_runtime_vs_N.png", dpi=200, bbox_inches="tight")

    # Optional runtime breakdown
    plt.figure()
    plt.errorbar(N, agg["qe_mean"], yerr=agg["qe_std"], capsize=3, label="Quantum seeding (sample + decode)")
    plt.errorbar(N, agg["mts_mean"], yerr=agg["mts_std"], capsize=3, label="MTS")
    plt.xlabel("Bitstring length N")
    plt.ylabel("Runtime [s]")
    plt.title("QE–MTS runtime breakdown vs N")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_runtime_breakdown_vs_N.png", dpi=200, bbox_inches="tight")

    # (b) MTS iterations used vs N
    plt.figure()
    plt.errorbar(N, agg["iters_mean"], yerr=agg["iters_std"], capsize=3)
    plt.xlabel("Bitstring length N")
    plt.ylabel("MTS iterations used")
    plt.title("QE–MTS: MTS iterations used vs N")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_mts_iters_used_vs_N.png", dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# Example usage (cap at N<=31)
# ============================================================
if __name__ == "__main__":
    df = sweep_qe_mts_scaling(
        N_min=29,
        N_max=35,
        repeats=1,                # set to 3+ for error bars
        k=10,
        n_steps=3,
        shots=4096,
        theta2_max=0.4,
        theta4_max=0.2,
        theta_schedule="linear",
        theta_layout="interleaved",
        use_symmetry_population=True,
        mts_iterations=100,
        mutation_rate=0.05,
        use_early_stop=True,      # set False => iterations plot becomes flat
        patience=10,
        base_seed=0
    )

    agg = aggregate_for_plot(df)
    plot_qe_mts_scaling(agg, save_prefix="qe_mts_CPU")

    df.to_csv("qe_mts_dynamicG_raw.csv", index=False)
    agg.to_csv("qe_mts_dynamicG_agg.csv", index=False)
    print("\nSaved: qe_mts_dynamicG_raw.csv and qe_mts_dynamicG_agg.csv")

