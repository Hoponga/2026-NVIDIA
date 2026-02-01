## The Workflow : How did you organize your AI agents? (e.g., "We used a Cursor agent for coding and a separate ChatGPT instance for documentation").
We used chatGPT for understanding and documentation. First used ChatGPT to sanity-check our understanding of the algorithms, then we written down in plain English and writing prompts. Once we are comfortable with the idea, we used coda as our coding assistant. After that we will first run the test, if it passed, we will run our verifaction code. 

## Verification Strategy : How did you validate code created by AI?
We did not trust AI-generated code by default. Every AI-assisted change had to pass **(1) unit tests**, **(2) physical/symmetry invariants**, and **(3) known optimal energy solution** before being merged. The test.py should be able to find here: .

### 1) Unit tests (deterministic + fast)
We wrote unit tests to lock down correctness of core components:
- Objective/energy function returns expected values on hand-constructed inputs
- Shape/type checks and bounds checks for inputs (N, graph edges, parameter vectors)
- Reproducibility: fixed random seeds and verified stable outputs where applicable

These tests run locally in seconds and catch basic logic bugs introduced by AI edits and protect against subtle sign mistakes, wrong Hamiltonian terms, or incorrect aggregation logic.


### 2) Physics/symmetry invariants (domain-specific correctness)
We encoded problem invariants that must always hold, and we assert them directly:
- **Global flip symmetry:** `E(x) == E(-x)` (for Ising-style objectives / Hamiltonians)
- **Permutation / relabeling consistency:** relabeling vertices does not change the energy distribution (when applicable)
- **Known edge-case states:** energies for all-zeros / all-ones / alternating patterns match expected values

If any invariant fails, it’s treated as a correctness failure

### 3) Best energy compares with optimal energy
We verified that our answers were never below the pre-existing optimal Energy values for any given N we tested, by referring to Mertens’ Exhaustive Table (1996).

### 4) AI code review rules (process guardrails)
To reduce hallucination risk, we applied strict review rules:
- AI outputs must be small, localized diffs (no giant rewrites)
- Each change must be traceable to a test or invariant it satisfies
- Performance changes are only accepted after correctness passes 

## The "Vibe" Log :
### Win: One instance where AI saved you hours.
- We used AI as a “math/physics TA” for having a deeper understanding of the paper we were replicating. A big blocker was Instead of spending hours bouncing between the paper, lecture notes, and random StackExchange posts, we pasted the relevant equations and asked the AI to restate the physics in plain language. That one guided explanation compressed what would easily have been a full afternoon of background reading and derivations into ~20–30 minutes, and let us move straight on to implementing and testing the actual kernels.

### Learn: One instance where you altered your prompting strategy (provided context, created a skills.md file, etc) to get better results from your interaction with the AI agent.
- Our LABS + QE-MTS project spanned dozens of code cells and several iterations of CUDA-Q kernels, MTS variants, and plotting logic. Early on, we noticed that the AI often “forgot” earlier design choices (e.g., our ±1 spin convention or the exact LABS energy definition) once the conversation got long. In one case, we asked it to extend a CUDA-Q kernel and it quietly switched back to a 0/1 encoding and a different energy formula because the earlier context had fallen out of its window. After that, we deliberately changed our prompting strategy: before each new request we pasted a short recap block with the key invariants (e.g., “spins are ±1, energy is labs_energy_pm1(s) = Σₖ Cₖ² with Cₖ = Σᵢ sᵢ sᵢ₊ₖ”, plus the current function signatures), and explicitly told the model “do not change these assumptions.”

### Fail: One instance where AI failed/hallucinated, and how you fixed it.
- When we first asked the AI to write the full QE-MTS workflow, it produced CUDA-Q code that looked perfectly reasonable: a trotterized_circuit kernel plus interaction sets G2 and G4. But later we realized it had silently hard-wired the interaction structure: it was effectively using constant G2/G4 derived for N = 5 and then reusing those for any larger N. That meant the counteradiabatic evolution was only ever acting non-trivially on the first 5 qubits of the LABS Hamiltonian, even when we set N = 13 or N = 18. The rest of the system was basically untouched. Symptom-wise, this showed up as QE-MTS having suspiciously tiny and almost N-independent runtime on the “quantum” part, and the number of two- and four-body terms stayed constant when we printed len(G2) and len(G4) for different N. At that point it was clear the AI code wasn’t implementing Eq. (15) from the paper correctly; it had hallucinated a fixed interaction pattern instead of generating it as a function of N.

### Context Dump: Share any prompts, skills.md files, MCP etc. that demonstrate thoughtful prompting. 
- Domain:
  - Quantum computing (QAOA, adiabatic / counterdiabatic methods, CUDA-Q)
  - Classical metaheuristics (memetic tabu search, local search, genetic-style operators)
  - Numerical optimization (SciPy, gradient-free methods)

- Project:
    - QAOA algorithm
    - LABS energy function for ±1 spin sequences
    - Classical MTS (combine, mutate, tabu search)
    - Counterdiabatic Trotterized CUDA-Q circuit

- Constraints:
  - Maintain consistent convention: spins are ±1, LABS energy E = Σ_k C_k²
  - Do not change the LABS definition or spin encoding unless explicitly asked
  - Prefer clear, testable Python / CUDA-Q code over clever one-liners
