# Product Requirements Document (PRD)

**Project Name: LABS With Accelerated GPU Workflows**  
**Team Name: KernelCompilers**   
**GitHub Repository:** [https://github.com/Hoponga/2026-NVIDIA](https://github.com/Hoponga/2026-NVIDIA) 

## 1\. Team Roles & Responsibilities \[You can DM the judges this information instead of including it in the repository\]

| Role | Name | GitHub Handle | Discord Handle |
| :---- | :---- | :---- | :---- |
| **Project Lead** (Architect) | Keshav | \[@kdeoskar\] | \[@kdeoskar\] |
| **GPU Acceleration PIC** (Builder) | Kailash | @Hoponga | @kaleguy |
| **Quality Assurance PIC** (Verifier) | Evan  | @evjwang | @evjwg |
| **Technical Marketing PIC** (Storyteller) | Angela | @AngelaWuRX | @rxw\_fish |

---

## 2\. The Architecture

**Owner:** Project Lead

### Choice of Quantum Algorithm

* **Algorithm:** STA-Inspired ADAPT-QAOA for LABS.  
    
* **Motivation:**  
  *  QAOA is a foundational optimization algorithm, however it has been shown that the original ansatz is generally suboptimal. We hope to implement the recently introduced Adapt QAOA framework, which iteratively adapts the ansatz to the specific optimization problem.  
      
  * Further, QAOA  is suited to our task of seeding as the number of QAOA layers effectively interpolates between a Gibbs distribution (p=1) and a single ground state (p-\>infinity). This gives us a tunable collection of states to use as the seed population, unlike other optimization algorithms such as VQE, which produce one approximate ground state and may get stuck in a local minimum.

### Literature Review

* **Reference:** Adaptive quantum approximate optimization algorithm for solving combinatorial problems on a quantum computer; Linghua Zhu, Ho Lun Tang, George S. Barron, F. A. Calderon-Vargas; [https://doi.org/10.1103/PhysRevResearch.4.033029](https://doi.org/10.1103/PhysRevResearch.4.033029) 


* **Relevance:** \[How does this paper support your plan?\]  
  * Rather than using a fixed-form QAOA, ADAPT-QAOA builds its circuit iteratively from a pool of candidate mixer operators, selecting at each step the operator whose gradient with respect to the cost Hamiltonian is largest. For LABS, we want to replicate that.  
  * More importantly, the paper reports that its adaptive mixer selection appears connected to the theory of shortcuts to adiabaticity, but this connection is not fully explored in realistic settings. In our project, we want to investigate a practical STA-flavored ansatz for the LABS problem under realistic GPU constraints. Since for LABS, we define the cost Hamiltonian as the LABS energy and construct an operator pool from Pauli strings, then constrain the ansatz by a depth or a budget compatible with our GPU credits, resulting in “STA-flavored” ADAPT-QAOA.  
      
      
* **Reference:** Pelofske, E. (2025). Depth One Quantum Alternating Operator Ansatz as an Approximate Gibbs Distribution Sampler. *arXiv preprint arXiv:2510.10345*   
    
* **Relevance:**   
  * Motivates the usage of low-p QAOA algorithms for seeding

---

## 3\. The Acceleration Strategy

**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)

* **Strategy:**   
  * We will test with our own A100 GPU instance to look at perf baselines of our QE-MTS as well as our implementation of ADAPT-QAOA. For all the CUDA-Q operations, these will natively be run on the GPU backend through CUDA-Q. We will look into different techniques for scheduling the kernels. That is, the Trotter evolution is done with a sequential for loop of sub-circuits in cuda-q, and so we plan to explore ways to potentially reduce scheduling overhead \+ introduce parallelism into the classical components (of tabu search/MTS iteration).   
  * One key issue with the QE-MTS algorithm as is is the limited VRAM constraint of GPUs when the Trotterized circuit for time evolution of the LABS Hamiltonian is simulated classically on the GPU. A back of envelope calculation allocates 2^30 x 8 \~= 8 GB for the statevector (in FP64) of a 30 qubit system, which can comfortably fit on a single A100. Even when we extend this to  40 qubits, that explodes to 8 terabytes, all of a sudden infeasible without a large-scale cluster.   
  * To solve this exploding VRAM problem, we instead propose ADAPT-QAOA, which builds the circuit itself in an iterative fashion and relaxes the static VRAM requirement of QE-MTS. 

### Classical Acceleration (MTS)

* **Strategy:**  
  * *Example:* "The standard MTS evaluates neighbors one by one. We will use `cupy` to rewrite the energy function to evaluate a batch of 1,000 neighbor flips simultaneously on the GPU."

### Hardware Targets

* **Dev Environment:** Qbraid (CPU) for logic, Brev L4 for initial GPU testing, own A100s  
* **Production Environment:** Brev A100-80GB for final N=100 benchmarks

---

## 4\. The Verification Plan

**Owner:** Quality Assurance PIC

### Unit Testing Strategy

* **Framework:** pytest  
* **AI Hallucination Guardrails:** \[How do you know the AI code is right?\]  
  * We will will use pytest in combination with the Hypothesis library to ensure that AI generated code never outputs physically impossible data  
  * Binary sequence energies are bounded by the model (all 1s and all 0s, no negative energies)  
  * The number of interaction terms is a combinatorial value that we can verify with the AI code  
  * Select a few random LABS sequences and their generated energies by the AI code, validate these energies with a slow but 100% correct classical function  
    

### Core Correctness Checks

* **Check 1 (Symmetry):**  
  * LABS sequence and its negation have identical energies. We will assert that these two energies are equal `energy(S) == energy(-S)`  
  * LABS sequence and its reflection (bit reversal) have identical energies. We will assert these two energies are equal `energy(s_i… s_i+1) == energy(s_n-i… s_n-(i+1))`  
      
* **Check 2 (Ground Truth):**  
  * For N \< 5, we can easily find all known energies. Our test suite will assert that our GPU kernel returns the correct energies for n \< 5\.  
  * For N \< 66, the LABS sequence with the global low energy has been found, we will check that our algorithm converges on the same binary sequence

---

## 5\. Execution Strategy & Success Metrics

**Owner:** Technical Marketing PIC

### Agentic Workflow

* **Plan:** \[How will you orchestrate your tools?\]  
  * *Example:* "We are using Cursor as the IDE. We have created a `skills.md` file containing the CUDA-Q documentation so the agent doesn't hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes the error log back into the Agent to refactor."  
  * We will use qBraid \+ Brev as our compute environment and Cursor \+ chatGPT as our “Vibe coding” agents.  
  * Design phase: This will be human-led. Project PIC and QA PIC will first write down the following in [context.md](http://context.md) so the agent doesn’t hallucinate APIs.   
    * LABS energy definition  
    * DCQA/ADAPT-QAOA kernel signature  
    * Unit tests  
  * Then we will prompt the AI agent to generate or refactor  
  * QA PIC will run [tests.py](http://tests.py)  
  * After tests pass, GPU PIC switches to GPU on Brev and runs, all GPU runs are scripted with fixed configurations so they can be replayed  
  * Technical marketing PIC will collects run logs include the following:  
    * N  
      * Method  
      * Best energy  
      * Runtime  
      * Gate counts  
    * Then put into a CSV and uses Python and matplot to generate plots for the final presentations which will highlight success metrics

### Success Metrics

* **Metric 1 (Approximation ratio):** Approximation Ratio within 0.05 up to N \= 60\.  
* **Metric 2 (Speedup):** 5x speedup over the CPU-only Tutorial baseline  
* **Metric 3 (Scale):** Successfully run a simulation for N \> 100

### Visualization Plan

* **Plot 1:** "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed  
* **Plot 2:** “Convergence Rate” (Computation time vs. Bitstring Length) for base CPU implementation vs GPU-accelerated implementation  
* **Plot 3:** “Convergence Rate” (Energy vs. Iteration count) comparing Adaptive vs Standard QAOA Implementations 

---

## 6\. Resource Management Plan

**Owner:** GPU Acceleration PIC

* **Plan:** \[How will you avoid burning all your credits?\]  
  * First we have used qBraid to test that our algorithm works for correctness relative to classical counterparts with smaller qubit numbers. Then, in tandem, we are using our own GPUs (A100) to verify that our algorithm extends to the GPU backend via cuda-q and get timing numbers for comparison  
  * Finally, once we have verified both timing and correctness of our algorithm, we can deploy with larger GPU clusters on Brev to verify that the entire algorithm works and understand how it scales with larger \# of qubits. 
