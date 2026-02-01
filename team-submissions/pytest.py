import unittest
import numpy as np
import cudaq
from cudaq import spin

# Import the functions from your provided code
# (Assuming the code is in the current namespace or imported)
from labs_solver import (
    labs_hamiltonian,
    calculate_energy_fast,
    canonicalize_bitstring,
    get_transferred_parameters,
    tabu_search_fast,
    get_labs_interactions_flat,
    bitstring_to_array
)
import unittest
import numpy as np
import cudaq
from cudaq import spin

# Import the functions from your provided code
from labs_solver import (
    labs_hamiltonian,
    calculate_energy_fast,
    canonicalize_bitstring,
    get_transferred_parameters,
    tabu_search_fast,
    get_labs_interactions_flat,
    bitstring_to_array
)

class TestLABSImplementation(unittest.TestCase):
    
    def setUp(self):
        """Set up known ground truths for small N (Barker Codes)."""
        self.ground_truths = {
            3: (1, [1, 1, 0]),
            4: (2, [1, 1, 1, 0]),
            5: (2, [1, 1, 1, 0, 1]),
            7: (3, [1, 1, 1, 0, 0, 1, 0])
        }
        
    def test_energy_bounds_and_negative_cases(self):
        """
        Verifies:
        1. Energy is never negative (Sum of Squares property).
        2. Upper Bound: The energy of 'trivial' sequences (all 0s or all 1s).
           For a sequence of length N of all 1s (or 0s), the autocorrelation C_k = N-k.
           Energy E = sum_{k=1}^{N-1} (N-k)^2.
        """
        print("\nTesting Energy Bounds (Non-negative & Upper Limits)...")
        
        # Test for a range of N (e.g., 3 to 20)
        for n in range(3, 21):
            # --- 1. Theoretical Upper Bound Calculation ---
            # For all 1s (or all 0s), C_k = N - k
            # E = sum_{k=1}^{N-1} (N-k)^2 = sum_{j=1}^{N-1} j^2
            # Sum of squares formula: n(n+1)(2n+1)/6
            # Here the limit is N-1, so: (N-1)(N)(2(N-1)+1)/6
            theoretical_max = (n - 1) * n * (2 * n - 1) // 6
            
            # --- 2. Construct all-ones and all-zeros sequences ---
            all_ones = np.ones(n, dtype=np.int8)
            all_zeros = np.zeros(n, dtype=np.int8)
            
            # --- 3. Calculate Energy using your function ---
            e_ones = calculate_energy_fast(all_ones)
            e_zeros = calculate_energy_fast(all_zeros)
            
            # --- Check Upper Bounds ---
            self.assertEqual(e_ones, theoretical_max, 
                             f"All-ones sequence for N={n} did not match theoretical max energy.")
            self.assertEqual(e_zeros, theoretical_max, 
                             f"All-zeros sequence for N={n} did not match theoretical max energy.")
            
            # --- 4. Random Checks for Non-Negativity & Bound Adherence ---
            # Generate 10 random sequences per N
            for _ in range(10):
                rand_seq = np.random.randint(0, 2, n, dtype=np.int8)
                e_rand = calculate_energy_fast(rand_seq)
                
                # Assert Non-negative
                self.assertGreaterEqual(e_rand, 0, 
                                        f"Found impossible negative energy for seq {rand_seq}")
                
                # Assert Within Bounds
                # No random sequence should exceed the energy of the all-1s sequence
                self.assertLessEqual(e_rand, theoretical_max,
                                     f"Random sequence energy {e_rand} exceeded theoretical max {theoretical_max}")

            print(f"  N={n}: Max Energy {theoretical_max} verified. Non-negative checks passed.")
    
    # =================================================================
    # 1. PHYSICS & HAMILTONIAN VERIFICATION
    # =================================================================

    def test_classical_energy_correctness(self):
        """
        Verify the vectorized NumPy energy calculation against known Barker code energies.
        This ensures the objective function itself is correct.
        """
        print("\nTesting Classical Energy Calculation...")
        for n, (expected_energy, seq) in self.ground_truths.items():
            arr = np.array(seq, dtype=np.int8)
            calc_energy = calculate_energy_fast(arr)
            print(f"  N={n}, Seq={seq}, Expected={expected_energy}, Got={calc_energy}")
            self.assertEqual(calc_energy, expected_energy, 
                             f"Energy mismatch for N={n}")

    def test_hamiltonian_classical_correspondence(self):
        print("\nTesting Hamiltonian <-> Classical Correspondence...")
        
        # Test sequences
        test_cases = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 0, 1]] 
        n_qubits = 5
        hamiltonian = labs_hamiltonian(n_qubits)
        
        # The Missing Piece: Constant Offset
        # Classical Energy = Quantum Interaction Energy + N(N-1)/2
        offset = n_qubits * (n_qubits - 1) / 2
        
        for seq in test_cases:
            # 1. Calculate Classical Energy (Ground Truth)
            classical_energy = calculate_energy_fast(np.array(seq))
            
            # 2. Measure Quantum Energy
            kernel = cudaq.make_kernel()
            q = kernel.qalloc(n_qubits)
            for i, bit in enumerate(seq):
                if bit == 0: 
                    kernel.x(q[i]) # Map 0 to |1> for measurement consistency
            
            # 3. Add the Offset
            obs_val = cudaq.observe(kernel, hamiltonian).expectation()
            total_quantum_energy = obs_val + offset
            
            print(f"  Seq={seq}: Class={classical_energy}, Quant+Offset={total_quantum_energy}")
            
            # 4. Assert they match (using a small tolerance for floating point math)
            self.assertAlmostEqual(classical_energy, total_quantum_energy, places=5)
            
    # =================================================================
    # 2. SYMMETRY & CANONICALIZATION VERIFICATION
    # =================================================================

    def test_symmetry_canonicalization(self):
        """
        Verify that bit-flip and reversal symmetries map to the same canonical string.
        """
        print("\nTesting Symmetry Canonicalization...")
        
        # Base sequence
        s = [1, 1, 0, 1, 0] # 5 bits
        
        # Transformations
        s_arr = np.array(s)
        s_flip = (1 - s_arr).tolist()
        s_rev = s_arr[::-1].tolist()
        s_flip_rev = (1 - s_arr[::-1]).tolist()
        
        canonical_base = canonicalize_bitstring(s)
        
        # Assert all variations map to the exact same list
        self.assertEqual(canonical_base, canonicalize_bitstring(s_flip), "Bit-flip symmetry failed")
        self.assertEqual(canonical_base, canonicalize_bitstring(s_rev), "Reversal symmetry failed")
        self.assertEqual(canonical_base, canonicalize_bitstring(s_flip_rev), "Flip+Reverse symmetry failed")
        
        print("  Symmetries verified: s, NOT(s), REV(s), NOT(REV(s)) all map to same canonical form.")

    # =================================================================
    # 3. ALGORITHMIC & HELPER VERIFICATION
    # =================================================================

    def test_tabu_search_improvement(self):
        """
        Verify that the Tabu search actually improves a suboptimal solution.
        """
        print("\nTesting Tabu Search Effectiveness...")
        N = 10
        # Start with a very poor sequence (all ones)
        initial_seq = [1] * N
        initial_energy = calculate_energy_fast(np.array(initial_seq))
        
        # Run tabu search - ONLY catching the one returned value (the list)
        improved_seq = tabu_search_fast(N, initial_seq, max_iters=20)
        
        # Calculate the energy manually in the test
        improved_energy = calculate_energy_fast(np.array(improved_seq))
        
        print(f"  Initial Energy: {initial_energy}, Improved Energy: {improved_energy}")
        
        # The improved energy should be strictly less than the energy of all ones
        self.assertLess(improved_energy, initial_energy, 
                        "Tabu search failed to find a better sequence than all-ones.")
    def test_parameter_transfer_schedule(self):
        """
        Verify the parameter schedule is monotonic/logical (Linear Ramp Check).
        """
        print("\nTesting Parameter Transfer Schedule...")
        p = 5
        gamma, beta = get_transferred_parameters(p, 10)
        
        # Gamma should increase (approx)
        is_gamma_increasing = all(gamma[i] <= gamma[i+1] for i in range(len(gamma)-1))
        # Beta should decrease (approx)
        is_beta_decreasing = all(beta[i] >= beta[i+1] for i in range(len(beta)-1))
        
        self.assertTrue(is_gamma_increasing, "Gamma schedule is not increasing")
        self.assertTrue(is_beta_decreasing, "Beta schedule is not decreasing")
        print(f"  Gamma: {gamma}")
        print(f"  Beta:  {beta}")

    def test_interaction_indices(self):
        """
        Verify interaction lists are within bounds and unique.
        """
        print("\nTesting Interaction Indices...")
        n = 10
        fb_0, fb_1, fb_2, fb_3, tb_0, tb_1 = get_labs_interactions_flat(n)
        
        # Check 4-body bounds
        for i in range(len(fb_0)):
            indices = [fb_0[i], fb_1[i], fb_2[i], fb_3[i]]
            self.assertTrue(max(indices) < n, f"4-body index out of bounds: {indices}")
            self.assertEqual(len(set(indices)), 4, f"4-body indices not unique: {indices}")
            
        # Check 2-body bounds
        for i in range(len(tb_0)):
            indices = [tb_0[i], tb_1[i]]
            self.assertTrue(max(indices) < n, f"2-body index out of bounds: {indices}")
            self.assertNotEqual(tb_0[i], tb_1[i], f"2-body self-interaction: {indices}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)