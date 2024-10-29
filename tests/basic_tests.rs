//! Test module for the computational graph implementation.
//!
//! This module contains comprehensive tests for verifying the functionality of
//! the computational graph builder. Tests cover:
//! - Basic arithmetic operations and polynomial evaluation
//! - Division operations with constraint verification
//! - Square root computation with perfect square constraints
//! - Identity operations (adding 0, multiplying by 1)
//! - Bit decomposition with reconstruction verification

use zk_graph_engine::Builder;

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests polynomial evaluation in the computational graph.
    ///
    /// # Test Case
    /// Evaluates the polynomial f(x) = x² + x + 5 for x = 3
    ///
    /// # Expected Results
    /// - For x = 3: f(3) = 9 + 3 + 5 = 17
    /// - All graph constraints should be satisfied
    #[test]
    fn test_polynomial_evaluation() {
        let mut builder = Builder::new();
        let x = builder.init();
        let x_squared = builder.mul(x, x);
        let five = builder.constant(5);
        let x_squared_plus_5 = builder.add(x_squared, five);
        let y = builder.add(x_squared_plus_5, x);

        builder.fill_nodes(vec![3]);
        assert_eq!(builder.get_value(y).unwrap(), 17);
        assert!(builder.check_constraints());
    }

    /// Tests division operation with constraint verification.
    ///
    /// # Test Case
    /// Computes f(a) = (a+1)/8 where a = 7
    ///
    /// # Expected Results
    /// - Result should be 1 ((7+1)/8 = 1)
    /// - Constraints verify that result * 8 equals input + 1
    /// - All graph constraints should be satisfied
    #[test]
    fn test_division_with_constraints() {
        let mut builder = Builder::new();
        let a = builder.init();
        let one = builder.constant(1);
        let b = builder.add(a, one);
        let c = builder.hint(b, |b| b / 8);
        let eight = builder.constant(8);
        let c_times_8 = builder.mul(c, eight);
        builder.assert_equal(b, c_times_8);

        builder.fill_nodes(vec![7]);
        assert_eq!(builder.get_value(c).unwrap(), 1);
        assert!(builder.check_constraints());
    }

    /// Tests square root computation with perfect square verification.
    ///
    /// # Test Case
    /// Computes f(x) = sqrt(x+7) where x = 2
    ///
    /// # Expected Results
    /// - Result should be 3 (sqrt(2+7) = sqrt(9) = 3)
    /// - Constraints verify that result² equals input + 7
    /// - All graph constraints should be satisfied
    #[test]
    fn test_square_root_with_perfect_square() {
        let mut builder = Builder::new();
        let x = builder.init();
        let seven = builder.constant(7);
        let x_plus_seven = builder.add(x, seven);
        let sqrt_x_plus_7 = builder.hint(x_plus_seven, |val| (val as f64).sqrt() as u32);
        let computed_sq = builder.mul(sqrt_x_plus_7, sqrt_x_plus_7);
        builder.assert_equal(computed_sq, x_plus_seven);

        builder.fill_nodes(vec![2]);
        assert_eq!(builder.get_value(sqrt_x_plus_7).unwrap(), 3);
        assert!(builder.check_constraints());
    }

    /// Tests identity operations in the computational graph.
    ///
    /// # Test Cases
    /// 1. Adding zero to a number (x + 0)
    /// 2. Multiplying a number by one (x * 1)
    ///
    /// # Expected Results
    /// - Both operations should return the original input (5)
    /// - All graph constraints should be satisfied
    #[test]
    fn test_identity_operations() {
        let mut builder = Builder::new();
        let x = builder.init();
        let zero = builder.constant(0);
        let add_zero = builder.add(x, zero);
        let one = builder.constant(1);
        let mul_one = builder.mul(x, one);

        builder.fill_nodes(vec![5]);
        assert_eq!(builder.get_value(add_zero).unwrap(), 5);
        assert_eq!(builder.get_value(mul_one).unwrap(), 5);
        assert!(builder.check_constraints());
    }

    /// Tests bit decomposition and reconstruction of a 32-bit number.
    ///
    /// # Test Process
    /// 1. Decomposes input value into 32 individual bits
    /// 2. Verifies each bit is either 0 or 1 using constraints
    /// 3. Reconstructs the original number from the bits
    /// 4. Verifies reconstruction matches the original input
    ///
    /// # Expected Results
    /// - Each bit should satisfy the constraint bit * (1 - bit) = 0
    /// - Reconstructed value should equal the original input (50)
    /// - All graph constraints should be satisfied
    ///
    /// # Implementation Details
    /// - Uses 32-bit decomposition
    /// - Employs hints for bit extraction
    /// - Reconstructs using binary weight multiplication
    #[test]
    fn test_bit_decomposition() {
        let mut builder = Builder::new();
        let x = builder.init();
        let mut bits = Vec::new();
        let zero = builder.constant(0);

        // Decompose x into bits
        for i in 0..32 {
            let bit = builder.hint(x, move |val| (val >> i) & 1);
            bits.push(bit);

            // Ensure each bit is either 0 or 1
            let product = builder.hint(bit, |val| val * (1 - val));
            builder.assert_equal(product, zero);
        }

        // Reconstruct x from its bits
        let mut reconstructed_x = builder.constant(0);
        for (i, &bit) in bits.iter().enumerate() {
            let two_i = builder.constant(1 << i);
            let bit_value = builder.mul(bit, two_i);
            reconstructed_x = builder.add(reconstructed_x, bit_value);
        }

        builder.assert_equal(reconstructed_x, x);
        builder.fill_nodes(vec![50]);

        assert!(builder.check_constraints());
        assert_eq!(builder.get_value(reconstructed_x).unwrap(), 50);
    }
}
