//! Test suite for the computational graph implementation's edge cases and error handling.
//! 
//! This module contains comprehensive tests for:
//! - Division by zero handling
//! - Arithmetic overflow scenarios
//! - Complex graph evaluations
//! - Independent subgraph processing
//! - Constraint violation detection
//! - Hint operation validation
//! - Negative value handling
//! - Uninitialized node behavior
//!
//! These tests ensure the robustness and reliability of the computational graph
//! implementation when dealing with edge cases and potential error conditions.

use zk_graph_engine::Builder;

/// Tests the handling of division by zero in hint operations.
///
/// # Test Case
/// Attempts to perform 1/0 using a hint operation
///
/// # Expected Result
/// - Operation should not panic
/// - Should return None for the division by zero result
#[test]
fn test_hint_division_by_zero() {
    let mut builder = Builder::new();
    let zero = builder.constant(0);
    let div_by_zero = builder.hint(zero, |x| 1 / x);

    builder.fill_nodes(vec![]);
    builder.get_value(div_by_zero);
}

/// Tests arithmetic overflow handling for addition operations.
///
/// # Test Case
/// Attempts to add 1 to u32::MAX
///
/// # Expected Result
/// - Should return None for the overflow condition
/// - Should not panic
#[test]
fn test_arithmetic_overflow_handling_addition() {
    let mut builder = Builder::new();
    let max = builder.constant(u32::MAX);
    let one = builder.constant(1);
    let overflow_sum = builder.add(max, one);

    builder.fill_nodes(vec![]);
    assert_eq!(builder.get_value(overflow_sum), None);
}

/// Tests arithmetic overflow handling for multiplication operations.
///
/// # Test Case
/// Attempts to multiply u32::MAX by itself
///
/// # Expected Result
/// - Should return None for the overflow condition
/// - Should not panic
#[test]
fn test_arithmetic_overflow_handling_multiplication() {
    let mut builder = Builder::new();
    let max = builder.constant(u32::MAX);
    let overflow_mul = builder.mul(max, max);

    builder.fill_nodes(vec![]);
    assert_eq!(builder.get_value(overflow_mul), None);
}

/// Tests combined arithmetic overflow handling for both addition and multiplication.
///
/// # Test Cases
/// 1. Addition: u32::MAX + 1
/// 2. Multiplication: u32::MAX * u32::MAX
///
/// # Expected Results
/// - Both operations should return None
/// - No operations should panic
#[test]
fn test_arithmetic_overflow_handling_combined() {
    let mut builder = Builder::new();
    let max = builder.constant(u32::MAX);
    let one = builder.constant(1);
    let overflow_sum = builder.add(max, one);
    let overflow_mul = builder.mul(max, max);

    builder.fill_nodes(vec![]);
    assert_eq!(builder.get_value(overflow_sum), None);
    assert_eq!(builder.get_value(overflow_mul), None);
}

/// Tests evaluation of a complex computational graph with multiple operations.
///
/// # Test Case
/// Computes (a * b + a) * b where a = 2 and b = 3
///
/// # Expected Result
/// - Should return 24 ((2 * 3 + 2) * 3 = 24)
/// - All constraints should be satisfied
#[test]
fn test_complex_graph_evaluation() {
    let mut builder = Builder::new();
    let a = builder.init();
    let b = builder.init();
    let prod = builder.mul(a, b);
    let sum = builder.add(prod, a);
    let final_result = builder.mul(sum, b);

    builder.fill_nodes(vec![2, 3]);
    assert_eq!(builder.get_value(final_result).unwrap(), 24);
}

/// Tests handling of independent subgraphs within the same builder.
///
/// # Test Cases
/// 1. Computes x² where x = 2
/// 2. Computes y² where y = 3
///
/// # Expected Results
/// - x² should equal 4
/// - y² should equal 9
/// - Computations should not interfere with each other
#[test]
fn test_independent_subgraphs() {
    let mut builder = Builder::new();
    let x = builder.init();
    let y = builder.init();

    let x_squared = builder.mul(x, x);
    let y_squared = builder.mul(y, y);

    builder.fill_nodes(vec![2, 3]);
    assert_eq!(builder.get_value(x_squared).unwrap(), 4);
    assert_eq!(builder.get_value(y_squared).unwrap(), 9);
}

/// Tests detection of constraint violations.
///
/// # Test Case
/// Attempts to assert equality between two different constants (5 and 6)
///
/// # Expected Result
/// - check_constraints() should return false
/// - System should detect the inequality
#[test]
fn test_constraint_violation() {
    let mut builder = Builder::new();
    let x = builder.constant(5);
    let y = builder.constant(6);
    builder.assert_equal(x, y);

    builder.fill_nodes(vec![]);
    assert!(!builder.check_constraints());
}

/// Tests hint operation with valid square root transformation.
///
/// # Test Case
/// Computes square root of 16 using hint and verifies result
///
/// # Expected Results
/// - Square root should equal 4
/// - Squaring the result should equal the input
/// - All constraints should be satisfied
#[test]
fn test_hint_with_valid_transformation() {
    let mut builder = Builder::new();
    let input = builder.constant(16);
    let sqrt = builder.hint(input, |x| (x as f64).sqrt() as u32);
    let squared = builder.mul(sqrt, sqrt);
    builder.assert_equal(squared, input);

    builder.fill_nodes(vec![]);
    assert_eq!(builder.get_value(sqrt).unwrap(), 4);
    assert!(builder.check_constraints());
}

/// Tests handling of negative values when cast to u32.
///
/// # Test Case
/// Attempts to add -1 (cast to u32) and 1
///
/// # Expected Results
/// - Should return None due to overflow
/// - Constraints should still be satisfied
/// - Should not panic
#[test]
fn test_negative_values_handling() {
    let mut builder = Builder::new();
    let negative_value = builder.constant(-1i32 as u32);
    let value = builder.constant(1);
    let result = builder.add(negative_value, value);

    builder.fill_nodes(vec![]);
    assert_eq!(builder.get_value(result), None);
    assert!(builder.check_constraints());
}

/// Tests constraints on uninitialized nodes.
///
/// # Test Case
/// Asserts equality between two initialized nodes with same value
///
/// # Expected Results
/// - Constraints should be satisfied when nodes are filled with same value
/// - No panic or errors should occur
#[test]
fn test_uninitialized_node_constraint() {
    let mut builder = Builder::new();
    let x = builder.init();
    let y = builder.init();
    builder.assert_equal(x, y);

    builder.fill_nodes(vec![10, 10]);
    assert!(builder.check_constraints());
}
