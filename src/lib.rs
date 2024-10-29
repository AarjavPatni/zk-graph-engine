//! A computational graph library for mathematical operations.
//!
//! This crate provides a builder pattern for creating and executing computational graphs
//! with support for basic arithmetic operations, constraints, and hints.
//!
//! ## Features
//! - Basic arithmetic operations (addition, multiplication)
//! - Input nodes and constant values
//! - Constraint checking between nodes
//! - Hint system for custom operations
//! - Verbose logging support for debugging
//!
//! ## Example
//! ```rust
//! use zk_graph_engine::Builder;
//!
//! let mut builder = Builder::new(true);
//! let x = builder.init();  // Create input node
//! let y = builder.constant(5);  // Create constant node
//! let sum = builder.add(x, y);  // Add nodes
//!
//! builder.fill_nodes(vec![3]);  // Set input value
//! assert_eq!(builder.get_value(sum).unwrap(), 8);
//! ```

use log::{debug, error, info};
use simple_logger::SimpleLogger;
use std::sync::Once;

/// A builder for creating and managing computational graphs.
///
/// The `Builder` struct allows for the construction of computational graphs
/// consisting of nodes that perform arithmetic operations and can be checked
/// for constraints.
///
/// # Components
/// - `nodes`: List of nodes in the graph.
/// - `constraints`: List of pairs of node indices that must be equal.
/// - `hints`: List of tuples containing a target node index, a source node index, and a hint function.
/// - `input_nodes_count`: Number of input nodes.
/// - `verbose`: Whether to print debug information.
pub struct Builder {
    nodes: Vec<Box<Node>>,
    constraints: Vec<(usize, usize)>,
    hints: Vec<(usize, usize, Box<dyn Fn(u32) -> u32>)>,
    input_nodes_count: usize,
    verbose: bool,
}

/// Enum representing the types of operations that can be performed on nodes.
///
/// - `None`: No operation. Used for input and constant nodes.
/// - `Add`: Addition of two nodes.
/// - `Mul`: Multiplication of two nodes.
/// - `Hint`: A custom operation defined by a hint function upon calling `fill_nodes`.
enum Operation {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    Hint(usize),
}

/// A node in the computational graph.
/// Values of the node are computed when `fill_nodes` is called.
///
/// # Components
/// - `value`: The value of the node, if it has been computed.
/// - `operation`: The operation used to compute the node.
/// - `is_input`: Whether the node is an input node.
/// 
struct Node {
    value: Option<u32>,
    operation: Operation,
    is_input: bool,
}

/// Methods for the `Node` struct.
impl Node {
    /// Creates a new input node.
    fn new() -> Self {
        Self {
            value: None,
            operation: Operation::None,
            is_input: true,
        }
    }

    /// Creates a new node with a constant value.
    fn new_const(value: u32) -> Self {
        Self {
            value: Some(value),
            operation: Operation::None,
            is_input: false,
        }
    }
}

/// Methods for the `Builder` struct.
impl Builder {
    /// Creates a new `Builder`.
    ///
    /// # Parameters
    /// - `verbose`: If true, enables logging of operations for debugging purposes.
    pub fn new(verbose: bool) -> Self {
        // Initialize the logger once
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            if verbose {
                SimpleLogger::new().init().unwrap();
            }
        });

        Self {
            nodes: vec![],
            constraints: vec![],
            hints: vec![],
            input_nodes_count: 0,
            verbose,
        }
    }

    /// Initializes an input node in the graph.
    ///
    /// Returns the index of the newly created input node.
    pub fn init(&mut self) -> usize {
        let node = Box::new(Node::new());
        self.nodes.push(node);
        self.input_nodes_count += 1;
        let idx = self.nodes.len() - 1;
        if self.verbose {
            info!("INIT_INPUT – node {}", idx);
        }
        idx
    }

    /// Initializes a node in the graph set to a constant value.
    ///
    /// # Parameters
    /// - `value`: The constant value to assign to the node.
    ///
    /// Returns the index of the newly created constant node.
    pub fn constant(&mut self, value: u32) -> usize {
        let node = Box::new(Node::new_const(value));
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        if self.verbose {
            debug!("INIT_CONST – node {} = {}", idx, value);
        }
        idx
    }

    /// Adds two nodes in the graph, returning a new node.
    ///
    /// # Parameters
    /// - `a`: The index of the first node to add.
    /// - `b`: The index of the second node to add.
    ///
    /// Returns the index of the newly created node.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let node = Box::new(Node {
            value: None,
            operation: Operation::Add(a, b),
            is_input: false,
        });
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        if self.verbose {
            info!("SET_ADD_OPT – node {} = node {} + node {}", idx, a, b);
        }
        idx
    }

    /// Multiplies two nodes in the graph, returning a new node.
    ///
    /// # Parameters
    /// - `a`: The index of the first node to multiply.
    /// - `b`: The index of the second node to multiply.
    ///
    /// Returns the index of the newly created node.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let node = Box::new(Node {
            value: None,
            operation: Operation::Mul(a, b),
            is_input: false,
        });
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        if self.verbose {
            info!("SET_MUL_OPT – node {} = node {} * node {}", idx, a, b);
        }
        idx
    }

    /// Asserts that two nodes are equal.
    ///
    /// # Parameters
    /// - `a`: The index of the first node.
    /// - `b`: The index of the second node.
    pub fn assert_equal(&mut self, a: usize, b: usize) {
        self.constraints.push((a, b));
        if self.verbose {
            info!("SET_CONSTRAINT – Assert node {} == node {}", a, b);
        }
    }

    /// Computes the value of a node based on its operation.
    ///
    /// # Parameters
    /// - `node_idx`: The index of the node to compute.
    ///
    /// Returns the computed value of the node, or `None` if it cannot be computed.
    fn compute_node_value(&self, node_idx: usize) -> Option<u32> {
        let node = &self.nodes[node_idx];

        match &node.operation {
            Operation::None => node.value,
            Operation::Add(a, b) => {
                let a_val = self.nodes[*a].value?;
                let b_val = self.nodes[*b].value?;
                if let Some(val) = a_val.checked_add(b_val) {
                    debug!("COMPUTE_ADD – node {} = node {} + node {}", node_idx, a, b);
                    Some(val)
                } else {
                    error!(
                        "ERR_OVERFLOW_ADD – Overflow when adding {} and {}",
                        a_val, b_val
                    );
                    None
                }
            }
            Operation::Mul(a, b) => {
                let a_val = self.nodes[*a].value?;
                let b_val = self.nodes[*b].value?;
                if let Some(val) = a_val.checked_mul(b_val) {
                    debug!("COMPUTE_MUL – node {} = node {} * node {}", node_idx, a, b);
                    Some(val)
                } else {
                    error!(
                        "ERR_OVERFLOW_MUL – Overflow when multiplying {} and {}",
                        a_val, b_val
                    );
                    None
                }
            }
            // TODO: Change to use HashMap
            Operation::Hint(source) => {
                if let Some(value) = self.nodes[*source].value {
                    if let Some((_, _, hint_fn)) = self
                        .hints
                        .iter()
                        .find(|(target, src, _)| *target == node_idx && *src == *source)
                    {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            hint_fn(value)
                        }));
                        // TODO: Handle error more gracefully
                        match result {
                            Ok(val) => Some(val),
                            Err(_) => {
                                error!("ERR_DIV_BY_ZERO – Attempt to divide by zero");
                                panic!("ERR_DIV_BY_ZERO – Attempt to divide by zero");
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Fills the input nodes with values and propagates values through the graph.
    ///
    /// # Parameters
    /// - `inputs`: A vector of values to assign to the input nodes.
    ///
    /// Panics if the number of inputs does not match the number of input nodes.
    pub fn fill_nodes(&mut self, inputs: Vec<u32>) {
        if inputs.len() != self.input_nodes_count {
            error!(
                "ERR_INPUT_COUNT_MISMATCH – Number of inputs does not match the number of input nodes"
            );
            panic!(
                "ERR_INPUT_COUNT_MISMATCH – Number of inputs does not match the number of input nodes"
            );
        }
        // Set input values
        let mut input_idx = 0;
        for node in &mut self.nodes {
            if node.is_input {
                node.value = Some(inputs[input_idx]);
                if self.verbose {
                    debug!("SET_INPUT – node {} = {}", input_idx, inputs[input_idx]);
                }
                input_idx += 1;
            }
        }

        // Propagate values through the graph
        let num_nodes = self.nodes.len();
        for i in 0..num_nodes {
            if self.nodes[i].value.is_none() {
                if let Some(computed_value) = self.compute_node_value(i) {
                    self.nodes[i].value = Some(computed_value);
                }
            }
        }

        if self.verbose {
            info!("FILL_NODES – Input nodes filled and values propagated");
            // TODO: Log final graph state
        }
    }

    /// Checks that all the constraints in the graph hold.
    ///
    /// Returns `true` if all constraints are satisfied, otherwise `false`.
    pub fn check_constraints(&self) -> bool {
        for (a, b) in &self.constraints {
            match (self.nodes[*a].value, self.nodes[*b].value) {
                (Some(val_a), Some(val_b)) if val_a != val_b => {
                    if self.verbose {
                        error!(
                            "ERR_FAILED_CONSTRAINT – node {} (value {}) != node {} (value {})",
                            a, val_a, b, val_b
                        );
                    }
                    return false;
                }
                (None, _) | (_, None) => {
                    if self.verbose {
                        error!("ERR_FAILED_CONSTRAINT – uninitialized node values");
                    }
                    return false;
                }
                _ => continue,
            }
        }
        true
    }

    /// Creates a new node with a hint function applied to the value of the source node.
    ///
    /// # Parameters
    /// - `source_node`: The index of the source node.
    /// - `hint_function`: A function that computes the value of the new node based on the source node.
    ///
    /// Returns the index of the newly created node.
    pub fn hint<F>(&mut self, source_node: usize, hint_function: F) -> usize
    where
        F: Fn(u32) -> u32 + 'static,
    {
        let new_node = Box::new(Node {
            value: None,
            operation: Operation::Hint(source_node),
            is_input: false,
        });
        let new_node_idx = self.nodes.len();
        self.nodes.push(new_node);
        self.hints
            .push((new_node_idx, source_node, Box::new(hint_function)));
        if self.verbose {
            info!(
                "SET_HINT – node {} based on node {}",
                new_node_idx, source_node
            );
        }
        new_node_idx
    }

    /// Prints the current state of the graph for debugging purposes.
    pub fn print_graph(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!("Node {}: {:?}", i, node.value);
        }
    }

    /// Retrieves the value of a node by its index.
    ///
    /// # Parameters
    /// - `node_idx`: The index of the node.
    ///
    /// Returns an `Option<u32>` containing the node's value if it exists.
    pub fn get_value(&self, node_idx: usize) -> Option<u32> {
        self.nodes[node_idx].value
    }
}

#[cfg(test)]
/// Test module for the computational graph implementation.
/// Contains tests for:
/// - Basic arithmetic operations
/// - Error conditions (overflow, division by zero)
/// - Complex graph structures using hints
/// - Edge cases (zero values, large numbers)
/// - Constraint validation
mod tests {
    use super::*;

    #[test]
    fn test_x_squared_plus_x_plus_5() {
        // Example 1: f(x) = x^2 + x + 5

        let mut builder = Builder::new(true);
        let x = builder.init();
        let x_squared = builder.mul(x, x);
        let five = builder.constant(5);
        let x_squared_plus_5 = builder.add(x_squared, five);
        let y = builder.add(x_squared_plus_5, x);

        builder.fill_nodes(vec![3]);

        assert_eq!(builder.get_value(y).unwrap(), 17);
        assert!(builder.check_constraints());
    }

    #[test]
    fn a_plus_1_over_8() {
        // Example 2: f(a) = (a+1) / 8
        //
        // function f(a):
        //     b = a + 1
        //     c = b / 8
        //     return c

        let mut builder = Builder::new(true);
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

    #[test]
    fn sqrt_x_plus_7() {
        // Example 3: f(x) = sqrt(x+7)
        //
        // Assume that x+7 is a perfect square (so x = 2 or 9, etc.).

        let mut builder = Builder::new(true);
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

    #[test]
    fn test_zero_and_negative_values() {
        let mut builder = Builder::new(true);
        let x = builder.init();
        let zero = builder.constant(0);
        let negative_five = builder.constant(-5i32 as u32); // Cast to u32 for test

        let add_zero = builder.add(x, zero);
        let add_negative = builder.add(x, negative_five);

        builder.fill_nodes(vec![5]);
        assert_eq!(builder.get_value(add_zero).unwrap(), 5);
        assert_eq!(builder.get_value(add_negative), None); // 5 - 5 = 0
        assert!(builder.check_constraints());
    }

    #[test]
    #[allow(unconditional_panic)]
    #[should_panic(expected = "ERR_DIV_BY_ZERO – Attempt to divide by zero")]
    fn test_division_by_zero() {
        let mut builder = Builder::new(true);
        let _x = builder.init();
        let _zero = builder.constant(0);
        let one = builder.constant(1);

        let div_by_zero = builder.hint(one, |_| 1 / 0); // Intentional division by zero

        builder.fill_nodes(vec![1]);
        assert_eq!(builder.get_value(div_by_zero).unwrap(), 0); // Should panic before this
    }

    #[test]
    fn test_addition_overflow() {
        let mut builder = Builder::new(true);
        let max = builder.constant(u32::MAX);
        let one = builder.constant(1);
        let sum = builder.add(max, one);
        let final_sum = builder.add(sum, one);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(final_sum), None); // Overflow returns None
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_multiple_operations() {
        let mut builder = Builder::new(true);
        let a = builder.init();
        let b = builder.init();
        let c = builder.mul(a, b);
        let d = builder.add(c, a);
        let e = builder.mul(d, b);

        builder.fill_nodes(vec![2, 3]); // a = 2, b = 3
        assert_eq!(builder.get_value(c).unwrap(), 6); // 2 * 3
        assert_eq!(builder.get_value(d).unwrap(), 8); // 6 + 2
        assert_eq!(builder.get_value(e).unwrap(), 24); // 8 * 3
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_unconnected_nodes() {
        let mut builder = Builder::new(true);
        let x = builder.init();
        let _y = builder.init(); // Unconnected node
        let z = builder.constant(10);

        let add = builder.add(x, z);

        builder.fill_nodes(vec![5, 0]); // x = 5, y = 0 (unused)
        assert_eq!(builder.get_value(add).unwrap(), 15); // 5 + 10
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_large_numbers() {
        let mut builder = Builder::new(true);
        let a = builder.constant(u32::MAX);
        let b = builder.constant(1);

        let sum = builder.add(a, b);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(sum), None); // Overflow check
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_all_inputs() {
        let mut builder = Builder::new(true);
        let x = builder.init();
        let y = builder.init();
        let add = builder.add(x, y);

        builder.fill_nodes(vec![3, 4]); // x = 3, y = 4
        assert_eq!(builder.get_value(add).unwrap(), 7); // 3 + 4
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_large_multiplication() {
        // Test multiplication with large numbers to check for overflow handling
        let mut builder = Builder::new(true);
        let large_value = builder.constant(u32::MAX);
        let result = builder.mul(large_value, large_value);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(result), None); // Expected overflow
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_zero_multiplication() {
        // Test multiplication by zero
        let mut builder = Builder::new(true);
        let zero = builder.constant(0);
        let value = builder.constant(12345);
        let result = builder.mul(zero, value);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(result).unwrap(), 0);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_zero_addition() {
        // Test addition with zero
        let mut builder = Builder::new(true);
        let zero = builder.constant(0);
        let value = builder.constant(12345);
        let result = builder.add(zero, value);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(result).unwrap(), 12345);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_negative_values_handling() {
        // Test handling of negative values (cast to u32)
        let mut builder = Builder::new(true);
        let negative_value = builder.constant(-1i32 as u32);
        let value = builder.constant(1);
        let result = builder.add(negative_value, value);

        builder.fill_nodes(vec![]);
        builder.print_graph();
        assert_eq!(builder.get_value(result), None); // -1 + 1 = 0 in signed, but here it should be max value + 1
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_hint_function_edge_case() {
        // Test hint function with a division that could potentially be zero
        let mut builder = Builder::new(true);
        let value = builder.constant(8);
        let hint_node = builder.hint(value, |v| v / 8);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(hint_node).unwrap(), 1);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_multiple_constraints() {
        // Test multiple constraints
        let mut builder = Builder::new(true);
        let x = builder.init();
        let y = builder.constant(10);
        let sum = builder.add(x, y);
        let expected_sum = builder.constant(15);

        builder.assert_equal(sum, expected_sum);
        builder.fill_nodes(vec![5]);

        assert_eq!(builder.get_value(sum).unwrap(), 15);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_uninitialized_node_constraint() {
        // Test constraint on uninitialized nodes
        let mut builder = Builder::new(true);
        let x = builder.init();
        let y = builder.init();
        builder.assert_equal(x, y);

        builder.fill_nodes(vec![10, 10]);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_no_input_nodes() {
        // Test behavior with no input nodes
        let mut builder = Builder::new(true);
        let constant = builder.constant(42);

        builder.fill_nodes(vec![]); // No inputs provided
        assert_eq!(builder.get_value(constant).unwrap(), 42);
        assert!(builder.check_constraints());
    }

    #[test]
    fn test_all_operations_combined() {
        // Test a combination of all operations
        let mut builder = Builder::new(true);
        let x = builder.init();
        let y = builder.constant(5);
        let z = builder.add(x, y);
        let w = builder.mul(z, y);
        let hint = builder.hint(w, |val| val / 2);
        let final_sum = builder.add(hint, y);

        builder.fill_nodes(vec![3]);
        assert_eq!(builder.get_value(final_sum).unwrap(), 25); // ((3 + 5) * 5) / 2 + 5
        assert!(builder.check_constraints());
    }

    #[test]
    #[allow(unconditional_panic)]
    #[should_panic(expected = "ERR_DIV_BY_ZERO – Attempt to divide by zero")]
    fn test_edge_case_division_by_zero_hint() {
        // Test hint function that could cause division by zero
        let mut builder = Builder::new(true);
        let zero = builder.constant(0);
        let hint_node = builder.hint(zero, |_| 1 / 0); // Intentional division by zero

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(hint_node), None); // Division by zero should result in None
    }
}
