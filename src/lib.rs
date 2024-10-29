//! A computational graph library for zero-knowledge proofs.
//!
//! This crate provides a builder pattern for creating and executing computational graphs
//! with support for arithmetic operations, constraints, and hints.
//!
//! This is designed to be used in zero-knowledge proofs, where we are interested in proving
//! a statement which states that a function f evaluated at inputs (x_1, ..., x_n) results in
//! an output (y_1, ..., y_n). This function is represented as a computational graph, where each
//! node is an integer and relationships between nodes are defined by addition, multiplication,
//! or equality.
//!
//! ## Features
//! - Basic arithmetic operations (addition, multiplication)
//! - Input nodes and constant values
//! - Constraint checking between nodes
//! - Hint system for custom operations
//! - Logging support for debugging
//!
//! ## Example
//! ```rust
//! use zk_graph_engine::Builder;
//!
//! let mut builder = Builder::new();
//! let x = builder.init();  // Create input node
//! let y = builder.constant(5);  // Create constant node
//! let sum = builder.add(x, y);  // Add nodes
//!
//! builder.fill_nodes(vec![3]);  // Set input value
//! assert_eq!(builder.get_value(sum).unwrap(), 8);
//! ```

use log::{debug, error, info};
use std::collections::HashMap;
use std::env;
use std::sync::Once;

/// Type alias for hint functions that take and return u32
type HintFn = Box<dyn Fn(u32) -> u32>;

/// Type alias for hint storage containing source node index and hint function
type HintEntry = (usize, HintFn);

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
pub struct Builder {
    nodes: Vec<Node>,
    constraints: Vec<(usize, usize)>,
    hints: HashMap<usize, HintEntry>,
    input_nodes_count: usize,
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
    pub fn new() -> Self {
        // Initialize the logger once
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            if env::var("RUST_LOG").is_err() {
                env::set_var("RUST_LOG", "off");
            }
            env_logger::init();
        });

        Self {
            nodes: vec![],
            constraints: vec![],
            hints: HashMap::new(),
            input_nodes_count: 0,
        }
    }

    /// Initializes an input node in the graph.
    ///
    /// Returns the index of the newly created input node.
    pub fn init(&mut self) -> usize {
        let node = Node::new();
        self.nodes.push(node);
        self.input_nodes_count += 1;
        let idx = self.nodes.len() - 1;
        info!("INIT_INPUT – node {}", idx);
        idx
    }

    /// Initializes a node in the graph set to a constant value.
    ///
    /// # Parameters
    /// - `value`: The constant value to assign to the node.
    ///
    /// Returns the index of the newly created constant node.
    pub fn constant(&mut self, value: u32) -> usize {
        let node = Node::new_const(value);
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        debug!("INIT_CONST – node {} = {}", idx, value);
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
        let node = Node {
            value: None,
            operation: Operation::Add(a, b),
            is_input: false,
        };
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        info!("SET_ADD_OPT – node {} = node {} + node {}", idx, a, b);
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
        let node = Node {
            value: None,
            operation: Operation::Mul(a, b),
            is_input: false,
        };
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        info!("SET_MUL_OPT – node {} = node {} * node {}", idx, a, b);
        idx
    }

    /// Asserts that two nodes are equal.
    ///
    /// # Parameters
    /// - `a`: The index of the first node.
    /// - `b`: The index of the second node.
    pub fn assert_equal(&mut self, a: usize, b: usize) {
        self.constraints.push((a, b));
        info!("SET_CONSTRAINT – Assert node {} == node {}", a, b);
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

            Operation::Hint(source) => {
                let source_value = self.nodes[*source].value?;

                if let Some((_, hint_fn)) = self.hints.get(&node_idx) {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        hint_fn(source_value)
                    })) {
                        Ok(val) => Some(val),
                        Err(_) => {
                            error!(
                                "ERR_HINT_FAILED – Check hint function for node {}",
                                node_idx
                            );
                            None
                        }
                    }
                } else {
                    error!(
                        "ERR_MISSING_HINT – No hint function found for node {}",
                        node_idx
                    );
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
                debug!("SET_INPUT – node {} = {}", input_idx, inputs[input_idx]);
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

        info!("FILL_NODES – Input nodes filled and values propagated");
        // TODO: Log final graph state
    }

    /// Checks that all the constraints in the graph hold.
    ///
    /// Returns `true` if all constraints are satisfied, otherwise `false`.
    pub fn check_constraints(&self) -> bool {
        for (a, b) in &self.constraints {
            match (self.nodes[*a].value, self.nodes[*b].value) {
                (Some(val_a), Some(val_b)) if val_a != val_b => {
                    error!(
                        "ERR_FAILED_CONSTRAINT – node {} (value {}) != node {} (value {})",
                        a, val_a, b, val_b
                    );
                    return false;
                }
                (None, _) | (_, None) => {
                    error!("ERR_FAILED_CONSTRAINT – uninitialized node values");
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
        let new_node = Node {
            value: None,
            operation: Operation::Hint(source_node),
            is_input: false,
        };
        let new_node_idx = self.nodes.len();
        self.nodes.push(new_node);
        self.hints
            .insert(new_node_idx, (source_node, Box::new(hint_function)));
        info!(
            "SET_HINT – node {} based on node {}",
            new_node_idx, source_node
        );
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
