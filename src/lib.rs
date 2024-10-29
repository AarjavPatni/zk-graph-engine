/// A builder that will be used to create a computational graph.
pub struct Builder {
    nodes: Vec<Box<Node>>,
    constraints: Vec<(usize, usize)>,
    hints: Vec<(usize, usize, Box<dyn Fn(u32) -> u32>)>,
    input_nodes_count: usize,
}

enum Operation {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    Hint(usize),
}

/// A node in the computational graph.
struct Node {
    value: Option<u32>,
    operation: Operation,
    is_input: bool,
}

/// Methods for the `Node` struct.
impl Node {
    /// Creates a new node.
    pub fn new(is_input: bool) -> Self {
        Self {
            value: None,
            operation: Operation::None,
            is_input,
        }
    }

    /// Creates a new node with a constant value.
    pub fn new_const(value: u32) -> Self {
        Self {
            value: Some(value),
            operation: Operation::None,
            is_input: false,
        }
    }
}

impl Builder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            constraints: vec![],
            hints: vec![],
            input_nodes_count: 0,
        }
    }

    /// Initializes an input node in the graph.
    pub fn init(&mut self) -> usize {
        let node = Box::new(Node::new(true));
        self.nodes.push(node);
        self.input_nodes_count += 1;
        let idx = self.nodes.len() - 1;
        idx
    }

    /// Initializes a node in a graph, set to a constant value.
    pub fn constant(&mut self, value: u32) -> usize {
        let node = Box::new(Node::new_const(value));
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        idx
    }

    /// Adds 2 nodes in the graph, returning a new node.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let node = Box::new(Node {
            value: None,
            operation: Operation::Add(a, b),
            is_input: false,
        });
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        idx
    }

    /// Multiplies 2 nodes in the graph, returning a new node.
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let node = Box::new(Node {
            value: None,
            operation: Operation::Mul(a, b),
            is_input: false,
        });
        self.nodes.push(node);
        let idx = self.nodes.len() - 1;
        idx
    }

    /// Asserts that 2 nodes are equal.
    pub fn assert_equal(&mut self, a: usize, b: usize) {
        self.constraints.push((a, b));
    }

    /// Helper method to compute the value of a node based on its operation.
    fn compute_node_value(&self, node_idx: usize) -> Option<u32> {
        let node = &self.nodes[node_idx];

        match &node.operation {
            Operation::None => node.value,
            Operation::Add(a, b) => {
                let a_val = self.nodes[*a].value?;
                let b_val = self.nodes[*b].value?;
                if let Some(val) = a_val.checked_add(b_val) {
                    Some(val)
                } else {
                    None
                }
            }
            Operation::Mul(a, b) => {
                let a_val = self.nodes[*a].value?;
                let b_val = self.nodes[*b].value?;
                if let Some(val) = a_val.checked_mul(b_val) {
                    Some(val)
                } else {
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
                                panic!("Error: Attempt to divide by zero");
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

    /// Fills in all the nodes of the graph based on some inputs.
    pub fn fill_nodes(&mut self, inputs: Vec<u32>) {
        if inputs.len() != self.input_nodes_count {
            panic!("Number of inputs does not match the number of input nodes");
        }
        // First, set input values
        let mut input_idx = 0;
        for node in &mut self.nodes {
            if node.is_input {
                node.value = Some(inputs[input_idx]);
                input_idx += 1;
            }
        }

        // Propagate values through the graph
        let num_nodes = self.nodes.len();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..num_nodes {
                if self.nodes[i].value.is_none() {
                    if let Some(computed_value) = self.compute_node_value(i) {
                        self.nodes[i].value = Some(computed_value);
                        changed = true;
                    }
                }
            }
        }
    }

    /// Given a graph that has `fill_nodes` already called on it
    /// checks that all the constraints hold.
    pub fn check_constraints(&self) -> bool {
        for (a, b) in &self.constraints {
            match (self.nodes[*a].value, self.nodes[*b].value) {
                (Some(val_a), Some(val_b)) if val_a != val_b => {
                    return false;
                }
                (None, _) | (_, None) => {
                    return false;
                }
                _ => continue,
            }
        }
        true
    }

    /// An API for hinting values that allows you to perform operations
    /// like division or computing square roots.
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
        new_node_idx
    }

    pub fn print_graph(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!("Node {}: {:?}", i, node.value);
        }
    }

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
        // f(x) = x^2 + x + 5

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

    #[test]
    fn a_plus_1_over_8() {
        // f(a) = (a+1) / 8
        //
        // function f(a):
        //     b = a + 1
        //     c = b / 8
        //     return c

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

    #[test]
    fn sqrt_x_plus_7() {
        // Example 3: f(x) = sqrt(x+7)
        //
        // Assume that x+7 is a perfect square (so x = 2 or 9, etc.).

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

    #[test]
    fn test_zero_and_negative_values() {
        let mut builder = Builder::new();
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
    #[should_panic(expected = "Error: Attempt to divide by zero")]
    fn test_division_by_zero() {
        let mut builder = Builder::new();
        let _x = builder.init();
        let _zero = builder.constant(0);
        let one = builder.constant(1);

        let div_by_zero = builder.hint(one, |_| 1 / 0); // Intentional division by zero

        builder.fill_nodes(vec![1]);
        assert_eq!(builder.get_value(div_by_zero).unwrap(), 0); // Should panic before this
    }

    #[test]
    fn test_addition_overflow() {
        let mut builder = Builder::new();
        let max = builder.constant(u32::MAX);
        let one = builder.constant(1);
        let sum = builder.add(max, one);
        let final_sum = builder.add(sum, one);

        builder.fill_nodes(vec![]);
        assert_eq!(builder.get_value(final_sum), None); // Overflow returns None
        assert!(builder.check_constraints());
    }
}
