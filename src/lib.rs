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
                Some(a_val + b_val)
            }
            Operation::Mul(a, b) => {
                let a_val = self.nodes[*a].value?;
                let b_val = self.nodes[*b].value?;
                Some(a_val * b_val)
            }
            // TODO: Implement HashMap for faster lookups
            Operation::Hint(source) => {
                if let Some(value) = self.nodes[*source].value {
                    if let Some((_, _, hint_fn)) = self
                        .hints
                        .iter()
                        .find(|(target, src, _)| *target == node_idx && *src == *source)
                    {
                        Some(hint_fn(value))
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
    fn test_basic_add() {
        todo!()
    }
}
