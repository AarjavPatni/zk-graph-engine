# ZK Graph Engine

This Rust library provides a framework for creating and executing computational graphs with support for basic arithmetic operations, constraints, and hints. It is designed to be robust, efficient, and suitable for production environments.


## Installation

Once published, you should be able to install the crate using the following command:

```bash
cargo add zk-graph-engine
```


## Basic Features

- [x]  **Basic Arithmetic Operations**: Supports addition and multiplication of nodes.
- [x]  **Input and Constant Nodes**: Allows initialization of input nodes and constant values.
- [x]  **Constraint Checking**: Enables assertions of equality between nodes to ensure constraints are maintained.
- [x]  **Hint System**: Allows custom operations using hints for values computed outside the graph (e.g., division or square roots).
- [x]  **Comprehensive Testing**: Includes a suite of tests covering various use cases and edge cases.
- [x]  **Verbose Logging**: Configurable logging for debugging and monitoring purposes.


## Assumptions

This implementation assumes the following design decisions:

- Nodes can only hold values of type `u32`.
- Addition and multiplication overflow errors shouldn't panic but return `None` instead, along with an error log message.
- Invalid hint functions (e.g., division by zero) shouldn't panic but return `None` instead, along with a clear error message.


## Local Testing

The library includes a comprehensive suite of tests that cover various use cases and edge cases.

To run all tests:

```bash
cargo test
```

To run tests with verbose logging:

```bash
RUST_LOG=info cargo test
```

You can change the logging level by setting the `RUST_LOG` environment variable to the desired level in the order of `off`, `error`, `warn`, `info`, `debug`.

Since the tests are run in parallel, the logging output may be interleaved. To avoid this, you can also run the tests sequentially:

```bash
cargo test -- --test-threads=1
```


## Usage

Here is a simple example of creating and executing a computational graph:

```rust
use zk_graph_engine::Builder;

let mut builder = Builder::new();
let x = builder.init();
let one = builder.constant(1);
let y = builder.add(x, one);
builder.fill_nodes(vec![3]);
assert_eq!(builder.get_value(y).unwrap(), 4);
```

You can also add constraints to ensure the values of two nodes are equal:

```rust
let x = builder.init();
let y = builder.init();
builder.add_constraint(x, y);
builder.fill_nodes(vec![3, 4]);
assert_eq!(builder.get_value(x).unwrap(), 4);
assert_eq!(builder.get_value(y).unwrap(), 3);
```

You can also use hints to compute the value of a node based on the value of another node:

```rust
let x = builder.init();
let hint_func = |x| x * x;
let y = builder.hint(x, hint_func);
builder.fill_nodes(vec![3]);
assert_eq!(builder.get_value(y).unwrap(), 9);
```

## Future Improvements

- [ ] **Hint using multiple nodes**: Allow hints to use multiple nodes directly to compute the value.
- [ ] **Topological Traversal**: Implement a method to traverse the graph in topological order, making the overall computation more efficient.
- [ ] **Graph Visualization**: Implement a method to visualize the graph using a library like `petgraph` or `dot` for better debugging and monitoring.
