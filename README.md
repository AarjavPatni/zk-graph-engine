# ZK Graph Engine – Computational Graph Library

This Rust library provides a framework for creating and executing computational graphs with support for basic arithmetic operations, constraints, and hints. It is designed to be robust, efficient, and suitable for production environments.

## Basic Features

- [x]  **Basic Arithmetic Operations**: Supports addition and multiplication of nodes.
- [x]  **Input and Constant Nodes**: Allows initialization of input nodes and constant values.
- [x]  **Constraint Checking**: Enables assertions of equality between nodes to ensure constraints are maintained.
- [x]  **Hint System**: Allows custom operations using hints for values computed outside the graph (e.g., division or square roots).
- [ ]  **Comprehensive Testing**: Includes a suite of tests covering various use cases and edge cases.
- [ ]  **Verbose Logging**: Configurable logging for debugging and monitoring purposes.


## Usage

Will be updated once the library is published.

## Future Improvements

- [ ]  **Graceful Error Handling**: Instead of panicking, the library should return a `Result` type for an improved developer experience.
- [ ]  **Topological Traversal**: Implement a method to traverse the graph in topological order, making the overall computation more efficient.
- [ ]  **Graph Visualization**: Implement a method to visualize the graph using a library like `petgraph` or `dot` for better debugging and monitoring.
