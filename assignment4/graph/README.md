# PySpark PageRank Tutorial

A comprehensive Jupyter notebook teaching PageRank algorithm using three different implementations:

1. Manual implementation from scratch
2. NetworkX built-in function
3. PySpark distributed implementation

## What You'll Learn

- PageRank mathematical formula and intuition
- Step-by-step manual implementation
- Using NetworkX for small-to-medium graphs
- Scaling to large graphs with PySpark
- Performance comparison across methods

## Requirements

- Python 3.8+
- NetworkX
- PySpark
- Matplotlib
- NumPy

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install networkx pyspark matplotlib numpy jupyter
```

## Usage

```bash
source venv/bin/activate
jupyter notebook PageRank_Tutorial.ipynb
```

## Performance Notes

- Manual/NetworkX: Best for graphs <100K nodes
- PySpark: Best for graphs >100K nodes due to distributed computing overhead

## License

MIT