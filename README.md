# NameMatching

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

AI-based name comparison system for accurate name matching and similarity scoring.

## Overview

NameMatching is a Python library that uses artificial intelligence to compare and match person names. It combines deep learning with handcrafted features to provide accurate similarity scores between name pairs, handling various naming conventions and abbreviations.

## Features

- **AI-Powered Matching**: Uses a trained neural network with BiLSTM architecture
- **Character-Level Processing**: Handles typos, abbreviations, and variations
- **Rich Feature Set**: Combines multiple similarity metrics (Jaro-Winkler, Levenshtein, etc.)
- **Handles Complex Cases**: 
  - Initials vs full names ("J. Smith" ↔ "John Smith")
  - Name order variations
  - Nickname matching
  - Suffix handling (Jr., Sr., III, etc.)
- **Easy Integration**: Simple API for quick integration into existing systems

## Installation

```bash
pip install namematching
```

Or install from source:

```bash
git clone https://github.com/Seizh7/NAMEMATCHING.git
cd NAMEMATCHING
pip install -e .
```

## Quick Start

```python
from namematching import compare_names

# Simple comparison
similarity = compare_names("John Smith", "J. Smith")
print(f"Similarity: {similarity:.2f}")  # Output: Similarity: 0.95

# Advanced usage with NameMatcher class
from namematching import NameMatcher

matcher = NameMatcher()
score = matcher.similarity("Barack Obama", "Barack H. Obama")
is_match = matcher.is_match("Joe Biden", "Joseph Biden", threshold=0.8)

# Batch processing
pairs = [
    ("Donald Trump", "Donald J. Trump"),
    ("Hillary Clinton", "Hillary Rodham Clinton"),
    ("Bernie Sanders", "Bernard Sanders")
]
scores = matcher.batch_similarity(pairs)
```

## How It Works

The system combines multiple approaches:

1. **Character-Level Neural Network**: BiLSTM processes character sequences
2. **Handcrafted Features**: String similarity metrics and linguistic rules
3. **Smart Penalties**: Reduces false positives for common surnames
4. **Contextual Understanding**: Learns patterns from training data

## Model Performance

The model has been trained on a diverse dataset of names and aliases, achieving:
- High accuracy on exact matches and common variations
- Handling of abbreviations and initials
- Effective disambiguation of common surnames

## Use Cases

- **Data Deduplication**: Identify duplicate records in databases
- **Record Linkage**: Match records across different data sources
- **Search Enhancement**: Improve name-based search functionality
- **Data Quality**: Clean and standardize name data

## Project Structure

```
namematching/
├── namematching/           # Main package
│   ├── matcher.py         # Core matching functionality
│   ├── resources.py       # Resource management
│   ├── utils.py          # Utility functions
│   ├── data/             # Pre-trained models and tokenizers
│   ├── model/            # Model building and training
│   ├── metrics/          # Feature extraction
│   ├── databuild/        # Data generation
│   └── scraping/         # Data collection tools
├── tests/                # Test suite
├── data/                 # Training data (not included)
└── export/              # Model exports
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- pandas
- numpy
- scikit-learn
- textdistance

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
