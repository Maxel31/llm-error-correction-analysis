# LLM Error Correction Analysis

This project investigates error correction in language models (LLMs) by analyzing activation differences when feeding in pairs of sentences that differ by only one token. The hypothesis is that if certain dimensions (or layers) consistently show small activation changes across multiple different sentence pairs, those dimensions (or layers) might be contributing to error correction.

## Project Overview

The project consists of three main components:

1. **Dataset Generation**: Creating pairs of sentences that differ by only one token using the ChatGPT API.
2. **Activation Analysis**: Extracting and comparing activations from the Llama-3-7B model for each sentence pair.
3. **Results Storage and Analysis**: Storing and analyzing the activation differences to identify patterns.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-error-correction.git
cd llm-error-correction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

2. Generate the dataset of sentence pairs:
```bash
python -m src.data_generation.generate_dataset
```

3. Analyze activations and store results:
```bash
python -m src.model_analysis.analyze_activations
```

4. Visualize the results:
```bash
python -m src.visualization.visualize_results
```

## Project Structure

```
llm-error-correction/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # Directory for storing generated datasets
├── src/
│   ├── __init__.py
│   ├── data_generation/   # Scripts for generating sentence pairs
│   │   ├── __init__.py
│   │   └── generate_dataset.py
│   ├── model_analysis/    # Scripts for analyzing model activations
│   │   ├── __init__.py
│   │   └── analyze_activations.py
│   └── visualization/     # Scripts for visualizing results
│       ├── __init__.py
│       └── visualize_results.py
```

## Research Methodology

The research methodology involves:

1. **Creating Sentence Pairs**: Generating pairs of sentences that differ by only one token, categorized by type (e.g., meaning differences, grammatical errors).
2. **Extracting Activations**: Inputting these sentence pairs into the Llama-3-7B model and extracting activations for each layer and dimension.
3. **Analyzing Differences**: Identifying dimensions where the activation difference is minimal, which might indicate error correction behavior.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
