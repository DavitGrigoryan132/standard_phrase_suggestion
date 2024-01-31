# Text Standardization Tool

This tool provides suggestions for standardizing text based on pre-defined phrases and embeddings. It uses a combination of natural language processing and cosine similarity to suggest changes in your text.

## Installation
```bash
git clone https://github.com/your_username/text-standardization.git
cd text-standardization
pip install -r requirements.txt
```

## Usage

Run the main script standardize_text.py to suggest changes in your text:
```bash
python standardize_text.py --input input_files/sample_text.txt --terms_csv input_files/Standardised_terms.csv
```

- --input: Path to the input text file or the input text itself.
- --terms_csv: Path to the CSV file containing standard terms.

The script reads the input text, identifies phrases to be standardized, suggests replacements, and prompts the user to accept or reject each suggestion.

## Scripts and Modules

- `standard.py`: Contains the StandardisedPhrases class for handling standardized phrases and providing suggestions based on embeddings.
- `standardize_text.py`: Main script for suggesting changes in the input text using standardized phrases.
- `embeddings.py`: Contains the EmbeddingsModel class for generating sentence embeddings using pre-trained transformer models.
