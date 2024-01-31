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
- --threshold: Threshold for suggestions score

The script reads the input text, identifies phrases to be standardized, suggests replacements, and prompts the user to accept or reject each suggestion.

## Scripts and Modules

- `standard.py`: Contains the StandardisedPhrases class for handling standardized phrases and providing suggestions based on embeddings.
- `standardize_text.py`: Main script for suggesting changes in the input text using standardized phrases.
- `embeddings.py`: Contains the EmbeddingsModel class for generating sentence embeddings using pre-trained transformer models.

## Example 

Files for example are in input_files directory

```bash
python standardize_text.py --input "input_files/sample_text.txt" --terms_csv "input_files/Standardised terms.csv"
```

After running this command you can choose interactively which suggestions you want to apply

```text
...better in terms of performance. Sally bro...
             ^ Monitor performance metrics
             0.5081683

Do you want to apply suggestion? (y/n) y
...ortant to make good use of what w...
             ^ Utilise resources
             0.45006603

Do you want to apply suggestion? (y/n) n
...ld aim to be more efficient and look ...
             ^ Enhance productivity
             0.59330505

Do you want to apply suggestion? (y/n) y
...ly tasks. Growth is essential for our f...
             ^ Drive growth
             0.5458478

Do you want to apply suggestion? (y/n) y
... over our plans carefully and consider ...
             ^ Execute strategies
             0.47909778

Do you want to apply suggestion? (y/n) n
In today's meeting, we discussed a variety of issues affecting our department. The weather was unusually sunny, a pleasant backdrop to our serious discussions. We came to the consensus that we need to do better in terms of performance. Sally brought doughnuts, which lightened the mood. It's important to make good use of what we have at our disposal. During the coffee break, we talked about the upcoming company picnic. We should aim to be more efficient and look for ways to be more creative in our daily tasks. Drive growth for our future, but equally important is building strong relationships with our team members. As a reminder, the annual staff survey is due next Friday. Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward. On a side note, David mentioned that his cat is recovering well from surgery.
```
