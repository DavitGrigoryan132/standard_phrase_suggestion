import os
import re
import argparse
from standard import StandardisedPhrases
from typing import List, Tuple


def highlight_suggestion(original_text: str, original_term: str, suggestion: str, score: float) -> None:
    """
    Print the original text with the suggested replacement highlighted and its corresponding score.

    Parameters:
        original_text (str): The original text.
        original_term (str): The term in the original text being replaced.
        suggestion (str): The suggested replacement.
        score (float): The similarity score for the suggestion.
    """
    start_index = original_text.find(original_term)
    caret_spaces = ' ' * start_index

    print(original_text)
    print(caret_spaces + '^ ' + suggestion)
    print(caret_spaces + str(score))
    print()


def apply_suggestions(input_text: str, suggestions: List[Tuple[str, str, float]], ignore_indexes: List[int]) -> str:
    """
    Apply the accepted suggestions to the input text.

    Parameters:
        input_text (str): The input text.
        suggestions (List[Tuple[str, str, float]]): List of suggestions, each containing input phrase,
                                                    suggested phrase, and score.
        ignore_indexes (List[int]): List of indexes of suggestions to be ignored.

    Returns:
        str: The text with applied suggestions.
    """
    output = input_text
    for i, suggestion in enumerate(suggestions):
        if i not in ignore_indexes:
            word, replacement, score = suggestion
            pattern = re.compile(r'\b' + re.escape(word) + r'\b')
            output = pattern.sub(f'{replacement}', input_text)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This program will suggest changes in your text to use standardized phrases")

    parser.add_argument("--input", type=str, required=True,
                        help='Input text for suggestions or path to the file with text')
    parser.add_argument("--terms_csv", type=str, default="input_files/Standardised terms.csv",
                        help="Path to the csv file which will contain standard terms")
    parser.add_argument("--threshold", type=float, default=0.45, help="Choose threshold for suggestions score")

    args = parser.parse_args()

    # Initialize StandardisedPhrases class
    standard = StandardisedPhrases()

    # Read standardized phrases from the CSV file
    standard.read_phrases(args.terms_csv)

    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            text = f.read().strip()
    else:
        text = args.input

    # Get suggestions for standardized phrases
    suggestions = standard.give_standardised_suggestions(text, threshold=args.threshold)
    ignore_suggestions_index = []

    for i, suggestion in enumerate(suggestions):
        start_index = text.find(suggestion[0])
        end_index = start_index + len(suggestion[0])

        start_index = max(0, start_index - 10)
        end_index = min(len(text), end_index + 10)

        substring = text[start_index: end_index]
        if start_index != 0:
            substring = "..." + substring
        if end_index != len(text):
            substring += "..."

        # Highlight the suggestion and ask user for acceptance
        highlight_suggestion(substring, suggestion[0], suggestion[1], suggestion[2])

        answer = input("Do you want to apply suggestion? (y/n) ")
        if answer != "y":
            ignore_suggestions_index.append(i)

    # Apply accepted suggestions to the text
    result = apply_suggestions(text, suggestions, ignore_suggestions_index)
    print(result)
