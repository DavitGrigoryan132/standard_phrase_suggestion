import os
import re
import argparse
from standard import StandardisedPhrases


def highlight_suggestion(original_text, original_term, suggestion, score):
    start_index = original_text.find(original_term)
    caret_spaces = ' ' * start_index

    print(original_text)
    print(caret_spaces + '^ ' + suggestion)
    print(caret_spaces + str(score))
    print()


def apply_suggestions(input_text, suggestions, ignore_indexes):
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

    args = parser.parse_args()

    standard = StandardisedPhrases()
    standard.read_phrases(args.terms_csv)

    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            text = f.read().strip()
    else:
        text = args.input

    suggestions = standard.give_standardised_suggestions(text)
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

        highlight_suggestion(substring, suggestion[0], suggestion[1], suggestion[2].numpy())

        answer = input("Do you want to apply suggestion? (y/n) ")
        if answer != "y":
            ignore_suggestions_index.append(i)

    result = apply_suggestions(text, suggestions, ignore_suggestions_index)
    print(result)
