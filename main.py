from standard import StandardisedPhrases


standard = StandardisedPhrases()
standard.read_phrases("input_files/Standardised terms.csv")

text = "In today's meeting, we discussed a variety of issues affecting our department. The weather was unusually " \
       "sunny, a pleasant backdrop to our serious discussions. We came to the consensus that we need to do better " \
       "in terms of performance. Sally brought doughnuts, which lightened the mood. It's important to make good use " \
       "of what we have at our disposal. During the coffee break, we talked about the upcoming company picnic. We " \
       "should aim to be more efficient and look for ways to be more creative in our daily tasks. Growth is essential" \
       " for our future, but equally important is building strong relationships with our team members. As a reminder," \
       " the annual staff survey is due next Friday. Lastly, we agreed that we must take time to look over our plans" \
       " carefully and consider all angles before moving forward. On a side note, David mentioned that his cat is" \
       " recovering well from surgery."

standard.give_standardised_suggestions(text)
