# NLP

---
## Kaggle progress

[Data fields](https://www.kaggle.com/c/word2vec-nlp-tutorial)
- id - Unique ID of each review
- sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
- review - Text of the review

### Bag-of-Words with Naive Bayes Model
#### Results
Control the number of words selected as the features
- most 1000 frequent words -> 0.83064
- most 5000 frequent words -> 0.83832

#### Evaluation
- Lose of context information(No relations between words)
- Language is ambiguous
The most of the part of each review is talking about the idea of the user, which leads into the phenomenon of ambiguity. For example, `affect` has 76.4% probability to be appeared in the positive sentiment, which does not make any sense.

#### TODO
Conclusion usually appears in the first sentence and last sentence(remove the noise data from the users' discussion). So the underlying ways to improve the performance may be using these sentences to generate a more appropriate words distribution.