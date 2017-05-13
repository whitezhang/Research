# Mode 2
# Evaluate the wifis prediction
cat all.data | python gen_data.py -m 2 -na 2

# Evaluate the wifi distribution
cat all.data | python gen_data.py -m 3

- Do multinomial-distribution

### KDE
Using KDE function to solve problems with fewer data
- can improve? (Baseline: over sampling)

Contribution
1. Based on independent feature construction
2. Based on dependent feature construction
which needs to discuss the KDE function

Limitation of KDE function approaches
1. in which cases, this model can not solve


### Transfer Learning
intance-based transfer
feature representation transfer
paramter transfer
knowledge-based transfer

Thoughts
1. relations between data distribution and features(not strong in poi prediction, since top K features may decide poi) -> MLT approach
2. learn from labeled data distribution. Only use the features that follows the distribution since the model can not learn from the data

TODO
Difference between multi-task lasso regression and rigde regression. Global or local?

gcc -o ap2embnew ap2emb.c -lm -lpthread
