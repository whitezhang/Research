# Framework for Big Machine Learning
- Reference paper: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf

The framework is designed as follows:
1. read data and vectorize it as dict (mainly for discreted data)
2. use chimerge to get the intervals for discretization
3. use intervals to discret the continuous data (do hash)
4. do combination for features (do hash)
5. use fm to train the data
