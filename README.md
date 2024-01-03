# ID3

The ID3 (Iterative Dichotomiser 3) algorithm is a method used in machine learning to construct decision trees. It is employed for data classification based on their attributes.

The fundamental concept behind ID3 involves selecting the best attribute to split the data into the most homogeneous subgroups (the most pure classes). This process iterates recursively until all data is classified or specific stopping conditions are met.

The selection of the best attribute is determined by using a measure of informativeness, such as entropy or information gain, which aids in determining the attribute that best separates the data.

ID3 has certain limitations, such as a tendency to overfit with a high number of attributes or an inability to handle missing values.

Nevertheless, ID3 serves as a foundational algorithm that paved the way for more advanced decision tree construction methods like C4.5 and CART.

**[CODE](main.py)**

## TREE
## accuracy: 78,57%
![](https://github.com/gabrpavel/ID3/blob/main/Tree.jpg)
