import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sklearn
from sklearn import datasets
import os
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide")

st.title("Decision Trees")

st.markdown(
    r"""
Decision Trees are powerful Machine learning algorithms that can perform both classification and regression tasks which are non-parametric and non-linear. 
The main advantages are that they are easy to visualize which is useful for interpretation. This is called a white box model. Moreover, they are fast
to train and evaluate which is especially beneficial when you are dealing with huge data sets. Additionaly, since they are non-parametric, no
pre-assumptions about the data is needed. However, they do also have a disadvantage. They have a strong tendency to overfit the training data
which leads to poor generalisation. Nonetheless, they are a fundamental component of Ranodm Forests which is among the most powerful ML algorithms
today. Therefore, it worthwhile taking a closer look at decision trees. Let's have a look at the theory. 
"""
)

st.header("Theory")

st.markdown(
    r"""

The construction of a tree includes following three elements:

* 1.) Selection of the division: How do we decide which feature is taken for a division at a given node and how do we find a threshold?
* 2.) How do we decide when a node is a final one or we have to continue with dividing our data? (Still a branch or already a leaf)
* 3.) The leafs must be assigned to a certain class. How do we do that?

There are several things we have to take into consideration. First, we can only as binary questions that can be answered with yes or no. Second, questions always asked regarding one 
feature. Third, for interval-scaled and ordinal features we have to ask: Is $x_i^j  \leq c_j$? For nominal feature one has to find an apppropriate division into two groups
which is quite computationaly heave since $2^{L-1}-1$ possible divisions must be analised. ($L$ are the factor levels)

The optimal question is one that leads to partitions that are as pure as possible (contain only objects of the same class). Therefore, we have to measure the impurity $I(N)$ 
in node $N$ which can be done with the Gini impurity with the following formula:

$I_G(N) = \sum_{k=1}^g (error \, rate \ class \, j) * p(\Omega_j | N) = ... = 1 - \sum_{k=1}^g p^2 (\Omega_k | N)$

i.e. it corresponds to the expected error rate at node N if the class label $\Omega_l$ is randomly selected from the classes present at node $N$.
In the case of only two classes, the Gini index is proportional to the variance of a binomial experiment with success parameter $p ⟨\Omega_2 | N⟩$.
The variance of a Bernoulli experiment with success parameter $p$ is $p (1 − p)$ or $p * q$ if $q = 1 − p$. Since $p^2 + q2 = (p + q)^2 − 2pq$, we have
$2pq = 1 − p^2 − q^2$.

Now choose the question such that the level of contamination is reduced as much as possible, where the level is given by the following expression:
$∆I⟨N⟩ = I⟨N⟩ − a_L * I⟨N_L⟩ − (1 − a_L) * I⟨N_R⟩$
NL and NR designate the left and right subsequent nodes, respectively, and $a_L$ the proportion of those objects at node $N$ that go into the left subsequent node.
The best question at node $N$ is now the one that maximizes $∆I⟨N⟩$.
i.e.
* The best distribution is sought in each feature
* and then one looks for the best split in each feature

Remarks:
* The solution is not necessarily unique
    (If e.g. for all $c_j ∈ (c_j^L, c_j^R )$ the question “$x^{(j)} \leq c_j$” leads to the same improvement, one chooses $c_j = (c_j^L + c_j^R )/2)$
* The optimization is always performed locally, i.e. at each individual node. Therefore, one does not necessarily find the global optimum - is a so-called "greedy search" procedure

With regards to the 2.) and 3.) elements of constructing a tree:

* For the time being, we let the tree grow as long as we find a feature so that $∆I ⟨N⟩ > 0$.
*  Class assignment in an end node: Because not every end node is necessarily pure, it is assigned to the class that has the majority in the end node based on the learning sample.

Let's have a look at some data and an example.
"""
)

st.header("Wine Tree")

# Get data
from sklearn.datasets import load_wine

# Getting data
@st.cache
def load_data():
    data = load_wine(as_frame=True)
    return data


data = load_data()

# End get data

st.markdown(
    r"""
We use the wine dataset from sklearn's library which contains the following attributes:

* Alcohol
* Malic acid
* Ash
* Alcalinity of ash
* Magnesium
* Total phenols
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

which looks as follows:
"""
)

st.dataframe(data.frame)

st.markdown(
    r"""
There are three classes of wine which are stored in the columns *target* and labelled as class_0, class_1 and class_2. The goal is to classify them with a 
decision tree just like we have discussed in the theory section which looks as follows:
"""
)

X = data.frame.loc[:, ["alcohol", "color_intensity"]]
y = data.frame.iloc[:, -1:]

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, y)

from sklearn import tree

fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(
    tree_clf,
    feature_names=["alcohol", "color_intensity"],  # wine.feature_names
    class_names=["class_0", "class_1", "class_2"],  # wine.target_names
    filled=True,
)
st.pyplot(fig)

st.markdown(
    r"""
Holy smokes, that escalated quickly. Remember when I said that decision trees have a strong tendency to overvit? This is what I was talking about. Therefore, 
we have to do some regularisation of the hyperparameters in order to overcome the overfitting. 
"""
)

st.header("Regularisation of Hyperparameters")

st.markdown(
    r"""
There are three techniques for regularise the hyperparameters:

* 1.) Restrict growth by allowing the tree to grow only as long as $∆I ⟨N⟩$ at a node $N$ does not fall below a predefined threshold value $cp$.
    * **Advantages**: 
        * The tree is trained on all the training data
        * Different end nodes can be reached by a different number of intermediate nodes.
    * **Disadvantage**:
        * It is pretty hard to find a perfect threshold $cp$ apriori and globally so that every treee achieves optimal performance
* 2.) The minimum number of objects at which a node is to be further divided or which must at least be at hand in an end node has to be specified.
    * For example 10 objects or 5% of all the objects.
* 3.) Pruning of the trees: Deep trees have two main problems: They overfit and are complex. The complexity of trees is measured by the amount of
    end notes (leafs). Therefore, one can stop the growth of the tree after it has reached a certain depth. However, if we stop the tree too early,
    we might miss something because we do not know what would have happened if we had let the tree grow deeper which is called the horizon effect. To avoid this problem,
    we let the tree grow without any restrictions and then the unnecessary branches are cut back which is called **Pruning**. 

However, there are several ways how one can prune trees which are discussed in the following section:
"""
)

st.subheader("Pruning")
st.subheader("Cost-complexity pruning")

st.markdown(
    r"""
The established method to prune a classification tree is the so-called "cost-complexity pruning". 
The cost-complexity measure R ⟨β⟩ weighs the accuracy against the complexity of a tree where the complexity of a tree looks as follows:

$Cost-complexity \, R ⟨β⟩ = error \, rate + β - number \, of  \, terminal  \,nodes$

where β (= "penalty" per additional end node) is called the complexity parameter.


The cost complexity has the following properties:
- When β approaches zero, the optimal trees with respect to the cost complexity become larger.
- If β = 0, then the cost complexity  takes its minimum at the largest possible tree.
- On the other hand, if β grows and is large (about infinite), then a tree with only one end node (the tree root) has the lowest cost complexity.
- The optimal tree (i.e. the tree that is not too complex but still has a low error rate) will lie between the above extremes.
"""
)

st.subheader("Bottom-up pruning")

st.markdown(
    r"""
The pruning starts from the end nodes ("bottom up").
- Prune from the end nodes as long as the in-sample error rate remains the same.
- Then search for the so called weakest connection,
    i.e. the node at which cutting the underlying tree increases the in-sample error rate the least, so that the smallest increase in the penalty parameter $β$
      is necessary for the cost complexity to be the same with and without the underlying tree.
- Cut off the subtree of the weakest connection.
- Repeat the search for the weakest connection in the truncated tree iteratively until only the tree root is left.
    The complexity parameter $β$ increases and the number of nodes in the tree decreases.
- Then determine the tree with the optimal prediction accuracy from this series of subtrees with the help of cross-validation or test data. 
    A series of increasing β-values $(β_k , k = 1, 2, . . .)$ arises, which result in ever simpler trees.


The out-of-sample error rate of these thumbs contained in each other is now to be determined with a 10-fold cross-validation.
For this purpose
- First, maximum trees are generated on the basis of 9/10 of the learning sample in each case,
- Then these trees are pruned with the same complexity parameters βk as above in the total learning sample, and
- The cross-validated error rate is determined for all these trees. Note that the number of terminal nodes in such trees need not be equal, but only the $β_k$ .
    Caution: The results change with each new run because of the random sampling in the cross-validation.
    Therefore, a random seed is set for the random number generator at the beginning of the cross-validation in order to obtain reproducable results.
"""
)

st.subheader("Pruned tree")

st.markdown(
    r"""
A simple trick as constraining the tree to grow to a maximum depth of 3 already results in the following tree:
"""
)

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X, y)

from sklearn import tree

fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(
    tree_clf,
    feature_names=["alcohol", "color_intensity"],  # wine.feature_names
    class_names=["class_0", "class_1", "class_2"],  # wine.target_names
    filled=True,
)
st.pyplot(fig)

st.markdown(
    r"""
The `DecisionTreeClassifier` class has a few other parameters that similarly restrict the shape of the Decision Tree:

* `max_depth`: The maximum depth of the tree.
* `min_samples_split`: The minimum number of samples required to split an internal node.
* `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
* `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.
* `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion.
* `max_features`: The number of features to consider when looking for the best split.
"""
)

st.header("Instability")

st.header("Regression")

# https://scikit-learn.org/stable/modules/tree.html


# if st.button("Time for Party!"):
#     st.balloons()


# Useful information:
# Normally, the data must be numerical (not with DT). However, one can onehotencode the data as follows:

# one_hot_data = pd.get_dummies(dat[['Outlook', 'Temperature', 'Humidity', 'Windy']],drop_first=True)
# tree_clf.fit(one_hot_data, dat['Play'])

# And then you can use it as well. I learned also something interesting. Nested list with 3 elements

# targets = [{'no': 0, 'yes': 1}.get(i, 'none') for i in list(dat['Play'])]
# dat["Play"] = targets
