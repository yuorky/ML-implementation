import math
from typing import List
from anytree import Node, RenderTree
from collections import Counter

import dt_global
from dt_provided import *
from collections import defaultdict


def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values
    :rtype: List[float]
    """

    split_points = list()

    feature_idx = get_feature_index(feature)

    feat_label_dict = defaultdict(set)
    for row in examples:
        feat_label_dict[row[feature_idx]].add(row[-1])

    keys = sorted(feat_label_dict.keys())
    for i in range(len(keys)-1):
        if feat_label_dict[keys[i]] != feat_label_dict[keys[i+1]] or len(feat_label_dict[keys[i]]) + len(feat_label_dict[keys[i+1]]) > 2:
            value = (keys[i] + keys[i+1])/2
            split_points.append(value)

    return split_points


def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """
    gain_max_global = float('-inf')
    best_feature_idx = None
    best_split_global = None
    for feature in features:
        split_points = get_splits(examples, feature)
        gain_max_feat = float('-inf')
        best_split_feat = None
        if len(split_points) == 0:
            continue
        for point in split_points:
            gain = get_information_gain(examples, feature, point)
            if gain > gain_max_feat:
                gain_max_feat = gain
                best_split_feat = point
            elif math.isclose(gain, gain_max_feat, abs_tol=1e-5):
                best_split_feat = min(best_split_feat, point)
        if gain_max_feat > gain_max_global:
            gain_max_global = gain_max_feat
            best_feature_idx = get_feature_index(feature)
            best_split_global = best_split_feat

        elif math.isclose(gain_max_feat, gain_max_global, abs_tol=1e-5):
            if gain_max_feat == float('-inf'):
                pass

            elif get_feature_index(feature) < best_feature_idx:
                best_feature_idx = get_feature_index(feature)
                best_split_global = best_split_feat
            elif get_feature_index(feature) == best_feature_idx:
                best_split_global = min(best_split_global, best_split_feat)

    if best_feature_idx is None:
        return None, -1

    return dt_global.feature_names[best_feature_idx], best_split_global


def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """
    list_1 = list()
    list_2 = list()
    feature_idx = get_feature_index(feature)

    for i in range(len(examples)):
        if examples[i][feature_idx] <= split:
            list_1.append(examples[i])
        else:
            list_2.append(examples[i])

    return list_1, list_2


def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """

    # if no examples left
    # return
    # cur_node has been set to a leaf node before
    if len(examples) == 0:
        return

    label_col = [row[-1] for row in examples]
    cur_node.majority = get_majority(label_col)

    # if all examples are in the same class
    if label_col.count(label_col[0]) == len(label_col):
        cur_node.decision = label_col[0]
        return

    # if no features left in features
    if len(features) == 0:
        cur_node.decision = get_majority(label_col)
        return

    # if reach maximum depth
    if cur_node.depth >= max_depth:
        cur_node.decision = get_majority(label_col)
        return

    feature, split_value = choose_feature_split(examples, features)
    # if feature is None
    if feature is None:
        cur_node.decision = get_majority(label_col)
        return
    else:
        cur_node.feature = feature
        cur_node.split = split_value
        child1, child2 = split_examples(examples, feature, split_value)

    if child1:
        cur_node1 = Node("left", parent=cur_node, nums=len(child1))
    else:
        # child no example
        # child is a leaf node with majority decision at parent node
        #cur_node1 = Node("left", parent=cur_node, decision=get_majority(label_col), nums=0, \
        #                 majority=get_majority(label_col))
        cur_node.decision = get_majority(label_col)
        return

    if child2:
        cur_node2 = Node("right", parent=cur_node, nums=len(child2))
    else:
        # cur_node2 = Node("right", parent=cur_node, decision=get_majority(label_col), nums=0,\
        #                  majority=get_majority(label_col))
        cur_node.decision = get_majority(label_col)
        return

    split_node(cur_node=cur_node1, examples=child1, features=features, max_depth=max_depth)
    split_node(cur_node=cur_node2, examples=child2, features=features, max_depth=max_depth)


def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """

    # Construct cur_node
    cur_node = Node('root', parent=None, nums=len(examples))

    # nothing in examples provided
    if len(examples) == 0:
        return cur_node

    # feed into split_node, cur_node gets update
    split_node(cur_node=cur_node, examples=examples, features=features, max_depth=max_depth)

    return cur_node


def predict(cur_node: Node, example, max_depth=math.inf, min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than min_num_examples, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """
    # if we come to the maximum depth
    # return majority decision at this node
    if cur_node.depth >= max_depth:
        return cur_node.majority

    # if the number of examples at the node is less than min_num_examples
    # return majority decision at this node
    elif cur_node.nums < min_num_examples:
        return cur_node.majority

    # if it is a leaf node, return decision
    elif cur_node.is_leaf:
        return cur_node.decision

    # non-leaf node has to have these two attr
    feature = cur_node.feature
    split_value = cur_node.split
    feature_idx = get_feature_index(feature)

    if example[feature_idx] <= split_value:
        child1 = cur_node.children[0]
        return predict(cur_node=child1, example=example, max_depth=max_depth, min_num_examples=min_num_examples)
    else:
        child2 = cur_node.children[1]
        return predict(cur_node=child2, example=example, max_depth=max_depth, min_num_examples=min_num_examples)


def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """

    acc_num = 0
    for example in examples:
        pred = predict(cur_node=cur_node, example=example, max_depth=max_depth, min_num_examples=min_num_examples)
        label = example[-1]
        if pred == label:
            acc_num += 1

    acc = acc_num/len(examples)
    return acc


def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    if cur_node.is_leaf:
        return
    else:
        post_prune(cur_node=cur_node.children[0], min_num_examples=min_num_examples)
        post_prune(cur_node=cur_node.children[1], min_num_examples=min_num_examples)

        if cur_node.children[0].is_leaf and cur_node.children[1].is_leaf and cur_node.nums < min_num_examples:
            cur_node.decision = cur_node.majority
            delattr(cur_node, 'children')


def get_feature_index(feature_name):
    """

    :param feature_name: str
    :return: idx: int
    """
    return dt_global.feature_names.index(feature_name)


# calculate entropy function
def get_entropy(examples):
    label_data = [row[-1] for row in examples]
    labels, label_counts = np.unique(label_data, return_counts=True)
    entropy_value = np.sum([(-label_counts[i]/len(label_data))*np.log2(label_counts[i]/len(label_data))
                            for i in range((len(labels)))])
    return entropy_value


def get_information_gain(examples, feature, split_point):
    I_before = get_entropy(examples)

    exa_1, exa_2 = split_examples(examples,feature,split_point)

    P1 = len(exa_1)/(len(exa_1) + len(exa_2))
    P2 = len(exa_2)/(len(exa_1) + len(exa_2))

    I_after = P1*get_entropy(exa_1) + P2*get_entropy(exa_2)

    info_gain = I_before - I_after
    return info_gain


def get_majority(labels):
    """

    :param labels: List
    :return: the majority decision: int
    """
    freqDict = Counter(labels)
    freq = 0
    vote = None
    for key, value in freqDict.items():
        if value > freq:
            freq = value
            vote = key
        elif value == freq:
            if key < vote:
                vote = key
    return vote


def get_depth(cur_node):
    if cur_node.is_leaf:
        return cur_node.depth
    else:
        return max(get_depth(cur_node.children[0]), get_depth(cur_node.children[1]))


# aa = read_data('data.csv')
# a_folds = preprocess(aa)  # 3-D list
# # a_folds[0] -> example List[List[any]]
#
# exa = a_folds[0]
#
# feature_col1 = [row[0] for row in exa]
# label_col1 = [row[-1] for row in exa]
# feature_col1, label_col1 = zip(*sorted(zip(feature_col1, label_col1)))
#
# split_points = get_splits(exa, 'mcg')
#
# split_points2 = get_splits(exa, 'pox')
#
#
# # test get get_information_gain
# test1 = exa[0:10]
#
# test1_1, test1_2 = split_examples(test1, 'mcg', 0.5)
#
# test1_gain = get_information_gain(test1, 'mcg', 0.5)
#
#
#
# # test choose feature_split
# choose_feature_split(test1, ['mcg'])
#
# # Node
# root = Node("root")
# root1 = Node("root1", parent=root, feature="mcg", decision=0.5)
#
# root2 = Node("root2", parent=root1, feature='mcg', decision=0.5)
#
# # test split_node
#
# features = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
# feature, split_value = choose_feature_split(exa, features)
# cur_node = Node('root', parent=None)
# split_node(cur_node=cur_node, examples=exa, features=features)
#
# # test learn_dt
# final_node = learn_dt(examples=aa, features=features)
# final_node2 = learn_dt(examples=exa, features=[])
#
# test predict
# t1_node = Node('root', parent=None, feature='mcg', split=1.5, nums=5, majority=0)
# t2_node = Node('left', parent=t1_node, decision=0, nums=2, majority=0)
# t3_node = Node('right', parent=t1_node, feature='gvh', split=0.5, nums=3, majority=0)
# t4_node = Node('left', parent=t3_node, decision=1, nums=2, majority=1)
# t5_node = Node('right', parent=t3_node, decision=0, nums=1, majority=1)
#
# texa1 = [0.63, 0.56, 0.52, 0.21, 0.5, 0.0, 0.5, 0.22, 0]
# predict1 = predict(cur_node=t1_node, example=texa1) # None
#
# predict2 = predict(cur_node=final_node, example=texa1)
#
#

# test post_prune

# post_prune(cur_node=t1_node, min_num_examples=4)
# #
# m = 0
# for i in final_node.leaves:
#     if i.depth > m:
#         m = i.depth
#
# depth_4_node = learn_dt(examples=aa, features=features, max_depth=4)
# print(RenderTree(depth_4_node))