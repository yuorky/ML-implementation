from typing import List

import dt_global
from dt_core import *
from dt_provided import *


def prepare_full_tree(folds:List):
    """

    :param folds: folds for cross validation
    :return fulltree_list: 10 full tree in the list for prediction
    :return train_list: 10 train set train_list for prediction
    """
    train_list_all, full_tree_list = list(), list()
    for i in range(len(folds)):
        train_fold = folds[:i] + folds[i+1:]
        train_set = list()
        for fold in train_fold:
            for elem in fold:
                train_set.append(elem)
        train_list_all.append(train_set)
        full_tree_list.append(learn_dt(examples=train_set, features=dt_global.feature_names[:-1]))

    return train_list_all, full_tree_list


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values (maximum depth)
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """

    train_list_all, full_tree_list = prepare_full_tree(folds)
    train_acc_global_list, val_acc_global_list = list(), list()

    for value in value_list:
        train_acc_param = list()
        val_acc_param = list()

        for j in range(len(folds)):
            train_acc = get_prediction_accuracy(cur_node=full_tree_list[j], examples=train_list_all[j], max_depth=value)
            val_acc = get_prediction_accuracy(cur_node=full_tree_list[j], examples=folds[j], max_depth=value)

            train_acc_param.append(train_acc)
            val_acc_param.append(val_acc)

        train_acc_global = sum(train_acc_param)/len(train_acc_param)
        val_acc_global = sum(val_acc_param)/len(val_acc_param)
        train_acc_global_list.append(train_acc_global)
        val_acc_global_list.append(val_acc_global)

    return train_acc_global_list, val_acc_global_list


def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation (minimum number of examples)
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """

    train_list_all, full_tree_list = prepare_full_tree(folds)
    train_acc_global_list, val_acc_global_list = list(), list()

    for value in value_list:
        train_acc_param = list()
        val_acc_param = list()

        for j in range(len(folds)):
            train_acc = get_prediction_accuracy(cur_node=full_tree_list[j], examples=train_list_all[j], min_num_examples=value)
            val_acc = get_prediction_accuracy(cur_node=full_tree_list[j], examples=folds[j], min_num_examples=value)

            train_acc_param.append(train_acc)
            val_acc_param.append(val_acc)

        train_acc_global = sum(train_acc_param) / len( train_acc_param)
        val_acc_global = sum(val_acc_param) / len( val_acc_param)
        train_acc_global_list.append(train_acc_global)
        val_acc_global_list.append(val_acc_global)

    return train_acc_global_list, val_acc_global_list



# aa = read_data('data.csv')
# a_folds = preprocess(aa)
#
#
# def get_int_list(num):
#     int_list = []
#     for i in range(num+1):
#         int_list.append(i)
#
#     return int_list
#
#
# def get_int_list2(num):
#     int_list = []
#     i = 0
#     while i <= num:
#         int_list.append(i)
#         i += 20
#
#     return int_list


# b(2)
# depth_list = get_int_list(30)
# train_acc, val_acc = cv_pre_prune(folds=a_folds, value_list=depth_list)

# b(3)
# min_exa_list = get_int_list2(300)
# train_acc3,val_acc3 = cv_post_prune(a_folds, min_exa_list)
# print(train_acc)
# print(val_acc)