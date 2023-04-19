import numpy as np
import pandas as pd

def entropy(
        features_data, target, class_target):

    class_count = features_data.shape[0]
    entropy = 0

    for c in class_target:
        label_class_count = features_data[features_data[target] == c].shape[0]

        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = - probability_class * np.log2(probability_class)

        entropy += entropy_class

    return entropy


def total_entropy(
        train_data, target, class_target):

    total_row = train_data.shape[0]
    total_entr = 0

    for c in class_target:
        total_class_count = train_data[train_data[target] == c].shape[0]
        if total_class_count == 0:
            total_class_entr = 0
        else:
            total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count / total_row)
        total_entr += total_class_entr

    return total_entr


def information_gain(
        feature_name, train_data, target, class_target):

    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = entropy(feature_value_data, target, class_target)
        feature_value_probability = feature_value_count / total_row
        feature_info += feature_value_probability * feature_value_entropy

    return total_entropy(train_data, target, class_target) - feature_info


def most_informative_feature(
        train_data, target, class_target):

    feature_list = train_data.columns.drop(target)
    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:
        feature_info_gain = information_gain(feature, train_data, target, class_target)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def build_sub_tree(
        feature_name, train_data, target, class_target):

    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}

    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]

        assigned_to_node = False
        for c in class_target:
            class_count = feature_value_data[feature_value_data[target] == c].shape[0]

            if class_count == count:
                tree[feature_value] = c
                train_data = train_data[train_data[feature_name] != feature_value]
                assigned_to_node = True
        if not assigned_to_node:
            tree[feature_value] = "?"

    return tree, train_data


def build_tree(
        root, prev_feature_value,
        train_data, target, class_target):

    if train_data.shape[0] != 0:
        max_info_feature = most_informative_feature(train_data, target, class_target)
        tree, train_data = build_sub_tree(max_info_feature, train_data, target, class_target)
        next_root = None

        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = train_data[train_data[max_info_feature] == node]
                build_tree(next_root, node, feature_value_data, target, class_target)


def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    build_tree(tree, None, train_data, label, class_list)

    return tree


def predict(
        tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            return None


def evaluate(
        tree, test_data, label):

    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data.iterrows():
        result = predict(tree, test_data.iloc[index])
        if result == test_data[label].iloc[index]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    return accuracy

train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')


train_data['Jitter'] = pd.qcut(train_data['Jitter'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['Shimmer'] = pd.qcut(train_data['Shimmer'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['GNE'] = pd.qcut(train_data['GNE'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['Irregularity'] = pd.qcut(train_data['Irregularity'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['Noise'] = pd.qcut(train_data['Noise'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['OverallSeverity'] = pd.qcut(train_data['OverallSeverity'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['mean_F0'] = pd.qcut(train_data['mean_F0'], q=6, labels=['1', '2', '3', '4', '5', '6'])
train_data['SD_F0'] = pd.qcut(train_data['SD_F0'], q=6, labels=['1', '2', '3', '4', '5', '6'])

test_data['Jitter'] = pd.qcut(test_data['Jitter'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['Shimmer'] = pd.qcut(test_data['Shimmer'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['GNE'] = pd.qcut(test_data['GNE'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['Irregularity'] = pd.qcut(test_data['Irregularity'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['Noise'] = pd.qcut(test_data['Noise'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['OverallSeverity'] = pd.qcut(test_data['OverallSeverity'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['mean_F0'] = pd.qcut(test_data['mean_F0'], q=6, labels=['1', '2', '3', '4', '5', '6'])
test_data['SD_F0'] = pd.qcut(test_data['SD_F0'], q=6, labels=['1', '2', '3', '4', '5', '6'])


train = {'Jitter': train_data['Jitter'], 'Shimmer': train_data['Shimmer'],
     'GNE': train_data['GNE'], 'Irregularity': train_data['Irregularity'],
     'Noise': train_data['Noise'], 'OverallSeverity': train_data['OverallSeverity'],
     'mean_F0': train_data['mean_F0'], 'SD_F0': train_data['SD_F0'],
     'Diagnosis': train_data['Diagnosis']}

test = {'Jitter': test_data['Jitter'], 'Shimmer': test_data['Shimmer'],
     'GNE': test_data['GNE'], 'Irregularity': test_data['Irregularity'],
     'Noise': test_data['Noise'], 'OverallSeverity': test_data['OverallSeverity'],
     'mean_F0': test_data['mean_F0'], 'SD_F0': test_data['SD_F0'],
     'Diagnosis': test_data['Diagnosis']}

train = pd.DataFrame(train)
test = pd.DataFrame(test)

print(train)
print(test)

features = ['Jitter', 'Shimmer', 'GNE', 'Irregularity', 'Noise', 'OverallSeverity', 'mean_F0', 'SD_F0']
tree = id3(train, 'Diagnosis')
print('Tree: ', tree)

accuracy = evaluate(tree, test, 'Diagnosis')
print("accuracy:", accuracy)
