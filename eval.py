import os

def read_results(filepath, gold_label):
    with open(filepath, errors="ignore") as f:
        data = f.readlines()

    # get prediction from each testing data
    result_set = [] # contain tuple (gold label, predicted label)
    for each in data:
        each = each.strip("\n")
        predicted = each.split(" ")[-1]
        tuple = (gold_label, predicted)
        result_set.append(tuple)

    return result_set


def get_eval(result_set):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for each in result_set:
        gold_label = each[0]
        predicted = each[1]
        if gold_label == "spam" and predicted == "spam":
            true_pos += 1
        elif gold_label == "ham" and predicted == "spam":
            false_pos += 1
        elif gold_label == "spam" and predicted == "ham":
            false_neg +=1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 / ((1/precision) + (1/recall))

    result = (precision, recall, f1)

    return result

path = "eval"

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        # full_path = os.path.join(path, folder)
        # print(folder_path)
        ham = None
        spam = None
        for file in os.listdir(folder_path):
            full_path = folder_path + "\\" + file
            # print("full_path")
            # print(full_path)
            if "ham" in full_path:
                ham = read_results(full_path, "ham")
            elif "spam" in full_path:
                spam = read_results(full_path, "spam")

        result_data = ham + spam
        print(get_eval(ham + spam))



