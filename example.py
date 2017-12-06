from sklearn import svm
from itertools import combinations
import random
import numpy as np

def cosine_distnace(v1, v2):
    cos = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos

def get_dataset(features_file):
	with open(features_file) as f:
		lines = [line.strip("\n").split(" ") for line in f.readlines()]
	features_comb = list(combinations(lines, 2))
	_same = []
	_diff = []
	for one_combination in features_comb:
		# print one_combination
		one_combination = list(one_combination)
		if one_combination[0][0] == one_combination[1][0] and one_combination[0][1] != one_combination[1][1]:
			one_combination.append(1)
			_same.append(one_combination)
		else:
			one_combination.append(0)
			_diff.append(one_combination)
	
	random.shuffle(_diff)
	random.shuffle(_diff)
	samples = []
	for i in range(len(_same)):
		# diff_sample = [map(float, _diff[i][0][2:]) + map(float, _diff[i][1][2:]), 0]
		# same_sample = [map(float, _same[i][0][2:]) + map(float, _same[i][1][2:]), 1]
		# samples.extend(diff_sample)
		# samples.extend(same_sample)
		diff_sample = [[cosine_distnace(map(float, _diff[i][0][2:]), map(float, _diff[i][1][2:]))], 0]
		same_sample = [[cosine_distnace(map(float, _same[i][0][2:]), map(float, _same[i][1][2:]))], 1]
		samples.extend(diff_sample)
		samples.extend(same_sample)
	samples = np.reshape(np.array(samples), (-1, 2))
	random.shuffle(samples)
	return samples[:, 0].tolist(), samples[:, 1].tolist()
	
if __name__ == "__main__":
	train_data, train_label = get_dataset("./train_features.txt")
	test_data, test_label = get_dataset("./test_features.txt")
	# clf = svm.SVC(kernel='linear', probability=True, C=30.0)
	# clf = svm.SVC(kernel='poly', degree=4, probability=True, C=100.0)
	# clf = svm.SVC(probability=True, C=30.0)
	# clf = svm.SVC(kernel='sigmoid', probability=True, C=10)
	clf.fit(train_data, train_label)
	result = clf.predict_proba(test_data)
	correct = 0
	for (p1, p2), y in zip(result, test_label):
		predict_label = 0 if p1 > p2 else 1
		correct += (y == predict_label)
		# print correct
		print "class 0:", p1, "\t\tclass 1:\t", p2, "\treal: ",  y, "\tpredict: ", predict_label
	print float(correct)/len(test_label)
	print clf.classes_
