import numpy as np
import sklearn.metrics as metrics
from scipy import stats
from math import log


class Node:
    """Represents a node in the decision tree. """
    def __init__(self, left=None, right=None, label=None, split_rule=(), class0probability=-1, class1probability=-1):
        self.left = left
        self.right = right
        self.label = label # set to mode later
        self.split_rule = split_rule
        self.class0probability = class0probability
        self.class1probability = class1probability

class DecisionTree:
    """Represents a single decision tree in the forest. """
    def __init__(self, max_depth): # include params
        self.head = Node()
        self.max_depth = max_depth

    def find_entropy(self, hist):
        """Defines the entropy for the tree. """
        if hist[0] == 0 or hist[1] == 0:
            return 0
        total = hist[0] + hist[1]
        p0 = hist[0] / total
        p1 = hist[1] / total
        return -1 * p0 * log(p0) - 1 * p1 * log(p1)

    def impurity(self, left_label_hist, right_label_hist):
        """Calculates the impurity formula. Will use dictionaries for histograms. """
        n1 = sum(left_label_hist)
        n2 = sum(right_label_hist)
        return (n1*self.find_entropy(left_label_hist) + n2*self.find_entropy(right_label_hist))/(n1 + n2)

    def segmenter(self, data, labels): # labels is one dimensional
        """Splits the data along the tree. """
        num_features = data.shape[1]
        numberOfOnes = sum(labels)
        numberOfZeros = len(labels) - numberOfOnes
        splitrule = (-1, -1)

        def findSplitGivenFeature(col):
            lefthist = [0, 0]
            righthist = [numberOfZeros, numberOfOnes]
            sortIndex = np.argsort(col)
            colUnique = np.unique(col)
            splits = np.zeros(len(colUnique)-1)
            for j in range(len(splits)):
                splits[j] = 0.5*(colUnique[j] + colUnique[j+1])

            minentropy = float("inf")
            minsplit = None
            i = 0
            splitIndex = -1
            for split in splits:
                while True:
                    if col[sortIndex[i]] >  split:
                        break
                    oneorzero = labels[sortIndex[i]]
                    lefthist[oneorzero] += 1
                    righthist[oneorzero] -= 1
                    i += 1
                imp = self.impurity(lefthist, righthist)
                if imp < minentropy:
                    splitIndex = i
                    minsplit = split
                    minentropy = imp
            return minsplit, minentropy, sortIndex, splitIndex

        ret_entropy = float("inf")
        ret_sortindex = None
        ret_splitIndex = -1
        for feat in range(num_features):
            split, min_entropy, sortIndex, splitIndex = findSplitGivenFeature(data[:, feat])

            if min_entropy < ret_entropy:
                splitrule = (feat, split)
                ret_entropy = min_entropy
                ret_sortindex, ret_splitIndex = sortIndex, splitIndex
        return splitrule, ret_entropy, ret_sortindex, ret_splitIndex

    def train(self, data, labels, node, depth):
        numberOfOnes = sum(labels)
        numberOfZeros = len(labels) - numberOfOnes
        if numberOfOnes == len(labels) or numberOfOnes == 0: # if it's a pure node
            node.label = labels[0]
            node.class0probability = numberOfZeros/(numberOfZeros + numberOfOnes)
            node.class1probability = numberOfOnes/(numberOfZeros + numberOfOnes)
            return
        elif depth == self.max_depth:
            node.label = stats.mode(labels)[0][0]
            node.class0probability = numberOfZeros / (numberOfZeros + numberOfOnes)
            node.class1probability = numberOfOnes / (numberOfZeros + numberOfOnes)
            return
        current_entropy = self.find_entropy([numberOfZeros, numberOfOnes])
        splitrule, entropy, sortIndex, splitIndex = self.segmenter(data, labels)
        if current_entropy == entropy or sortIndex == None: # no change in entropy
            node.label = stats.mode(labels)[0][0]
            node.class0probability = numberOfZeros / (numberOfZeros + numberOfOnes)
            node.class1probability = numberOfOnes / (numberOfZeros + numberOfOnes)
            return
        node.split_rule = splitrule
        lefthalfindices = sortIndex[0:splitIndex]
        righthalfindices = sortIndex[splitIndex:]
        leftdata, leftlabels = data[lefthalfindices], labels[lefthalfindices]
        rightdata, rightlabels = data[righthalfindices], labels[righthalfindices]

        node.left = Node()
        node.right = Node()

        self.train(leftdata, leftlabels, node.left, depth+1)
        self.train(rightdata, rightlabels, node.right, depth+1)

    def predict(self, x, showPath=False, numericalnames=None, categoricalnames=None):
        node = self.head
        firstsplit = -1
        while True:
            if node.label != None:
                if showPath:
                    print("Classification: ", node.label)
                return node.label, node.class0probability, node.class1probability, firstsplit
            rule = node.split_rule
            if x[rule[0]] < rule[1]:
                if showPath and numericalnames and categoricalnames:
                    print(getForestFeatureFromIndex(rule[0],categoricalnames,numericalnames), " < ", rule[1])
                    if node == self.head:
                        firstsplit = (getForestFeatureFromIndex(rule[0],categoricalnames,numericalnames), " < ", rule[1])
                elif showPath:
                    print(rule[0], " < ", rule[1])
                    if node == self.head:
                        firstsplit = (rule[0], "<", rule[1])
                node = node.left
            else:
                if showPath and numericalnames and categoricalnames:
                    print(getForestFeatureFromIndex(rule[0], categoricalnames, numericalnames), " > ", rule[1])
                    if node == self.head:
                        firstsplit = (getForestFeatureFromIndex(rule[0], categoricalnames, numericalnames), " > ", rule[1])
                elif showPath:
                    print(rule[0], " > ", rule[1])
                    if node == self.head:
                        firstsplit = (rule[0], " > ", rule[1])
                node = node.right

    def print_accuracy(self, test_x, test_y):
        prediction = np.zeros(test_y.size)
        for i in range(test_y.size):
            prediction[i] = self.predict(test_x[i])[0]
        accuracy = metrics.accuracy_score(prediction, test_y)
        print("Test accuracy: {0}".format(accuracy))
        return accuracy

def getForestFeatureFromIndex(index, categoricalnames, numericalnames):
    if index < len(categoricalnames):
        return categoricalnames[index]
    else:
        return numericalnames[index-len(categoricalnames)]

class RandomForest:
    def __init__(self, numTrees, bag, xtrain, ytrain):
        self.forest = [None] * numTrees
        self.numTrees = numTrees
        N = int(ytrain.size * bag)
        for i in range(numTrees):
            # print("training tree number: ", i)
            subsetIndices = np.random.choice(ytrain.size, N, replace=False)
            self.forest[i] = DecisionTree(10)
            self.forest[i].train(xtrain[subsetIndices], ytrain[subsetIndices], self.forest[i].head, 0)

    def predictPoint(self, x,numericalnames=None, categoricalnames=None):
        predictProbabilities = [0, 0] #probability of class 0, probability of class 1

        firstsplit = {}

        for i in range(self.numTrees):
            label, class0prob, class1prob, nodesplit = self.forest[i].predict(x,showPath=True,numericalnames=numericalnames, categoricalnames=categoricalnames)
            predictProbabilities[0] += class0prob
            predictProbabilities[1] += class1prob
            if nodesplit in firstsplit:
                firstsplit[nodesplit] += 1
            else:
                firstsplit[nodesplit] = 1
        print(firstsplit)
        return predictProbabilities.index(max(predictProbabilities))

    def predict(self, xtest):
        prediction = np.zeros(xtest.shape[0])
        for i in range(xtest.shape[0]):
            prediction[i] = self.predictPoint(xtest[i])
        return prediction

    def print_accuracy(self, prediction, ytest):
        accuracy = metrics.accuracy_score(prediction, ytest)
        print("Test accuracy: {0}".format(accuracy))
        return accuracy

    def submitSpamKaggle(self, xtest):
        prediction = self.predict(xtest)
        kagglePrediction = np.array(prediction)
        id = np.array(range(1, kagglePrediction.shape[0] + 1))
        submission = np.column_stack((id, kagglePrediction))
        np.savetxt("spamsubmissionkaggle.csv", submission.astype(int), fmt='%i', delimiter=",")

    def submitCensusKaggle(self, xtest):
        prediction = self.predict(xtest)
        kagglePrediction = np.array(prediction)
        id = np.array(range(1, kagglePrediction.shape[0] + 1))
        submission = np.column_stack((id, kagglePrediction))
        np.savetxt("censussubmissionkaggle.csv", submission.astype(int), fmt='%i', delimiter=",")