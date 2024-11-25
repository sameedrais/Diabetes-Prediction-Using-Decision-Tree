#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
using namespace std;

// Structure for a data point
struct DataPoint {
    vector<double> features;
    int label;
};

// Node structure for the decision tree
struct TreeNode {
    int label;
    int featureIndex;
    double threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int lbl) : label(lbl), featureIndex(-1), threshold(0.0), left(nullptr), right(nullptr) {}
};

// Function to calculate entropy
double entropy(const vector<DataPoint>& data) {
    map<int, int> labelCount;
    for (const auto& point : data) {
        labelCount[point.label]++;
    }
    double ent = 0.0;
    for (const auto& pair : labelCount) {
        double prob = static_cast<double>(pair.second) / data.size();
        ent -= prob * log2(prob);
    }
    return ent;
}

// Helper function to split data based on a numerical threshold
pair<vector<DataPoint>, vector<DataPoint>> splitData(const vector<DataPoint>& data, int featureIndex, double threshold) {
    vector<DataPoint> leftSubset, rightSubset;
    for (const auto& point : data) {
        if (point.features[featureIndex] < threshold) {
            leftSubset.push_back(point);
        } else {
            rightSubset.push_back(point);
        }
    }
    return make_pair(leftSubset, rightSubset);
}

// Modified information gain calculation for numerical features
double informationGain(const vector<DataPoint>& data, int featureIndex, double& bestThreshold) {
    double totalEntropy = entropy(data);
    double bestGain = -1.0;
    bestThreshold = 0.0;

    vector<DataPoint> sortedData = data;
    sort(sortedData.begin(), sortedData.end(), 
         [featureIndex](const DataPoint& a, const DataPoint& b) {
             return a.features[featureIndex] < b.features[featureIndex];
         });

    for (int i = 1; i < sortedData.size(); ++i) {
        double threshold = (sortedData[i - 1].features[featureIndex] + 
                            sortedData[i].features[featureIndex]) / 2;
        
        pair<vector<DataPoint>, vector<DataPoint>> subsets = splitData(data, featureIndex, threshold);
        vector<DataPoint>& leftSubset = subsets.first;
        vector<DataPoint>& rightSubset = subsets.second;

        double weightedEntropy = (leftSubset.size() / static_cast<double>(data.size())) * entropy(leftSubset) +
                                 (rightSubset.size() / static_cast<double>(data.size())) * entropy(rightSubset);

        double gain = totalEntropy - weightedEntropy;
        if (gain > bestGain) {
            bestGain = gain;
            bestThreshold = threshold;
        }
    }
    return bestGain;
}

// Function to find the best feature to split on
pair<int, double> bestFeatureToSplit(const vector<DataPoint>& data) {
    int bestFeature = -1;
    double bestGain = -1.0;
    double bestThreshold = 0.0;

    for (int i = 0; i < data[0].features.size(); ++i) {
        double threshold = 0.0;
        double gain = informationGain(data, i, threshold);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = i;
            bestThreshold = threshold;
        }
    }
    return make_pair(bestFeature, bestThreshold);
}

// Function to build the decision tree recursively
TreeNode* buildTree(const vector<DataPoint>& data) {
    map<int, int> labelCount;
    for (const auto& point : data) {
        labelCount[point.label]++;
    }

    if (labelCount.size() == 1) {
        return new TreeNode(data[0].label);
    }

    if (data.empty() || data[0].features.empty()) {
        int majorityLabel = max_element(labelCount.begin(), labelCount.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) {
                return a.second < b.second;
            })->first;
        return new TreeNode(majorityLabel);
    }

    pair<int, double> bestFeatureAndThreshold = bestFeatureToSplit(data);
    int bestFeature = bestFeatureAndThreshold.first;
    double bestThreshold = bestFeatureAndThreshold.second;

    if (bestFeature == -1) {
        int majorityLabel = max_element(labelCount.begin(), labelCount.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) {
                return a.second < b.second;
            })->first;
        return new TreeNode(majorityLabel);
    }

    pair<vector<DataPoint>, vector<DataPoint>> subsets = splitData(data, bestFeature, bestThreshold);
    vector<DataPoint>& leftData = subsets.first;
    vector<DataPoint>& rightData = subsets.second;

    TreeNode* node = new TreeNode(-1);
    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftData);
    node->right = buildTree(rightData);

    return node;
}
// Function to clean up tree memory
void deleteTree(TreeNode* node) {
    if (node) {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}

// Function to load CSV data with label
vector<DataPoint> loadCSV(const string& filename, bool hasLabel = true) {
    vector<DataPoint> data;
    ifstream file(filename);
    string line;

    bool firstLine = true;
    while (getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }

        stringstream ss(line);
        DataPoint point;
        string value;

        try {
            for (int i = 0; i < 8; ++i) {
                if (!getline(ss, value, ',')) throw invalid_argument("Missing feature value");
                if (value.empty()) throw invalid_argument("Empty feature value");
                point.features.push_back(stod(value));
            }

            if (hasLabel) {
                if (!getline(ss, value, ',')) throw invalid_argument("Missing label");
                if (value.empty()) throw invalid_argument("Empty label");
                point.label = stoi(value);
            }
            data.push_back(point);

        } catch (const invalid_argument& e) {
            cerr << "Error parsing line: " << line << endl;
            cerr << "Reason: " << e.what() << endl;
            continue;
        }
    }

    return data;
}

// Recursive function to serialize the tree into JSON
void serializeTree(TreeNode* node, ostream& os) {
    if (!node) {
        os << "null";
        return;
    }

    os << "{";
    os << "\"featureIndex\":" << node->featureIndex << ",";
    os << "\"threshold\":" << node->threshold << ",";
    os << "\"label\":" << (node->label == -1 ? "null" : to_string(node->label)) << ",";
    os << "\"left\":";
    serializeTree(node->left, os);
    os << ",";
    os << "\"right\":";
    serializeTree(node->right, os);
    os << "}";
}

// Function to save tree JSON to a file
void saveTreeToJson(TreeNode* root, const string& filename) {
    ofstream file(filename);
    serializeTree(root, file);
    file.close();
}

// Function to build a random forest with multiple trees
vector<TreeNode*> buildRandomForest(const vector<DataPoint>& data, int numTrees) {
    vector<TreeNode*> forest;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, data.size() - 1);

    for (int i = 0; i < numTrees; ++i) {
        cout << "Building tree " << i + 1 << "..." << endl;
        vector<DataPoint> sampleData;

        // Bootstrap sampling: randomly sample with replacement
        for (int j = 0; j < data.size(); ++j) {
            sampleData.push_back(data[dis(gen)]);
        }

        // Build a decision tree on the sampled data
        TreeNode* tree = buildTree(sampleData);
        forest.push_back(tree);
    }

    return forest;
}

// Function to make predictions with the decision tree
int predict(TreeNode* node, const DataPoint& point) {
    if (!node->left && !node->right) {
        return node->label;
    }
    if (point.features[node->featureIndex] < node->threshold) {
        return predict(node->left, point);
    } else {
        return predict(node->right, point);
    }
}

// Function to save predictions to CSV
void savePredictions(const vector<int>& predictions, const string& filename) {
    ofstream file(filename);
    file << "Output\n";
    for (int prediction : predictions) {
        file << prediction << "\n";
    }
}

vector<int> loadResults(const string& filename) {
    vector<int> results;
    ifstream file(filename);
    string line;

    // Skip the header
    getline(file, line);

    while (getline(file, line)) {
        try {
            results.push_back(stoi(line));
        } catch (const invalid_argument& e) {
            cerr << "Error parsing line: " << line << endl;
            continue;
        }
    }
    return results;
}

// Function to predict with majority voting from multiple trees in the forest
int predictWithForest(const vector<TreeNode*>& forest, const DataPoint& point) {
    map<int, int> voteCount;

    for (const auto& tree : forest) {
        int prediction = predict(tree, point);
        voteCount[prediction]++;
    }

    // Find the label with the most votes
    return max_element(voteCount.begin(), voteCount.end(),
                       [](const pair<int, int>& a, const pair<int, int>& b) {
                           return a.second < b.second;
                       })->first;
}

// Function to calculate accuracy by comparing predictions with actual results
double calculateAccuracy(const string& predictionFile, const string& resultFile) {
    vector<int> predictions = loadResults(predictionFile);
    vector<int> actualResults = loadResults(resultFile);

    if (predictions.size() != actualResults.size()) {
        cerr << "Prediction and actual result sizes do not match!" << endl;
        return -1.0;
    }

    int correctCount = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == actualResults[i]) {
            ++correctCount;
        }
    }
    return (static_cast<double>(correctCount) / predictions.size()) * 100.0;
}

// Main function
int main() {
    // Load training data
    string trainFile = "../data/diabetes_train.csv";
    vector<DataPoint> trainingData = loadCSV(trainFile);

    // Build a random forest with 15 trees
    int numTrees = 15;
    cout << "Building random forest with " << numTrees << " trees..." << endl;
    vector<TreeNode*> forest = buildRandomForest(trainingData, numTrees);

    // Save each tree in the forest to a JSON file
    for (size_t i = 0; i < forest.size(); ++i) {
        // string filename = "tree_" + to_string(i + 1) + ".json";
        string filename = "JSON/tree_" + to_string(i + 1) + ".json";
        saveTreeToJson(forest[i], filename);
        cout << "Tree " << i + 1 << " saved to '" << filename << "'" << endl;
    }

    string testFile = "../data/diabetes_test.csv";
    vector<DataPoint> testData = loadCSV(testFile, false);  // Load test data without labels


    // Get predictions for each test point using the forest
    vector<int> predictions;
    for (const auto& testPoint : testData) {
        predictions.push_back(predictWithForest(forest, testPoint));
    }

    string outputFile = "../data/diabetes_output.csv";
    savePredictions(predictions, outputFile);

    cout << "Predictions saved to " << outputFile << endl;

    // Calculate accuracy
    string resultFile = "../data/diabetes_result.csv";
    double accuracy = calculateAccuracy(outputFile, resultFile);

    if (accuracy >= 0) {
        cout << "Model Accuracy: " << accuracy << "%" << endl;
    }

    for (TreeNode* tree : forest) {
        deleteTree(tree);
    }

    return 0;
}
