#include <iostream>
#include <random>
#include <fstream>
#include <cmath>

#include "util.h"
#include "featurizer.h"
#include "fm.h"

using namespace lightfm;
using namespace std;

int main() {
    const string train_file ("/Users/shurain/prj/recsys/data/ra.train.shuffled.libfm");
    const string test_file ("/Users/shurain/prj/recsys/data/ra.test.shuffled.libfm");

    ifstream training_data (train_file);
    ifstream test_data (test_file);
    string line;

    Featurizer featurizer = Featurizer();

    vector<double> targets;
    vector<vector<int>> feature_indices;
    vector<vector<double>> feature_weights;

    // Load data and create feature vector from it.
    if (training_data.is_open()) {
        cout << "Loading train data..." << endl;
        while(getline(training_data, line)) {
            vector<string> tokens = split(line, ' ');

            // First token is the target
            targets.push_back(stod(tokens[0]));

            vector<int> feature_index;
            vector<double> feature_weight;
            // Rest of the tokens are the features
            for (int i = 1; i < tokens.size(); ++i) {
                vector<string> feat = split(tokens[i], ':');
                feature_index.push_back(featurizer.get_feature_index(feat[0]));
                feature_weight.push_back(stod(feat[1]));
            }
            feature_indices.push_back(feature_index);
            feature_weights.push_back(feature_weight);
        }
    }

    cout << "Loaded data" << endl;
    cout << "Number of rows: " << targets.size() << "\t" << "number of features: " << featurizer.size() << endl;

    vector<double> test_targets;
    vector<vector<int>> test_feature_indices;
    vector<vector<double>> test_feature_weights;

    if (test_data.is_open()) {
        cout << "Loading test data..." << endl;
        while(getline(test_data, line)) {
            vector<string> tokens = split(line, ' ');

            // First token is the target
            test_targets.push_back(stod(tokens[0]));

            vector<int> feature_index;
            vector<double> feature_weight;
            // Rest of the tokens are the features
            for (int i = 1; i < tokens.size(); ++i) {
                vector<string> feat = split(tokens[i], ':');
                feature_index.push_back(featurizer.get_feature_index(feat[0]));
                feature_weight.push_back(stod(feat[1]));
            }
            test_feature_indices.push_back(feature_index);
            test_feature_weights.push_back(feature_weight);
        }
    }

    int seed = 42;
    int epoch = 10;
    int k = 10;
    default_random_engine e1(seed);

    FM fm = FM(k, e1);

    for (int i = 0; i < epoch; ++i) {
        double train_error = 0.0;
        for (int j = 0; j < targets.size(); ++j) {
            double guess = fm.learn(feature_indices[j], feature_weights[j], targets[j]);
            train_error += pow(guess - targets[j], 2);
        }
        double total_error = 0.0;
        for (int l = 0; l < test_targets.size(); ++l) {
            double guess = fm.predict(test_feature_indices[l], test_feature_weights[l]);
            double partial_error = pow(guess - test_targets[l], 2);
            total_error += partial_error;
        }
        cout << "Epoch: " << i << "\tTrain: " << sqrt(train_error / targets.size()) << "\tTest: " << sqrt(total_error/test_targets.size()) << endl;
    }
}
