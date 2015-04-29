#include <iostream>
#include <random>
#include <fstream>
#include <cmath>

#include "featurizer.h"
#include "fm.h"
#include "parser.h"

using namespace lightfm;
using namespace std;

int main() {
    const string train_file ("/Users/shurain/prj/recsys/data/ra.train.shuffled.libfm");
    const string test_file ("/Users/shurain/prj/recsys/data/ra.test.shuffled.libfm");

    string line;

    int bit = 20;
    int rng_seed = 43;
    HashFeaturizer featurizer = HashFeaturizer(bit, rng_seed);
    /* DictFeaturizer featurizer = DictFeaturizer(); */

    vector<double> targets;
    vector<vector<uint32_t>> feature_indices;
    vector<vector<double>> feature_weights;

    cout << "Loading train data..." << endl;
    Parser parser = Parser(featurizer);
    parser.read_data(train_file, targets, feature_indices, feature_weights);

    cout << "Loaded data" << endl;
    cout << "Number of rows: " << targets.size() << "\t" << "number of features: " << featurizer.size() << endl;

    vector<double> test_targets;
    vector<vector<uint32_t>> test_feature_indices;
    vector<vector<double>> test_feature_weights;

    parser.read_data(test_file, test_targets, test_feature_indices, test_feature_weights);
    cout << "Test data rows: " << test_targets.size() << "\t" << "number of features: " << featurizer.size() << endl;

    int seed = 42;
    int epoch = 10;
    int k = 10;
    default_random_engine e1(seed);

    double w_reg = 0.01;
    double v_reg = 0.01;
    double learning_rate = 0.01;
    double stdev = 0.01;

    FM fm = FM(featurizer.size(), k, w_reg, v_reg, learning_rate, stdev, e1);

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
