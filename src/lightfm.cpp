#include <iostream>
#include <random>
#include <fstream>
#include <cmath>

#include "tclap/CmdLine.h"

#include "featurizer.h"
#include "fm.h"
#include "parser.h"


using namespace lightfm;
using namespace std;

int main(int argc, char** argv) {

    try {
        TCLAP::CmdLine cmd("LightFM", ' ', "0.01");
        TCLAP::ValueArg<string> train_arg("d", "training_file", "Training file path.", true, "file_name", "file name");
        TCLAP::ValueArg<string> test_arg("t", "test_file", "Test file path.", true, "file_name", "file name");

        TCLAP::ValueArg<int> n_pass_arg("", "n_pass", "Number of passes.", false, 100, "int");
        TCLAP::ValueArg<int> k_arg("", "n_factors", "Dimensionality of the factorization.", false, 10, "int");

        TCLAP::ValueArg<double> w_reg_arg("", "w_reg", "Regularization for w.", false, 0.01, "float");
        TCLAP::ValueArg<double> v_reg_arg("", "v_reg", "Regularization for v.", false, 0.01, "float");
        TCLAP::ValueArg<double> learning_rate_arg("", "learning_rate", "Learning rate.", false, 0.01, "float");
        TCLAP::ValueArg<double> stdev_arg("", "stdev", "Standard deviation for the weight initialization Gaussian distribution.", false, 0.01, "float");

        TCLAP::ValueArg<int> hash_bit_arg("b", "hash_bits", "Number of bits for feature hash table", false, 18, "int");

        cmd.add(train_arg);
        cmd.add(test_arg);

        cmd.add(n_pass_arg);
        cmd.add(k_arg);

        cmd.add(w_reg_arg);
        cmd.add(v_reg_arg);
        cmd.add(learning_rate_arg);
        cmd.add(stdev_arg);

        cmd.add(hash_bit_arg);

        cmd.parse(argc, argv);

        const string train_file = train_arg.getValue();
        const string test_file = test_arg.getValue();

        int n_pass = n_pass_arg.getValue();
        int k = k_arg.getValue();

        double w_reg = w_reg_arg.getValue();
        double v_reg = v_reg_arg.getValue();
        double learning_rate = learning_rate_arg.getValue();
        double stdev = stdev_arg.getValue();

        int bit = hash_bit_arg.getValue();
        random_device rd;

        unique_ptr<Featurizer> featurizer;
        if (!hash_bit_arg.isSet()) {
            featurizer = unique_ptr<DictFeaturizer> (new DictFeaturizer());
        }
        else {
            featurizer = unique_ptr<HashFeaturizer> (new HashFeaturizer(bit, rd()));
        }

        string line;

        vector<double> targets;
        vector<vector<uint32_t>> feature_indices;
        vector<vector<double>> feature_weights;

        cout << "Loading train data..." << endl;
        Parser parser = Parser(*featurizer);
        parser.read_data(train_file, targets, feature_indices, feature_weights);

        cout << "Loaded data" << endl;
        cout << "Number of rows: " << targets.size() << "\t" << "number of features: " << featurizer->size() << endl;

        vector<double> test_targets;
        vector<vector<uint32_t>> test_feature_indices;
        vector<vector<double>> test_feature_weights;

        parser.read_data(test_file, test_targets, test_feature_indices, test_feature_weights);
        cout << "Test data rows: " << test_targets.size() << "\t" << "number of features: " << featurizer->size() << endl;

        int seed = 42;
        default_random_engine e1(seed);

        FM fm = FM(featurizer->size(), k, w_reg, v_reg, learning_rate, stdev, e1);

        for (int i = 0; i < n_pass; ++i) {
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
    catch (TCLAP::ArgException &e) {
        cerr << "Error: " << e.error() << " for args " << e.argId() << endl;
    }
}
