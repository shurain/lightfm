#include <iostream>
#include <random>

#include "featurizer.h"
#include "fm.h"

using namespace lightfm;
using namespace std;

/* Some test code copied from Crow.
 * Should check the license later on.
 */

bool failed__ = false;
void error_print()
{
    cerr << endl;
}

template <typename A, typename ...Args>
void error_print(const A& a, Args...args)
{
    cerr<<a;
    error_print(args...);
}

template <typename ...Args>
void fail(Args...args) { error_print(args...);failed__ = true; }

#define ASSERT_TRUE(x) if (!(x)) fail(__FILE__ ":", __LINE__, ": Assert fail: expected ", #x, " is true, at " __FILE__ ":",__LINE__)
#define ASSERT_EQUAL(a, b) if ((a) != (b)) fail(__FILE__ ":", __LINE__, ": Assert fail: expected ", (a), " actual " , (b),  ", " #a " == " #b ", at " __FILE__ ":",__LINE__)
#define ASSERT_NOTEQUAL(a, b) if ((a) == (b)) fail(__FILE__ ":", __LINE__, ": Assert fail: not expected ", (a), ", " #a " != " #b ", at " __FILE__ ":",__LINE__)
// Assuming 0.000001 has enough resolution.
#define ASSERT_FLOAT_EQUAL(a, b) if (((a) != (b)) && (fabs((a) - (b)) > ( (fabs((a)) > fabs((b)) ? fabs((b)) : fabs((a))) * 0.000001)) ) fail(__FILE__ ":", __LINE__, ": Assert fail: expected ", (a), " actual  ", (b), ", " #a " == " #b ", at " __FILE__ ":",__LINE__)
#define ASSERT_FLOAT_NOTEQUAL(a, b) if (((a) == (b)) || (fabs((a) - (b)) <= ( (fabs((a)) > fabs((b)) ? fabs((b)) : fabs((a))) * 0.000001)) ) fail(__FILE__ ":", __LINE__, ": Assert fail: expected ", (a), " actual  ", (b), ", " #a " != " #b ", at " __FILE__ ":",__LINE__)



int main() {
    DictFeaturizer f = DictFeaturizer();
    vector<string> features = {"what", "i", "cannot", "create", "i", "do", "not", "understand"};
    f.get_feature_indices(features);

    vector<uint32_t> indices = f.get_feature_indices(vector<string>({"i", "cannot", "tell", "you", "what", "knowledge", "is" }));
    ASSERT_EQUAL(10, indices[indices.size() - 1]);

    // proper randomized algorithm should use random_device
    int seed = 44;
    int k = 3;

    default_random_engine e1(seed);

    vector<uint32_t> guess_index({0, 1, 2});
    vector<double> guess_weights({1.5, 0.2, 2.0});

    double learning_rate = 0.01;
    double w_reg = 0.01;
    double v_reg = 0.01;
    double stdev = 0.01;

    // 3 feauters
    FM fm = FM(3, k, w_reg, v_reg, learning_rate, stdev, e1);
    double guess = fm.predict(guess_index, guess_weights);
    ASSERT_EQUAL(guess_index.size(), fm.vi.size());
    ASSERT_EQUAL(k, fm.vi[0].size());

    // Naive calculation. O(kn^2). Actual implementation takes O(kn).
    double result = fm.w0;
    for (int i = 0; i < fm.wi.size(); ++i) {
        result += fm.wi[i] * guess_weights[i];
    }
    double tmp = 0.0;
    for (int i = 0; i < guess_index.size(); ++i) {
        for (int j = 0; j < guess_index.size(); ++j) {
            double v_dot = 0.0;
            for (int l = 0; l < k; ++l) {
                v_dot += fm.vi[i][l] * fm.vi[j][l];
            }
            tmp += v_dot * guess_weights[i] * guess_weights[j];
        }
    }
    for (int i = 0; i < guess_index.size(); ++i) {
        double v_dot = 0.0;
        for (int l = 0; l < k; ++l) {
            v_dot += fm.vi[i][l] * fm.vi[i][l];
        }
        tmp -= v_dot * guess_weights[i] * guess_weights[i];
    }

    tmp /= 2.0;
    result += tmp;
    ASSERT_FLOAT_EQUAL(3.0, 3.0000000000000000000000000001);
    ASSERT_FLOAT_EQUAL(result, guess);

    // learning
    // I have no idea how I should check this.
    // Let's just check if the error goes down.

    // Seven features as in 3 users, 4 items.
    // user : array([[-1.33025759,  1.27872403],
    //               [ 0.2494233 , -0.11624764],
    //               [ 1.08083429, -1.16247639]])
    //
    // item : array([[ 0.0168526 , -1.35498234],
    //               [ 1.60099728, -0.45764304],
    //               [-1.06171398,  1.30114198],
    //               [-0.5561359 ,  0.5114834 ]])
    //
    //        array([[-1.75506677, -2.71493793,  3.07615459,  1.39385011],
    //               [ 0.16171693,  0.45252594, -0.41607089, -0.19817199],
    //               [ 1.59334984,  2.26241198, -2.66008371, -1.19567812]])

    // This matrix can be factorized into two rank 2 matrices.
    vector<vector<double>> targets = {
        {-1.75506677, -2.71493793,  3.07615459,  1.39385011},
        { 0.16171693,  0.45252594, -0.41607089, -0.19817199},
        { 1.59334984,  2.26241198, -2.66008371, -1.19567812}
    };

    k = 2;
    // Seven features (3 user, 4 item)
    FM fm2 = FM(7, k, w_reg, v_reg, learning_rate, stdev, e1);

    double initial_error = 0.0;
    guess = fm2.predict({2, 6}, {1.0, 1.0});
    double target = -1.19567812;
    initial_error = guess - target;

    int epoch = 30000;
    for (int r = 0; r < epoch; ++r) {
        for (int i = 0; i < 3; ++i) {
            // each user
            for (int j = 0; j < 4; ++j) {
                // each item
                if (i == 2 && j == 3) {
                    // Don't let the algorithm peak at the answer.
                    continue;
                }
                vector<uint32_t> l({(uint32_t)i, (uint32_t)3 + j});
                fm2.learn(l, {1.0, 1.0}, targets[i][j]);
            }
        }
    }

    double final_error = 0.0;
    guess = fm2.predict({2, 6}, {1.0, 1.0});
    target = -1.19567812;
    final_error = guess - target;
    cout << "DictFeaturizer" << endl;
    cout << "initial error : " << initial_error << "\t" << "final error : " << final_error << endl;
    ASSERT_TRUE((final_error) < (initial_error));


    int bit = 18;
    int rng_seed = 42;
    HashFeaturizer f2 = HashFeaturizer(bit, rng_seed);
    vector<uint32_t> hash_indices = f2.get_feature_indices(features);
    ASSERT_EQUAL(hash_indices[0], 81566);


    FM fm3 = FM(pow(2, bit), k, w_reg, v_reg, learning_rate, stdev, e1);

    initial_error = 0.0;
    guess = fm3.predict({2, 6}, {1.0, 1.0});
    target = -1.19567812;
    initial_error = guess - target;

    epoch = 30000;
    vector<string> feature_string = {"user0", "user1", "user2", "item0", "item1", "item2", "item3"};
    for (int r = 0; r < epoch; ++r) {
        for (int i = 0; i < 3; ++i) {
            // each user
            for (int j = 0; j < 4; ++j) {
                // each item
                if (i == 2 && j == 3) {
                    // Don't let the algorithm peak at the answer.
                    continue;
                }
                // for (auto & x: f2.get_feature_indices({feature_string[i], feature_string[3 + j]})) {
                //     cout << x << endl;
                // }
                fm3.learn(f2.get_feature_indices({feature_string[i], feature_string[3 + j]}), {1.0, 1.0}, targets[i][j]);
            }
        }
    }

    final_error = 0.0;
    guess = fm3.predict(f2.get_feature_indices({feature_string[2], feature_string[6]}), {1.0, 1.0});
    target = -1.19567812;
    final_error = guess - target;
    cout << "HashFeaturizer" << endl;
    cout << "initial error : " << initial_error << "\t" << "final error : " << final_error << endl;
    ASSERT_TRUE((final_error) < (initial_error));


    return 0;
}
