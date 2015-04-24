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
#define ASSERT_FLOAT_EQUAL(a, b) if (((a) == (b)) || (fabs((a) - (b)) <= ( (fabs((a)) > fabs((b)) ? fabs((b)) : fabs((a))) * std::numeric_limits<double>::epsilon())) ) fail(__FILE__ ":", __LINE__, ": Assert fail: expected ", (a), " actual  ", (b), ", " #a " == " #b ", at " __FILE__ ":",__LINE__)



int main() {
    Featurizer f = Featurizer();
    ASSERT_EQUAL(0, f.size());
    ASSERT_NOTEQUAL(1, f.size());

    vector<string> features = {"what", "i", "cannot", "create", "i", "do", "not", "understand"};
    for (auto const& x: features) {
        f.add_feature(x);
    }
    ASSERT_EQUAL(7, f.size());  // `i` is counted twice.

    vector<int> indices = f.get_feature_indices(vector<string>({"i", "cannot", "tell", "you", "what", "knowledge", "is" }));
    ASSERT_EQUAL(10, indices[indices.size() - 1]);

    // proper randomized algorithm should use random_device
    int seed = 44;
    int k = 3;

    default_random_engine e1(seed);
    FM fm = FM(k, e1);

    vector<int> guess_index({0, 1, 2});
    vector<double> guess_weights({1.5, 0.2, 2.0});
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
    FM fm2 = FM(k, e1);

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
                fm2.learn({i, 3 + j}, {1.0, 1.0}, targets[i][j]);
            }
        }
    }

    double final_error = 0.0;
    guess = fm2.predict({2, 6}, {1.0, 1.0});
    target = -1.19567812;
    final_error = guess - target;
    ASSERT_TRUE((final_error) < (initial_error));

    return 0;
}
