#include <iostream>
#include <algorithm>
#include <cmath>

#include "fm.h"

using namespace lightfm;

FM::FM(int dim, int k, double w_reg, double v_reg, double learning_rate, double stdev, std::default_random_engine e1)
    : e1(e1),
    k(k),
    w_reg(w_reg),
    v_reg(v_reg),
    learning_rate(learning_rate)
{
    d = std::normal_distribution<>(0, stdev);

    w0 = d(e1);
    std::vector<double> w(dim);
    std::generate_n(w.begin(), dim, [&]{ return d(e1);});
    wi = std::move(w);

    for (int i = 0; i < dim; ++i) {
        std::vector<double> v(k);
        std::generate_n(v.begin(), k, [&]{ return d(e1);});

        vi.push_back(std::move(v));
    }
}

FM::~FM() {
}

double FM::_predict(const std::vector<uint32_t> & indices, const std::vector<double> & weights, std::vector<double> & vif_dot_wi, bool precompute) {
    /* Calculate the prediction given features.
     * While naive implementation of pairwise interaction takes O(kn^2)
     * computation, this implementation takes O(kn). Refer to Lemma 3.1 of
     * the original factorization machines paper.
     */
    // global bias
    double result = w0;

    // Lemma 3.1 first and second term inside the summation
    double first = 0.0;
    double second = 0.0;

    for (int i = 0; i < indices.size(); ++i) {
        uint32_t index = indices[i];

        // feature specific bias
        result += wi[index] * weights[i];
    }

    for (int f = 0; f < k; ++f) {
        double tmp = 0.0;
        for (int i = 0; i < indices.size(); ++i) {
            uint32_t index = indices[i];
            tmp += vi[index][f] * weights[i];

            second += pow(vi[index][f], 2) * pow(weights[i], 2);
        }

        if (precompute) {
            vif_dot_wi[f] = tmp;
        }

        first += pow(tmp, 2);
    }

    // pairwise interaction
    result += 0.5 * (first - second);
    return result;
}

double FM::predict(const std::vector<uint32_t> & indices, const std::vector<double> & weights) {
    std::vector<double> dummy;
    return _predict(indices, weights, dummy, false);
}

double FM::learn(const std::vector<uint32_t> & indices, const std::vector<double> & weights, double target) {
    /* Returns current guess before learning.
     */

    // guess target
    std::vector<double> vif_dot_wi(k);
    double guess = _predict(indices, weights, vif_dot_wi, true);

    double err = guess - target;

    // square loss
    //(target - guess) * grad

    // w0
    w0 -= learning_rate * err;
    for (int i = 0; i < indices.size(); ++i) {
        uint32_t index = indices[i];

        // wi
        double w = wi[index];
        wi[index] = w - learning_rate * (err * weights[i] + w_reg * w);
    }

    // vif
    for (int f = 0; f < k; ++f) {
        for (int i = 0; i < indices.size(); ++i) {
            uint32_t index = indices[i];

            double v = vi[index][f];
            vi[index][f] = v -  learning_rate * (err * (weights[i] * vif_dot_wi[f] - v * pow(weights[i], 2)) + v_reg * v);
        }
    }

    return guess;
}
