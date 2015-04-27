#include <iostream>
#include <algorithm>
#include <cmath>

#include "fm.h"

using namespace lightfm;

FM::FM(int dim, int k, std::default_random_engine e1)
    : e1(e1)
    , k(k)
    , dim(dim)
{
    // FIXME weight initialization should be parameterized
    d = std::normal_distribution<>(0, 0.01);

    w0 = d(e1);
    std::vector<double> w(dim);
    std::generate_n(w.begin(), dim, [&]{ return d(e1);});
    wi = std::move(w);

    for (int i = 0; i < dim; ++i) {
        std::vector<double> v(k);
        std::generate_n(v.begin(), k, [&]{ return d(e1);});

        vi.push_back(v);
    }
}

FM::~FM() {
}

double FM::predict(const std::vector<int> & indices, const std::vector<double> & weights) {
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
        int index = indices[i];

        // feature specific bias
        result += wi[index] * weights[i];

        for (auto& vif: vi[index]) {
            second += pow(vif, 2) * pow(weights[i], 2);
        }
    }

    for (int f = 0; f < k; ++f) {
        double tmp = 0.0;
        for (int i = 0; i < indices.size(); ++i) {
            int index = indices[i];
            tmp += vi[index][f] * weights[i];
        }
        first += pow(tmp, 2);
    }

    // pairwise interaction
    result += (0.5 * first) - (0.5 * second);
    return result;
}

double FM::learn(const std::vector<int> & indices, const std::vector<double> & weights, double target) {
    /* Returns current guess before learning.
     */
    // FIXME should be parameterized.
    double stepsize = 0.01;
    double reg = 0.01;

    // guess target
    double guess = predict(indices, weights);

    double err = guess - target;

    // square loss
    //(target - guess) * grad

    // w0
    w0 -= stepsize * err * 1.0;
    for (int i = 0; i < indices.size(); ++i) {
        int index = indices[i];

        // wi
        wi[index] -= stepsize * (err * weights[i] + reg * wi[index]);
    }

    // vif
    for (int f = 0; f < k; ++f) {
        for (int i = 0; i < indices.size(); ++i) {
            int index = indices[i];
            double tmp = 0.0;
            for (int j = 0; j < indices.size(); ++j) {
                // FIXME this term can be precomputed when guessing
                tmp += vi[indices[j]][f] * weights[j];
            }
            vi[index][f] -= stepsize * (err * (weights[i] * tmp - vi[index][f] * pow(weights[i], 2)) + reg * vi[index][f]);
        }
    }

    return guess;
}
