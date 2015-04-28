#include <iostream>
#include <algorithm>
#include <cmath>

#include "fm.h"
#include "featurizer.h"

using namespace lightfm;

FM::FM(int dim, int k, Featurizer & feat, std::default_random_engine e1)
    : e1(e1)
    , k(k)
    , feat(feat)
{
    // FIXME weight initialization should be parameterized
    d = std::normal_distribution<>(0, 0.01);

    w0 = d(e1);
    std::vector<double> w(dim);
    std::generate_n(w.begin(), dim, [&]{ return d(e1);});
    wi = std::move(w);

    for (int i = 0; i < k; ++i) {
        v_s.push_back("v" + std::to_string(i));
    }
}

FM::~FM() {
}

double FM::_predict(const std::vector<std::string> & features, const std::vector<double> & weights, std::vector<double> & vif_dot_wi, bool precompute) {
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

    for (int i = 0; i < features.size(); ++i) {
        uint32_t index = feat.get_feature_index(features[i]);

        // feature specific bias
        result += wi[index] * weights[i];
    }

    for (int f = 0; f < k; ++f) {
        double tmp = 0.0;
        for (int i = 0; i < features.size(); ++i) {
            uint32_t index = feat.get_feature_index(features[i] + v_s[f]);
            tmp += wi[index] * weights[i];

            second += pow(wi[index], 2) * pow(weights[i], 2);
        }
        if (precompute) {
            vif_dot_wi[f] = tmp;
        }
        first += pow(tmp, 2);
    }

    // for (int f = 0; f < k; ++f) {
    //     double tmp = 0.0;
    //     for (int i = 0; i < features.size(); ++i) {
    //         uint32_t index = feat.get_feature_index(features[i]);
    //         tmp += vi[index][f] * weights[i];

    //         second += pow(vi[index][f], 2) * pow(weights[i], 2);
    //     }

    //     if (precompute) {
    //         vif_dot_wi[f] = tmp;
    //     }

    //     first += pow(tmp, 2);
    // }

    // pairwise interaction
    result += 0.5 * (first - second);
    return result;
}

double FM::predict(const std::vector<std::string> & features, const std::vector<double> & weights) {
    std::vector<double> dummy;
    return _predict(features, weights, dummy, false);
}

double FM::learn(const std::vector<std::string> & features, const std::vector<double> & weights, double target) {
    /* Returns current guess before learning.
     */
    // FIXME should be parameterized.
    double stepsize = 0.01;
    double reg = 0.01;

    // guess target
    std::vector<double> vif_dot_wi(k);
    double guess = _predict(features, weights, vif_dot_wi, true);

    double err = guess - target;

    // square loss
    //(target - guess) * grad

    // w0
    w0 -= stepsize * err;
    for (int i = 0; i < features.size(); ++i) {
        uint32_t index = feat.get_feature_index(features[i]);

        // wi
        double w = wi[index];
        wi[index] = w - stepsize * (err * weights[i] + reg * w);
    }

    for (int f = 0; f < k; ++f) {
        for (int i = 0; i < features.size(); ++i) {
            uint32_t index = feat.get_feature_index(features[i] + v_s[f]);
            double v = wi[index];
            wi[index] = v - stepsize * (err * (weights[i] * vif_dot_wi[f] - v * pow(weights[i], 2)) + reg * v);
        }
    }

    // vif
    // for (int f = 0; f < k; ++f) {
    //     for (int i = 0; i < indices.size(); ++i) {
    //         uint32_t index = indices[i];
    //         // vif_dot_wi contains the precomputed values for the following
    //         // double tmp = 0.0;
    //         // for (int j = 0; j < indices.size(); ++j) {
    //         //     tmp += vi[indices[j]][f] * weights[j];
    //         // }
    //         // tmp == vif_dot_wi[f];

    //         double v = vi[index][f];
    //         vi[index][f] = v -  stepsize * (err * (weights[i] * vif_dot_wi[f] - v * pow(weights[i], 2)) + reg * v);
    //     }
    // }

    return guess;
}
