#include <iostream>
#include <algorithm>
#include <cmath>

#include "fm.h"

using namespace lightfm;

FM::FM(int k, std::default_random_engine e1)
    : e1(e1)
    , k(k)
{
    // FIXME weight initialization should be parameterized
    d = std::normal_distribution<>(0, 0.01);

    w0 = d(e1);
}

FM::~FM() {
}

double FM::predict(std::vector<int> indices, std::vector<double> weights) {
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
        while (index >= wi.size()) {
            // Unseen index. Initialize weights.
            wi.push_back(d(e1));

            std::vector<double> v(k);
            std::generate_n(v.begin(), k, [&]{ return d(e1);});

            vi.push_back(v);
        }

        // feature specific bias
        result += wi[index] * weights[i];

        for (auto vif: vi[index]) {
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

void FM::learn(std::vector<int> indices, std::vector<double> weights, double target) {
    double stepsize = 0.001;

    // guess target
    double guess = predict(indices, weights);

    double err = guess - target;
    /* std::cout << err << std::endl; */

    // udpate
    // square loss
    //(target - guess) * grad

    // w0
    /* std::cout << "w0 : " << w0; */
    w0 -= err * stepsize * 1.0;
    /* std::cout << " -> " << w0 << std::endl; */
    for (int i = 0; i < indices.size(); ++i) {
        int index = i;
        // wi
        /* std::cout << "w_" << index << " : " << wi[index]; */
        wi[index] -= err * stepsize * weights[i];
        /* std::cout << " -> " << wi[index] << std::endl; */
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
            /* std::cout << "v_" << index << f << " : " << vi[index][f]; */
            vi[index][f] -= err * stepsize * (weights[i] * tmp - vi[index][f] * pow(weights[i], 2));
            /* std::cout << " -> " << vi[index][f] << std::endl; */
        }
    }
}
