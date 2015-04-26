#pragma once
#include <vector>
#include <random>

namespace lightfm {
    class FM {
        public:
            FM(int k, std::default_random_engine e1);
            ~FM();
            double learn(const std::vector<int> & indices, const std::vector<double> & weights, double target);
            double predict(const std::vector<int> & indices, const std::vector<double> & weights);

            // FIXME model params should be visible but this feel uncomfortable
            double w0;  // global bias
            std::vector<double> wi;  // feature specific bias
            std::vector<std::vector<double>> vi;  // pairwise interaction vector

        private:
            std::default_random_engine e1;
            std::normal_distribution<> d;

            // dimension of factorization
            int k;
    };
}
