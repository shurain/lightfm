#pragma once
#include <vector>
#include <random>

namespace lightfm {
    class FM {
        public:
            FM(int dim, int k, std::default_random_engine e1);
            ~FM();
            double learn(const std::vector<uint32_t> & indices, const std::vector<double> & weights, double target);
            double predict(const std::vector<uint32_t> & indices, const std::vector<double> & weights);

            // FIXME model params should be visible but this feel uncomfortable
            double w0;  // global bias
            std::vector<double> wi;  // feature specific bias
            std::vector<std::vector<double>> vi;  // pairwise interaction vector

        private:
            double _predict(const std::vector<uint32_t> & indices, const std::vector<double> & weights, std::vector<double> & vif_dot_wi, bool precompute);
            std::default_random_engine e1;
            std::normal_distribution<> d;

            // dimension of factorization
            int k;
    };
}
