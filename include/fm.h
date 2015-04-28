#pragma once
#include <vector>
#include <random>

#include "featurizer.h"

namespace lightfm {
    class FM {
        public:
            FM(int dim, int k, Featurizer& feat, std::default_random_engine e1);
            ~FM();
            double learn(const std::vector<std::string> & features, const std::vector<double> & weights, double target);
            double predict(const std::vector<std::string> & features, const std::vector<double> & weights);

            // FIXME model params should be visible but this feel uncomfortable
            double w0;  // global bias
            std::vector<double> wi;  // feature specific bias

        private:
            double _predict(const std::vector<std::string> & features, const std::vector<double> & weights, std::vector<double> & vif_dot_wi, bool precompute);
            std::default_random_engine e1;
            std::normal_distribution<> d;

            // dimension of factorization
            int k;
            Featurizer& feat;
            std::vector<std::string> v_s;
    };
}
