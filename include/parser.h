#pragma once
#include <vector>
#include <string>

#include "featurizer.h"

namespace lightfm {
    class Parser {
        public:
            Parser(Featurizer & featurizer);
            ~Parser();
            void read_data(const std::string filename, std::vector<double> & targets, std::vector<std::vector<uint32_t>> & feature_indices, std::vector<std::vector<double>> & feature_weights);

        private:
            Featurizer& feat;
    };
}
