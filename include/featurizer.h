#pragma once

#include <map>
#include <vector>
#include <string>

namespace lightfm
{
    class Featurizer {
        public:
            Featurizer();
            ~Featurizer();

            int size();
            int add_feature(std::string feature);
            int get_feature_index(std::string features);
            std::vector<int> get_feature_indices(const std::vector<std::string> & features);
        private:
            int num;
            std::map<std::string, int> index_table;
    };
}
