#pragma once

#include <map>
#include <vector>
#include <string>

#include "hash.h"

namespace lightfm
{
    class Featurizer {
        public:
            virtual int size() = 0;
            virtual uint32_t get_feature_index(std::string features) = 0;
            virtual std::vector<uint32_t> get_feature_indices(const std::vector<std::string> & features) = 0;
    };

    class DictFeaturizer: public Featurizer {
        public:
            DictFeaturizer();
            int size();
            int add_feature(std::string feature);

            uint32_t get_feature_index(std::string features);
            std::vector<uint32_t> get_feature_indices(const std::vector<std::string> & features);

        private:
            int num;
            std::map<std::string, uint32_t> index_table;
    };

    class HashFeaturizer: public Featurizer {
        public:
            HashFeaturizer(int bit, int seed);

            int size();
            uint32_t get_feature_index(std::string features);
            std::vector<uint32_t> get_feature_indices(const std::vector<std::string> & features);

        private:
            int bit;
            int seed;
            int mask;
    };
}
