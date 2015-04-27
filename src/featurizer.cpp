#include <cmath>

#include "featurizer.h"
#include "hash.h"

using namespace lightfm;

DictFeaturizer::DictFeaturizer()
    : num(0)
{}

int DictFeaturizer::size() {
    return this->num;
}

int DictFeaturizer::add_feature(std::string feature) {
    /* Assign an index to a string. If the given string already exists,
     * silently ignore it. Returns the total number of indices maintained.
     */
    auto const& it = index_table.find(feature);
    if (it != index_table.end()) {
        return it->second;
    }
    index_table[feature] = num;
    num++;
    return num - 1;
}


uint32_t DictFeaturizer::get_feature_index(std::string feature) {
    /* Return the index for a given feature. If an unseen string is met, it is
     * added to `index_table`.
     */
    auto const& it = index_table.find(feature);
    if (it == index_table.end()) {
        int index = add_feature(feature);
        return index;
    }
    else {
        return it->second;
    }
}

int HashFeaturizer::size() {
    return pow(2, this->bit);
}

std::vector<uint32_t> DictFeaturizer::get_feature_indices(const std::vector<std::string> & features) {
    /* Return indices for given features. If an unseen string is met, it is
     * added to `index_table`.
     */
    std::vector<uint32_t> indices;
    for (auto const& x: features) {
        indices.push_back(get_feature_index(x));
    }
    return indices;
}


HashFeaturizer::HashFeaturizer(int bit, int seed)
    :bit(bit)
    , seed(seed)
{
    mask = (1 << bit) - 1;
}

uint32_t HashFeaturizer::get_feature_index(std::string feature) {
    return uniform_hash(feature.c_str(), feature.size(), seed) & mask;
}


std::vector<uint32_t> HashFeaturizer::get_feature_indices(const std::vector<std::string> & features) {
    std::vector<uint32_t> indices;
    for (auto const& x: features) {
        indices.push_back(get_feature_index(x));
    }
    return indices;
}
