#include "featurizer.h"

using namespace lightfm;

Featurizer::Featurizer()
    : num(0)
{}

Featurizer::~Featurizer() {
}

int Featurizer::size() {
    return this->num;
}

int Featurizer::add_feature(std::string feature) {
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


int Featurizer::get_feature_index(std::string feature) {
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

std::vector<int> Featurizer::get_feature_indices(std::vector<std::string> features) {
    /* Return indices for given features. If an unseen string is met, it is
     * added to `index_table`.
     */
    std::vector<int> indices;
    for (auto const& x: features) {
        indices.push_back(get_feature_index(x));
    }
    return indices;
}
