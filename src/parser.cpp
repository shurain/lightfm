#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "parser.h"
#include "featurizer.h"

using namespace lightfm;

Parser::Parser(Featurizer & featurizer)
    : feat(featurizer)
{}

Parser::~Parser() {
}

void Parser::read_data(const std::string filename, std::vector<double> & targets, std::vector<std::vector<uint32_t>> & feature_indices, std::vector<std::vector<double>> & feature_weights) {
    std::ifstream data (filename, std::ifstream::in);

    std::string line;

    if (data.is_open()) {
        while(getline(data, line)) {
            try {
                double target = stod(line);
                targets.push_back(target);
            }
            catch (std::invalid_argument except) {
                std::cerr << "Target not float." << std::endl;
                abort();
            }

            std::vector<uint32_t> feature_index;
            std::vector<double> feature_weight;

            std::string::size_type pos = line.find(' ') + 1;
            while (true) {
                std::string::size_type next = line.find(' ', pos);
                std::string::size_type split = line.find(':', pos);

                if (split > next) {
                    std::cerr << "Error" << std::endl;
                    abort();
                }

                std::string feature = line.substr(pos, split-pos);
                double val = stof(line.substr(split+1));

                feature_index.push_back(feat.get_feature_index(feature));
                feature_weight.push_back(val);

                pos = next + 1;

                if (next == std::string::npos) {
                    break;
                }
            }
            feature_indices.push_back(feature_index);
            feature_weights.push_back(feature_weight);
        }
    }
}
