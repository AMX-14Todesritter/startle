#ifndef CUDA_PARSIMONY_HPP
#define CUDA_PARSIMONY_HPP
#include <string>
#include <vector>
#include "flat_parsimony.hpp"
namespace starhomoplasy {
bool cuda_parsimony_available(std::string* reason = nullptr);
std::vector<double> cuda_score_nni_candidates(const flat_tree&, const std::vector<flat_nni_move>&);
}
#endif
