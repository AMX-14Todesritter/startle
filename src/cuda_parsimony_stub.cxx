#include "cuda_parsimony.hpp"
#include <stdexcept>
namespace starhomoplasy {
bool cuda_parsimony_available(std::string* reason) {
    if (reason) *reason = "binary built without STARTLE_ENABLE_CUDA";
    return false;
}
std::vector<double> cuda_score_nni_candidates(const flat_tree&, const std::vector<flat_nni_move>&) {
    throw std::runtime_error("CUDA parsimony requested from a CPU-only build");
}
}
