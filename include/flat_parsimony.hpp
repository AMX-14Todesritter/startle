#ifndef FLAT_PARSIMONY_HPP
#define FLAT_PARSIMONY_HPP

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "digraph.hpp"
#include "starhomoplasy.hpp"

namespace starhomoplasy {
    struct flat_nni_move {
        int u;
        int w;
        int v;
        int z;
    };

    class flat_tree {
    public:
        static flat_tree from_digraph(
            const digraph<star_homoplasy_data>& tree,
            const std::map<std::string, std::map<int, double>>& mutation_priors,
            const character_state_matrix& matrix,
            int root = 0
        );

        double score() const;
        void apply_nni(const flat_nni_move& move);

        size_t num_nodes() const { return parent_.size(); }
        size_t num_characters() const { return num_characters_; }

    private:
        int root_ = 0;
        size_t num_characters_ = 0;
        std::vector<int> parent_;
        std::vector<int> child_a_;
        std::vector<int> child_b_;
        std::vector<int> initial_states_;
        std::vector<size_t> prior_offsets_;
        std::vector<double> prior_values_;

        double prior(size_t character, int state) const;
    };

    std::vector<flat_nni_move> enumerate_nni_moves(
        const digraph<star_homoplasy_data>& tree
    );
}

#endif
