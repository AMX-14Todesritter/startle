#include "flat_parsimony.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace starhomoplasy {
    namespace {
        void replace_child(std::vector<int>& child_a, std::vector<int>& child_b,
                           int parent, int old_child, int new_child) {
            if (child_a.at(parent) == old_child) {
                child_a.at(parent) = new_child;
                return;
            }
            if (child_b.at(parent) == old_child) {
                child_b.at(parent) = new_child;
                return;
            }
            throw std::invalid_argument("NNI move references a non-child vertex");
        }
    }

    flat_tree flat_tree::from_digraph(
        const digraph<star_homoplasy_data>& tree,
        const std::map<std::string, std::map<int, double>>& mutation_priors,
        const character_state_matrix& matrix,
        int root
    ) {
        flat_tree result;
        result.root_ = root;
        result.num_characters_ = matrix.num_characters;

        const auto nodes = tree.nodes();
        const size_t node_count = nodes.size();
        result.parent_.assign(node_count, -1);
        result.child_a_.assign(node_count, -1);
        result.child_b_.assign(node_count, -1);
        result.initial_states_.assign(node_count * matrix.num_characters, -1);

        for (int node : nodes) {
            const auto& children = tree.successors(node);
            if (children.size() > 2) {
                throw std::invalid_argument("flat_tree requires a binary tree");
            }

            auto child_it = children.begin();
            if (child_it != children.end()) {
                result.child_a_.at(node) = *child_it++;
            }
            if (child_it != children.end()) {
                result.child_b_.at(node) = *child_it;
            }

            if (tree[node].data.character_states.size() != matrix.num_characters) {
                throw std::invalid_argument("node character-state width does not match matrix");
            }
            std::copy(
                tree[node].data.character_states.begin(),
                tree[node].data.character_states.end(),
                result.initial_states_.begin() + static_cast<size_t>(node) * matrix.num_characters
            );

            for (int child : children) {
                if (result.parent_.at(child) != -1) {
                    throw std::invalid_argument("flat_tree input contains a node with multiple parents");
                }
                result.parent_.at(child) = node;
            }
        }

        if (root < 0 || static_cast<size_t>(root) >= node_count || result.parent_.at(root) != -1) {
            throw std::invalid_argument("invalid flat_tree root");
        }

        result.prior_offsets_.reserve(matrix.num_characters + 1);
        result.prior_offsets_.push_back(0);
        for (const std::string& character : matrix.characters) {
            const auto& character_priors = mutation_priors.at(character);
            int max_state = 0;
            for (const auto& [state, value] : character_priors) {
                (void)value;
                if (state < 0) {
                    throw std::invalid_argument("mutation prior state must be non-negative");
                }
                max_state = std::max(max_state, state);
            }

            const size_t width = static_cast<size_t>(max_state) + 1;
            const size_t offset = result.prior_values_.size();
            result.prior_values_.resize(
                offset + width,
                std::numeric_limits<double>::quiet_NaN()
            );
            for (const auto& [state, value] : character_priors) {
                result.prior_values_.at(offset + static_cast<size_t>(state)) = value;
            }
            result.prior_offsets_.push_back(result.prior_values_.size());
        }

        return result;
    }

    double flat_tree::prior(size_t character, int state) const {
        if (state < 0 || character >= num_characters_) {
            throw std::out_of_range("invalid character state for mutation prior");
        }
        const size_t offset = prior_offsets_.at(character);
        const size_t width = prior_offsets_.at(character + 1) - offset;
        if (static_cast<size_t>(state) >= width) {
            throw std::out_of_range("mutation prior state is not present");
        }
        const double value = prior_values_.at(offset + static_cast<size_t>(state));
        if (std::isnan(value)) {
            throw std::out_of_range("mutation prior state is not present");
        }
        return value;
    }

    double flat_tree::score() const {
        const size_t node_count = parent_.size();
        std::vector<int> states = initial_states_;
        std::vector<double> scores(node_count, 0.0);
        std::vector<std::pair<int, bool>> stack;
        std::vector<int> postorder;
        stack.emplace_back(root_, false);

        while (!stack.empty()) {
            const auto [node, visited] = stack.back();
            stack.pop_back();
            if (visited) {
                postorder.push_back(node);
                continue;
            }

            stack.emplace_back(node, true);
            if (child_b_.at(node) != -1) {
                stack.emplace_back(child_b_.at(node), false);
            }
            if (child_a_.at(node) != -1) {
                stack.emplace_back(child_a_.at(node), false);
            }
        }

        if (postorder.size() != node_count) {
            throw std::invalid_argument("flat_tree topology is disconnected or cyclic");
        }

        for (int node : postorder) {
            const int child_a = child_a_.at(node);
            const int child_b = child_b_.at(node);
            if (child_a == -1 && child_b == -1) {
                continue;
            }
            if (child_a == -1 || child_b == -1) {
                throw std::invalid_argument("flat_tree contains a unary internal node");
            }

            const size_t node_offset = static_cast<size_t>(node) * num_characters_;
            const size_t a_offset = static_cast<size_t>(child_a) * num_characters_;
            const size_t b_offset = static_cast<size_t>(child_b) * num_characters_;

            for (size_t character = 0; character < num_characters_; ++character) {
                const int a_state = states.at(a_offset + character);
                const int b_state = states.at(b_offset + character);
                int parent_state = 0;
                if (a_state != 0 && b_state != 0) {
                    if (a_state == -1) {
                        parent_state = b_state;
                    } else if (b_state == -1 || a_state == b_state) {
                        parent_state = a_state;
                    }
                }
                states.at(node_offset + character) = parent_state;

                if (a_state != -1 && a_state != parent_state) {
                    scores.at(node) += prior(character, a_state);
                }
                if (b_state != -1 && b_state != parent_state) {
                    scores.at(node) += prior(character, b_state);
                }
            }

            scores.at(node) += scores.at(child_a) + scores.at(child_b);
        }

        const size_t root_offset = static_cast<size_t>(root_) * num_characters_;
        for (size_t character = 0; character < num_characters_; ++character) {
            const int state = states.at(root_offset + character);
            if (state != 0 && state != -1) {
                scores.at(root_) += prior(character, state);
            }
        }
        return scores.at(root_);
    }

    void flat_tree::apply_nni(const flat_nni_move& move) {
        if (parent_.at(move.v) != move.u || parent_.at(move.w) != move.u ||
            parent_.at(move.z) != move.v) {
            throw std::invalid_argument("invalid rooted NNI move");
        }
        replace_child(child_a_, child_b_, move.u, move.w, move.z);
        replace_child(child_a_, child_b_, move.v, move.z, move.w);
        parent_.at(move.w) = move.v;
        parent_.at(move.z) = move.u;
    }

    std::vector<flat_nni_move> enumerate_nni_moves(
        const digraph<star_homoplasy_data>& tree
    ) {
        std::vector<flat_nni_move> moves;
        for (const auto& [u, v] : tree.edges()) {
            if (tree.successors(v).empty()) {
                continue;
            }

            std::vector<int> siblings;
            for (int child : tree.successors(u)) {
                if (child != v) {
                    siblings.push_back(child);
                }
            }
            if (siblings.size() != 1 || tree.successors(v).size() != 2) {
                throw std::invalid_argument("NNI enumeration requires a binary tree");
            }

            for (int z : tree.successors(v)) {
                moves.push_back(flat_nni_move{u, siblings.front(), v, z});
            }
        }

        std::sort(moves.begin(), moves.end(), [](const flat_nni_move& a, const flat_nni_move& b) {
            return std::tie(a.u, a.v, a.w, a.z) < std::tie(b.u, b.v, b.w, b.z);
        });
        return moves;
    }
}
