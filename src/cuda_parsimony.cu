#include "cuda_parsimony.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace starhomoplasy {
namespace {
void check(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
template<class T> struct buffer {
    T* p = nullptr; size_t n = 0;
    void ensure(size_t count) {
        if (count <= n) return;
        if (p) check(cudaFree(p), "cudaFree");
        check(cudaMalloc(&p, count*sizeof(T)), "cudaMalloc");
        n=count;
    }
    ~buffer() { if (p) cudaFree(p); }
    size_t bytes(size_t count) const { return count*sizeof(T); }
};
struct cuda_cache {
    buffer<int> ca,cb,order,subtree_start,subtree_end,initial,work;
    buffer<flat_nni_move> moves;
    buffer<unsigned long long> offsets;
    buffer<double> priors,scores;
};
std::vector<int> make_postorder(const flat_tree& t) {
    std::vector<int> out;
    std::vector<std::pair<int,bool>> stack{{t.root(),false}};
    while (!stack.empty()) {
        auto [u, seen] = stack.back(); stack.pop_back();
        if (seen) { out.push_back(u); continue; }
        stack.emplace_back(u,true);
        if (t.child_b().at(u) != -1) stack.emplace_back(t.child_b().at(u),false);
        if (t.child_a().at(u) != -1) stack.emplace_back(t.child_a().at(u),false);
    }
    if (out.size() != t.num_nodes()) throw std::invalid_argument("invalid candidate topology");
    return out;
}
__device__ int swapped_order_node(int position, const int* order,
 const int* subtree_start, const int* subtree_end, flat_nni_move move) {
    int a=subtree_start[move.w], b=subtree_end[move.w];
    int c=subtree_start[move.z], d=subtree_end[move.z];
    if (c<a) { int x=a; a=c; c=x; x=b; b=d; d=x; }
    int first_len=b-a, middle_len=c-b, second_len=d-c;
    if (position<a || position>=d) return order[position];
    int relative=position-a;
    if (relative<second_len) return order[c+relative];
    relative-=second_len;
    if (relative<middle_len) return order[b+relative];
    return order[a+(relative-middle_len)];
}
__device__ int moved_child(int child, int node, flat_nni_move move) {
    if (node==move.u && child==move.w) return move.z;
    if (node==move.v && child==move.z) return move.w;
    return child;
}
__global__ void score_kernel(const int* ca, const int* cb, const int* order,
 const int* subtree_start, const int* subtree_end, const flat_nni_move* moves,
 const int* initial, const unsigned long long* offsets, const double* priors,
 int* work, double* scores, int count, int nodes, int chars, int root) {
    int c = blockIdx.x;
    if (c >= count) return;
    const flat_nni_move move=moves[c];
    double sum = 0.0;
    for (int ch = threadIdx.x; ch < chars; ch += blockDim.x) {
        int* state = work + ((size_t)c*chars + ch)*nodes;
        for (int u=0; u<nodes; ++u) state[u] = initial[u*chars+ch];
        for (int pos=0; pos<nodes; ++pos) {
            int u=swapped_order_node(pos,order,subtree_start,subtree_end,move);
            int a=moved_child(ca[u],u,move), b=moved_child(cb[u],u,move);
            if (a == -1) continue;
            int as=state[a], bs=state[b], ps=0;
            if (as != 0 && bs != 0) {
                if (as == -1) ps=bs;
                else if (bs == -1 || as == bs) ps=as;
            }
            state[u]=ps;
            size_t off=offsets[ch];
            if (as != -1 && as != ps) sum += priors[off+as];
            if (bs != -1 && bs != ps) sum += priors[off+bs];
        }
        int rs=state[root];
        if (rs != 0 && rs != -1) sum += priors[offsets[ch]+rs];
    }
    atomicAdd(scores+c,sum);
}
}
bool cuda_parsimony_available(std::string* reason) {
    int count=0; cudaError_t e=cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        if (reason) *reason = e == cudaSuccess ? "no visible CUDA device" : cudaGetErrorString(e);
        return false;
    }
    return true;
}
std::vector<double> cuda_score_nni_candidates(const flat_tree& base,
 const std::vector<flat_nni_move>& moves) {
    if (moves.empty()) return {};
    std::string why;
    if (!cuda_parsimony_available(&why)) throw std::runtime_error("CUDA unavailable: "+why);
    size_t nc=moves.size(), nn=base.num_nodes(), nch=base.num_characters();
    const auto order=make_postorder(base);
    std::vector<int> subtree_start(nn), subtree_end(nn);
    for (size_t pos=0; pos<nn; ++pos) {
        const int node=order[pos];
        subtree_end[node]=static_cast<int>(pos+1);
        const int a=base.child_a()[node], b=base.child_b()[node];
        subtree_start[node]=a==-1 ? static_cast<int>(pos)
            : std::min(subtree_start[a],subtree_start[b]);
    }
    std::vector<unsigned long long> offsets(base.prior_offsets().begin(),base.prior_offsets().end());
    std::vector<double> scores(nc,0.0);
    thread_local cuda_cache cache;
    constexpr size_t workspace_budget_bytes=256ULL*1024ULL*1024ULL;
    const size_t bytes_per_candidate=nch*nn*sizeof(int);
    const size_t tile_capacity=std::max<size_t>(
        1,std::min(nc,workspace_budget_bytes/bytes_per_candidate));
    const size_t tile_size=std::min(nc,tile_capacity);
    cache.ca.ensure(nn); cache.cb.ensure(nn); cache.order.ensure(nn);
    cache.subtree_start.ensure(nn); cache.subtree_end.ensure(nn);
    cache.initial.ensure(base.initial_states().size()); cache.work.ensure(tile_size*nch*nn);
    cache.moves.ensure(tile_size);
    cache.offsets.ensure(offsets.size()); cache.priors.ensure(base.prior_values().size());
    cache.scores.ensure(tile_size);
#define H2D(dst,src) check(cudaMemcpy((dst).p,(src).data(),(dst).bytes((src).size()),cudaMemcpyHostToDevice),"cudaMemcpy")
    H2D(cache.ca,base.child_a()); H2D(cache.cb,base.child_b()); H2D(cache.order,order);
    H2D(cache.subtree_start,subtree_start); H2D(cache.subtree_end,subtree_end);
    H2D(cache.initial,base.initial_states());
    H2D(cache.offsets,offsets); H2D(cache.priors,base.prior_values());
#undef H2D
    for (size_t begin=0; begin<nc; begin+=tile_capacity) {
        const size_t count=std::min(tile_capacity,nc-begin);
        check(cudaMemcpy(cache.moves.p,moves.data()+begin,cache.moves.bytes(count),
                         cudaMemcpyHostToDevice),"cudaMemcpy moves");
        check(cudaMemset(cache.scores.p,0,cache.scores.bytes(count)),"cudaMemset");
        score_kernel<<<(unsigned)count,128>>>(cache.ca.p,cache.cb.p,cache.order.p,
          cache.subtree_start.p,cache.subtree_end.p,cache.moves.p,cache.initial.p,
          cache.offsets.p,cache.priors.p,cache.work.p,cache.scores.p,
          (int)count,(int)nn,(int)nch,base.root());
        check(cudaGetLastError(),"score_kernel launch");
        check(cudaMemcpy(scores.data()+begin,cache.scores.p,cache.scores.bytes(count),
                         cudaMemcpyDeviceToHost),"cudaMemcpy scores");
    }
    return scores;
}
}
