#ifndef LBVH_BVH_CUH
#define LBVH_BVH_CUH
#include "aabb.cuh"
#include "cone.cuh"
#include "morton_code.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace lbvh
{
    namespace detail
    {
        struct node
        {
            std::uint32_t parent_idx; // parent node
            std::uint32_t left_idx;   // index of left  child node
            std::uint32_t right_idx;  // index of right child node
            std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
        };

        // a set of pointers to use it on device.
        template <typename Real, unsigned int dim, typename Object, bool IsConst>
        struct basic_device_bvh;
        template <typename Real, unsigned int dim, typename Object>
        struct basic_device_bvh<Real, dim, Object, false>
        {
            using real_type = Real;
            using aabb_type = aabb<real_type, dim>;
            using cone_type = cone<real_type, dim>;
            using node_type = detail::node;
            using index_type = std::uint32_t;
            using object_type = Object;

            unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
            unsigned int num_objects; // (# of leaves), the same as the number of objects

            node_type *nodes;
            aabb_type *aabbs;
            cone_type *cones;
            object_type *objects;

            SNCH_LBVH_CALLABLE basic_device_bvh() : num_nodes{0}, num_objects{0}, nodes{nullptr}, aabbs{nullptr}, cones{nullptr}, objects{nullptr} {};
            SNCH_LBVH_CALLABLE basic_device_bvh(unsigned int num_nodes_, unsigned int num_objects_,
                             node_type *nodes_, aabb_type *aabbs_, cone_type *cones_, object_type *objects_)
                : num_nodes(num_nodes_),
                  num_objects(num_objects_),
                  nodes(nodes_),
                  aabbs(aabbs_),
                  cones(cones_),
                  objects(objects_) {}
            SNCH_LBVH_CALLABLE basic_device_bvh(const basic_device_bvh<Real, dim, Object, false> &other)
                : num_nodes(other.num_nodes),
                  num_objects(other.num_objects),
                  nodes(other.nodes),
                  aabbs(other.aabbs),
                  cones(other.cones),
                  objects(other.objects) {}
        };
        template <typename Real, unsigned int dim, typename Object>
        struct basic_device_bvh<Real, dim, Object, true>
        {
            using real_type = Real;
            using aabb_type = aabb<real_type, dim>;
            using cone_type = cone<real_type, dim>;
            using node_type = detail::node;
            using index_type = std::uint32_t;
            using object_type = Object;

            unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
            unsigned int num_objects; // (# of leaves), the same as the number of objects

            node_type const *nodes;
            aabb_type const *aabbs;
            cone_type const *cones;
            object_type const *objects;

            SNCH_LBVH_CALLABLE basic_device_bvh() : num_nodes{0}, num_objects{0}, nodes{nullptr}, aabbs{nullptr}, cones{nullptr}, objects{nullptr} {};
            SNCH_LBVH_CALLABLE basic_device_bvh(unsigned int num_nodes_, unsigned int num_objects_,
                             node_type const *nodes_, aabb_type const *aabbs_, cone_type const *cones_, object_type const *objects_)
                : num_nodes(num_nodes_),
                  num_objects(num_objects_),
                  nodes(nodes_),
                  aabbs(aabbs_),
                  cones(cones_),
                  objects(objects_) {}
            SNCH_LBVH_CALLABLE basic_device_bvh(const basic_device_bvh<Real, dim, Object, true> &other)
                : num_nodes(other.num_nodes),
                  num_objects(other.num_objects),
                  nodes(other.nodes),
                  aabbs(other.aabbs),
                  cones(other.cones),
                  objects(other.objects) {}
        };

        template <typename UInt>
        SNCH_LBVH_DEVICE_INLINE uint2 determine_range(UInt const *node_code, const unsigned int num_leaves, unsigned int idx)
        {
            if (idx == 0)
            {
                return make_uint2(0, num_leaves - 1);
            }

            // determine direction of the range
            const UInt self_code = node_code[idx];
            const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
            const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
            const int d = (R_delta > L_delta) ? 1 : -1;

            // Compute upper bound for the length of the range

            const int delta_min = thrust::min(L_delta, R_delta);
            int l_max = 2;
            int delta = -1;
            int i_tmp = idx + d * l_max;
            if (0 <= i_tmp && i_tmp < num_leaves)
            {
                delta = common_upper_bits(self_code, node_code[i_tmp]);
            }
            while (delta > delta_min)
            {
                l_max <<= 1;
                i_tmp = idx + d * l_max;
                delta = -1;
                if (0 <= i_tmp && i_tmp < num_leaves)
                {
                    delta = common_upper_bits(self_code, node_code[i_tmp]);
                }
            }

            // Find the other end by binary search
            int l = 0;
            int t = l_max >> 1;
            while (t > 0)
            {
                i_tmp = idx + (l + t) * d;
                delta = -1;
                if (0 <= i_tmp && i_tmp < num_leaves)
                {
                    delta = common_upper_bits(self_code, node_code[i_tmp]);
                }
                if (delta > delta_min)
                {
                    l += t;
                }
                t >>= 1;
            }
            unsigned int jdx = idx + l * d;
            if (d < 0)
            {
                thrust::swap(idx, jdx); // make it sure that idx < jdx
            }
            return make_uint2(idx, jdx);
        }

        template <typename UInt>
        SNCH_LBVH_DEVICE_INLINE unsigned int find_split(
            UInt const *node_code, const unsigned int num_leaves, const unsigned int first, const unsigned int last) noexcept
        {
            const UInt first_code = node_code[first];
            const UInt last_code = node_code[last];
            if (first_code == last_code)
            {
                return (first + last) >> 1;
            }
            const int delta_node = common_upper_bits(first_code, last_code);

            // binary search...
            int split = first;
            int stride = last - first;
            do
            {
                stride = (stride + 1) >> 1;
                const int middle = split + stride;
                if (middle < last)
                {
                    const int delta = common_upper_bits(first_code, node_code[middle]);
                    if (delta > delta_node)
                    {
                        split = middle;
                    }
                }
            } while (stride > 1);

            return split;
        }
        template <typename Real, unsigned int dim, typename Object, bool IsConst, typename UInt>
        void construct_internal_nodes(
            const basic_device_bvh<Real, dim, Object, IsConst> &self, UInt const *node_code, const unsigned int num_objects)
        {
            thrust::for_each(
                thrust::device, thrust::make_counting_iterator<unsigned int>(0),
                thrust::make_counting_iterator<unsigned int>(num_objects - 1),
                [self, node_code, num_objects] SNCH_LBVH_DEVICE(const unsigned int idx)
                {
                    self.nodes[idx].object_idx = 0xFFFFFFFF; //  internal nodes

                    const uint2 ij = determine_range(node_code, num_objects, idx);
                    const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

                    self.nodes[idx].left_idx = gamma;
                    self.nodes[idx].right_idx = gamma + 1;
                    if (thrust::min(ij.x, ij.y) == gamma)
                    {
                        self.nodes[idx].left_idx += num_objects - 1;
                    }
                    if (thrust::max(ij.x, ij.y) == gamma + 1)
                    {
                        self.nodes[idx].right_idx += num_objects - 1;
                    }
                    self.nodes[self.nodes[idx].left_idx].parent_idx = idx;
                    self.nodes[self.nodes[idx].right_idx].parent_idx = idx;
                    return;
                });
            return;
        }

    } // namespace detail

    template <typename Real, unsigned int dim, typename Object>
    struct default_morton_code_calculator
    {
        default_morton_code_calculator(aabb<Real, dim> w) : whole(w) {}
        default_morton_code_calculator() = default;
        ~default_morton_code_calculator() = default;
        default_morton_code_calculator(default_morton_code_calculator const &) = default;
        default_morton_code_calculator(default_morton_code_calculator &&) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator const &) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator &&) = default;

        SNCH_LBVH_CALLABLE unsigned int operator()(const Object &, const aabb<Real, dim> &box) noexcept
        {
            auto p = centroid(box);
            p.x -= whole.lower.x;
            p.y -= whole.lower.y;
            // p.z -= whole.lower.z;
            p.x /= (whole.upper.x - whole.lower.x);
            p.y /= (whole.upper.y - whole.lower.y);
            // p.z /= (whole.upper.z - whole.lower.z);
            return morton_code(p);
        }
        aabb<Real, dim> whole;
    };

    template <typename Real, typename Object>
    struct default_morton_code_calculator<Real, 2, Object>
    {
        default_morton_code_calculator(aabb<Real, 2> w) : whole(w) {}
        default_morton_code_calculator() = default;
        ~default_morton_code_calculator() = default;
        default_morton_code_calculator(default_morton_code_calculator const &) = default;
        default_morton_code_calculator(default_morton_code_calculator &&) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator const &) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator &&) = default;

        SNCH_LBVH_CALLABLE unsigned int operator()(const Object &, const aabb<Real, 2> &box) noexcept
        {
            auto p = centroid(box);
            p.x -= whole.lower.x;
            p.y -= whole.lower.y;
            p.x /= (whole.upper.x - whole.lower.x);
            p.y /= (whole.upper.y - whole.lower.y);
            return morton_code(p);
        }
        aabb<Real, 2> whole;
    };

    template <typename Real, typename Object>
    struct default_morton_code_calculator<Real, 3, Object>
    {
        default_morton_code_calculator(aabb<Real, 3> w) : whole(w) {}
        default_morton_code_calculator() = default;
        ~default_morton_code_calculator() = default;
        default_morton_code_calculator(default_morton_code_calculator const &) = default;
        default_morton_code_calculator(default_morton_code_calculator &&) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator const &) = default;
        default_morton_code_calculator &operator=(default_morton_code_calculator &&) = default;

        SNCH_LBVH_CALLABLE unsigned int operator()(const Object &, const aabb<Real, 3> &box) noexcept
        {
            auto p = centroid(box);
            p.x -= whole.lower.x;
            p.y -= whole.lower.y;
            p.z -= whole.lower.z;
            p.x /= (whole.upper.x - whole.lower.x);
            p.y /= (whole.upper.y - whole.lower.y);
            p.z /= (whole.upper.z - whole.lower.z);
            return morton_code(p);
        }
        aabb<Real, 3> whole;
    };

    template <typename Real, unsigned int dim, typename Object>
    using bvh_device = detail::basic_device_bvh<Real, dim, Object, false>;
    template <typename Real, unsigned int dim, typename Object>
    using cbvh_device = detail::basic_device_bvh<Real, dim, Object, true>;

    template <
        typename Real, unsigned int dim, typename Object, typename AABBGetter, typename ConeGetter,
        typename MortonCodeCalculator = default_morton_code_calculator<Real, dim, Object>>
    class bvh
    {
    public:
        using real_type = Real;
        using index_type = std::uint32_t;
        using object_type = Object;
        using aabb_type = aabb<real_type, dim>;
        using cone_type = cone<real_type, dim>;
        using node_type = detail::node;
        using aabb_getter_type = AABBGetter;
        using cone_getter_type = ConeGetter;
        using morton_code_calculator_type = MortonCodeCalculator;

    public:
        template <typename InputIterator>
        bvh(InputIterator first, InputIterator last, bool query_host_enabled = false)
            : objects_h_(first, last), objects_d_(objects_h_), query_host_enabled_(query_host_enabled)
        {
            this->construct();
        }

        bvh() = default;
        ~bvh() = default;
        bvh(const bvh &) = default;
        bvh(bvh &&) = default;
        bvh &operator=(const bvh &) = default;
        bvh &operator=(bvh &&) = default;

        bool query_host_enabled() const noexcept { return query_host_enabled_; }
        bool &query_host_enabled() noexcept { return query_host_enabled_; }

        void clear()
        {
            this->objects_h_.clear();
            this->objects_d_.clear();
            this->aabbs_h_.clear();
            this->aabbs_.clear();
            this->cones_h_.clear();
            this->cones_.clear();
            this->nodes_h_.clear();
            this->nodes_.clear();
            return;
        }

        template <typename InputIterator>
        void assign(InputIterator first, InputIterator last)
        {
            this->objects_h_.assign(first, last);
            this->objects_d_ = this->objects_h_;
            this->construct();
            return;
        }

        bvh_device<real_type, dim, object_type> get_device_repr() noexcept
        {
            return bvh_device<real_type, dim, object_type>{
                static_cast<unsigned int>(nodes_.size()), static_cast<unsigned int>(objects_d_.size()), nodes_.data().get(),
                aabbs_.data().get(), cones_.data().get(), objects_d_.data().get()};
        }
        cbvh_device<real_type, dim, object_type> get_device_repr() const noexcept
        {
            return cbvh_device<real_type, dim, object_type>{
                static_cast<unsigned int>(nodes_.size()), static_cast<unsigned int>(objects_d_.size()), nodes_.data().get(),
                aabbs_.data().get(), cones_.data().get(), objects_d_.data().get()};
        }

        void construct()
        {
            assert(objects_h_.size() == objects_d_.size());
            if (objects_h_.size() == 0u)
            {
                return;
            }

            const unsigned int num_objects = objects_h_.size();
            const unsigned int num_internal_nodes = num_objects - 1;
            const unsigned int num_nodes = num_objects * 2 - 1;

            // --------------------------------------------------------------------
            // calculate morton code of each points

            const auto inf = std::numeric_limits<real_type>::infinity();
            aabb_type default_aabb;
            default_aabb.upper.x = -inf;
            default_aabb.lower.x = inf;
            default_aabb.upper.y = -inf;
            default_aabb.lower.y = inf;
            if constexpr (dim == 3)
            {
                default_aabb.upper.z = -inf;
                default_aabb.lower.z = inf;
            }

            cone_type default_cone;
            if constexpr (dim == 2)
            {
                default_cone.axis = make_float2(0.0f, 0.0f);
            }
            else if constexpr (dim == 3)
            {
                default_cone.axis = make_float3(0.0f, 0.0f, 0.0f);
            }
            default_cone.half_angle = M_PI;
            default_cone.radius = 0.0f;

            this->aabbs_.resize(num_nodes, default_aabb);
            this->cones_.resize(num_nodes, default_cone);

            // Initialize leaf nodes.
            thrust::transform(
                this->objects_d_.begin(), this->objects_d_.end(), aabbs_.begin() + num_internal_nodes, aabb_getter_type());
            thrust::transform(
                this->objects_d_.begin(), this->objects_d_.end(), cones_.begin() + num_internal_nodes, cone_getter_type());

            // Compute aabb_whole for morton code computation.
            const auto aabb_whole = thrust::reduce(
                aabbs_.begin() + num_internal_nodes, aabbs_.end(), default_aabb,
                [] SNCH_LBVH_HOST_DEVICE(const aabb_type &lhs, const aabb_type &rhs)
                { return merge(lhs, rhs); });

            // Compute morton code for each aabbs.
            // The object vector only provides span length, and does not participate in morton code computation.
            thrust::device_vector<unsigned int> morton(num_objects);
            thrust::transform(
                this->objects_d_.begin(), this->objects_d_.end(), aabbs_.begin() + num_internal_nodes, morton.begin(),
                morton_code_calculator_type(aabb_whole));

            // --------------------------------------------------------------------
            // sort object-indices by morton code

            // Use indices to refer to objects.
            thrust::device_vector<unsigned int> object_indices(num_objects);
            // Fill up indices vector with 0, 1, ..., num_objects - 1.
            thrust::copy(
                thrust::make_counting_iterator<index_type>(0), thrust::make_counting_iterator<index_type>(num_objects),
                object_indices.begin());
            // keep indices ascending order
            // Reorder aabbs as well as indices by morton codes.
            thrust::stable_sort_by_key(
                morton.begin(), morton.end(),
                thrust::make_zip_iterator(thrust::make_tuple(cones_.begin() + num_internal_nodes, aabbs_.begin() + num_internal_nodes, object_indices.begin())));

            // --------------------------------------------------------------------
            // check morton codes are unique

            thrust::device_vector<unsigned long long int> morton64(num_objects);
            const auto uniqued = thrust::unique_copy(morton.begin(), morton.end(), morton64.begin());

            // Use 64-bit morton codes if collision happens.
            const bool morton_code_is_unique = (morton64.end() == uniqued);
            if (!morton_code_is_unique)
            {
                std::cout << "Morton code collision detected." << std::endl;
                thrust::transform(
                    morton.begin(), morton.end(), object_indices.begin(), morton64.begin(),
                    [] SNCH_LBVH_DEVICE(const unsigned int m, const unsigned int idx)
                    {
                        unsigned long long int m64 = m;
                        m64 <<= 32;
                        m64 |= idx;
                        return m64;
                    });
            }

            // --------------------------------------------------------------------
            // construct leaf nodes and aabbs

            node_type default_node;
            default_node.parent_idx = 0xFFFFFFFF;
            default_node.left_idx = 0xFFFFFFFF;
            default_node.right_idx = 0xFFFFFFFF;
            default_node.object_idx = 0xFFFFFFFF;
            this->nodes_.resize(num_nodes, default_node);

            // Initialize leaf nodes.
            thrust::transform(
                object_indices.begin(), object_indices.end(), this->nodes_.begin() + num_internal_nodes,
                [] SNCH_LBVH_DEVICE(const index_type idx)
                {
                    node_type n;
                    n.parent_idx = 0xFFFFFFFF;
                    n.left_idx = 0xFFFFFFFF;
                    n.right_idx = 0xFFFFFFFF;
                    n.object_idx = idx;
                    return n;
                });

            // --------------------------------------------------------------------
            // construct internal nodes

            const auto self = this->get_device_repr();
            if (morton_code_is_unique)
            {
                // This line does not mean constructing from root, it just passes a pointer from where construct_internal_nodes can begin.
                const unsigned int *node_code = morton.data().get();
                detail::construct_internal_nodes(self, node_code, num_objects);
            }
            else // 64bit version
            {
                const unsigned long long int *node_code = morton64.data().get();
                detail::construct_internal_nodes(self, node_code, num_objects);
            }

            // --------------------------------------------------------------------
            // create AABB for each node by bottom-up strategy

            thrust::device_vector<int> flag_container(num_internal_nodes, 0);
            auto flags = flag_container.data().get();

            thrust::for_each(
                thrust::device, thrust::make_counting_iterator<index_type>(num_internal_nodes),
                thrust::make_counting_iterator<index_type>(num_nodes),
                [self, flags] SNCH_LBVH_DEVICE(index_type idx)
                {
                    unsigned int parent = self.nodes[idx].parent_idx;
                    while (parent != 0xFFFFFFFF) // means idx == 0
                    {
                        const int old = atomicCAS(flags + parent, 0, 1);
                        if (old == 0)
                        {
                            // this is the first thread entered here.
                            // wait the other thread from the other child node.
                            return;
                        }
                        assert(old == 1);
                        // here, the flag has already been 1. it means that this
                        // thread is the 2nd thread. merge AABB of both childlen.

                        const auto lidx = self.nodes[parent].left_idx;
                        const auto ridx = self.nodes[parent].right_idx;
                        const auto lbox = self.aabbs[lidx];
                        const auto rbox = self.aabbs[ridx];
                        self.aabbs[parent] = merge(lbox, rbox);

                        __threadfence();

                        // look the next parent...
                        parent = self.nodes[parent].parent_idx;
                    }
                    return;
                });

            // --------------------------------------------------------------------
            // create BoundingCone for each node by bottom-up strategy

            flag_container.clear();
            flag_container.resize(num_internal_nodes, 0);
            flags = flag_container.data().get();

            thrust::for_each(
                thrust::device, thrust::make_counting_iterator<index_type>(num_internal_nodes),
                thrust::make_counting_iterator<index_type>(num_nodes),
                [self, flags] SNCH_LBVH_DEVICE(index_type idx)
                {
                    unsigned int parent = self.nodes[idx].parent_idx;
                    while (parent != 0xFFFFFFFF) // means idx == 0
                    {
#ifdef __CUDACC__
                        const int old = atomicCAS(flags + parent, 0, 1);
#else
                        int old;
#pragma omp critical
                        {
                            old = *(flags + parent);
                            if (old == 0)
                                *(flags + parent) = 1;
                        }
#endif
                        if (old == 0)
                        {
                            // this is the first thread entered here.
                            // wait the other thread from the other child node.
                            return;
                        }
                        assert(old == 1);
                        // here, the flag has already been 1. it means that this
                        // thread is the 2nd thread. merge AABB of both childlen.

                        const auto lidx = self.nodes[parent].left_idx;
                        const auto ridx = self.nodes[parent].right_idx;
                        const auto lcone = self.cones[lidx];
                        const auto rcone = self.cones[ridx];
                        const auto lbox = self.aabbs[lidx];
                        const auto rbox = self.aabbs[ridx];
                        self.cones[parent] = merge(lcone, rcone, centroid(lbox), centroid(rbox), centroid(self.aabbs[parent]));

                        // look the next parent...
                        parent = self.nodes[parent].parent_idx;
                    }
                    return;
                });

            if (this->query_host_enabled_)
            {
                aabbs_h_ = aabbs_;
                nodes_h_ = nodes_;
                cones_h_ = cones_;
            }
            return;
        }

        thrust::host_vector<object_type> const &objects_host() const noexcept { return objects_h_; }
        thrust::host_vector<object_type> &objects_host() noexcept { return objects_h_; }
        thrust::host_vector<node_type> const &nodes_host() const noexcept { return nodes_h_; }
        thrust::host_vector<node_type> &nodes_host() noexcept { return nodes_h_; }
        thrust::host_vector<aabb_type> const &aabbs_host() const noexcept { return aabbs_h_; }
        thrust::host_vector<aabb_type> &aabbs_host() noexcept { return aabbs_h_; }
        thrust::host_vector<aabb_type> const &cones_host() const noexcept { return cones_h_; }
        thrust::host_vector<aabb_type> &cones_host() noexcept { return cones_h_; }

    private:
        thrust::host_vector<object_type> objects_h_;
        thrust::device_vector<object_type> objects_d_;
        thrust::host_vector<aabb_type> aabbs_h_;
        thrust::device_vector<aabb_type> aabbs_;
        thrust::host_vector<cone_type> cones_h_;
        thrust::device_vector<cone_type> cones_;
        thrust::host_vector<node_type> nodes_h_;
        thrust::device_vector<node_type> nodes_;
        bool query_host_enabled_;
    };

} // namespace lbvh
#endif // LBVH_BVH_CUH
