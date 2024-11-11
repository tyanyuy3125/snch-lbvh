#pragma once

#include "lbvh.cuh"
#include <memory>
#include <vector>
#include <exception>
#include <map>
#include <utility>

namespace lbvh
{
    constexpr float bvh_offset = 0.1f;

    __device__ __host__ inline float find_closest_point_triangle(const float3 &pa, const float3 &pb, const float3 &pc,
                                                                 const float3 &x, float3 *pt, float2 *t)
    {
        // source: real time collision detection
        // check if x in vertex region outside pa
        float3 ab = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        float3 ac = make_float3(pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);
        float3 ax = make_float3(x.x - pa.x, x.y - pa.y, x.z - pa.z);
        float d1 = dot(ab, ax);
        float d2 = dot(ac, ax);
        if (d1 <= 0.0f && d2 <= 0.0f)
        {
            // barycentric coordinates (1, 0, 0)
            get(*t, 0) = 1.0f;
            get(*t, 1) = 0.0f;
            *pt = pa;
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // check if x in vertex region outside pb
        float3 bx = make_float3(x.x - pb.x, x.y - pb.y, x.z - pb.z);
        float d3 = dot(ab, bx);
        float d4 = dot(ac, bx);
        if (d3 >= 0.0f && d4 <= d3)
        {
            // barycentric coordinates (0, 1, 0)
            get(*t, 0) = 0.0f;
            get(*t, 1) = 1.0f;
            *pt = pb;
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // check if x in vertex region outside pc
        float3 cx = make_float3(x.x - pc.x, x.y - pc.y, x.z - pc.z);
        float d5 = dot(ab, cx);
        float d6 = dot(ac, cx);
        if (d6 >= 0.0f && d5 <= d6)
        {
            // barycentric coordinates (0, 0, 1)
            get(*t, 0) = 0.0f;
            get(*t, 1) = 0.0f;
            *pt = pc;
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // check if x in edge region of ab, if so return projection of x onto ab
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        {
            // barycentric coordinates (1 - v, v, 0)
            float v = d1 / (d1 - d3);
            get(*t, 0) = 1.0f - v;
            get(*t, 1) = v;
            *pt = make_float3(pa.x + ab.x * v, pa.y + ab.y * v, pa.z + ab.z * v);
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // check if x in edge region of ac, if so return projection of x onto ac
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        {
            // barycentric coordinates (1 - w, 0, w)
            float w = d2 / (d2 - d6);
            get(*t, 0) = 1.0f - w;
            get(*t, 1) = 0.0f;
            *pt = make_float3(pa.x + ac.x * w, pa.y + ac.y * w, pa.z + ac.z * w);
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // check if x in edge region of bc, if so return projection of x onto bc
        float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
        {
            // barycentric coordinates (0, 1 - w, w)
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            get(*t, 0) = 0.0f;
            get(*t, 1) = 1.0f - w;
            *pt = make_float3(pb.x + (pc.x - pb.x) * w, pb.y + (pc.y - pb.y) * w, pb.z + (pc.z - pb.z) * w);
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        // x inside face region. Compute pt through its barycentric coordinates (u, v, w)
        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;
        get(*t, 0) = 1.0f - v - w;
        get(*t, 1) = v;

        *pt = make_float3(pa.x + ab.x * v + ac.x * w, pa.y + ab.y * v + ac.y * w, pa.z + ab.z * v + ac.z * w);
        return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
    }

    __device__ __host__ inline bool is_silhouette_vertex(const float2 &n0, const float2 &n1, const float2 &view_dir,
                                                         float d, bool flip_normal_orientation)
    {
        const float precision = 1e-3f; // TODO
        float sign = flip_normal_orientation ? 1.0f : -1.0f;

        if (d <= precision)
        {
            float det = n0.x * n1.y - n0.y * n1.x;
            return sign * det > precision;
        }

        float2 view_dir_unit = make_float2(view_dir.x / d, view_dir.y / d);
        float dot0 = dot(view_dir_unit, n0);
        float dot1 = dot(view_dir_unit, n1);

        bool is_zero_dot0 = std::fabs(dot0) <= precision;
        if (is_zero_dot0)
        {
            return sign * dot1 > precision;
        }

        bool is_zero_dot1 = std::fabs(dot1) <= precision;
        if (is_zero_dot1)
        {
            return sign * dot0 > precision;
        }

        return dot0 * dot1 < 0.0f;
    }

    __device__ __host__ inline bool is_silhouette_edge(const float3 &pa, const float3 &pb,
                                                       const float3 &n0, const float3 &n1, const float3 &view_dir,
                                                       float d, bool flip_normal_orientation)
    {
        const float precision = 1e-3f; // TODO
        float sign = flip_normal_orientation ? 1.0f : -1.0f;

        if (d <= precision)
        {
            float3 edge_dir = normalize(make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z));
            float signed_dihedral_angle = std::atan2(dot(edge_dir, cross(n0, n1)), dot(n0, n1));
            return sign * signed_dihedral_angle > precision;
        }

        float3 view_dir_unit = make_float3(view_dir.x, view_dir.y, view_dir.z);
        float dot0 = dot(view_dir_unit, n0);
        float dot1 = dot(view_dir_unit, n1);

        bool is_zero_dot0 = std::fabs(dot0) <= precision;
        if (is_zero_dot0)
        {
            return sign * dot1 > precision;
        }

        bool is_zero_dot1 = std::fabs(dot1) <= precision;
        if (is_zero_dot1)
        {
            return sign * dot0 > precision;
        }

        return dot0 * dot1 < 0.0f;
    }

    __device__ __host__ inline float find_closest_point_line_segment(const float2 &pa, const float2 &pb,
                                                                     const float2 &x, float2 *pt, float *t)
    {
        float2 u = make_float2(pb.x - pa.x, pb.y - pa.y);
        float2 v = make_float2(x.x - pa.x, x.y - pa.y);

        float c1 = dot(u, v);
        if (c1 <= 0.0f)
        {
            *pt = pa;
            *t = 0.0f;
            return length(make_float2(x.x - pt->x, x.y - pt->y));
        }

        float c2 = dot(u, u);
        if (c2 <= c1)
        {
            *pt = pb;
            *t = 1.0f;
            return length(make_float2(x.x - pt->x, x.y - pt->y));
        }

        *t = c1 / c2;
        *pt = make_float2(pa.x + u.x * (*t), pa.y + u.y * (*t));
        return length(make_float2(x.x - pt->x, x.y - pt->y));
    }

    __device__ __host__ inline double find_closest_point_line_segment(const double2 &pa, const double2 &pb,
                                                                      const double2 &x, double2 *pt, double *t)
    {
        double2 u = make_double2(pb.x - pa.x, pb.y - pa.y);
        double2 v = make_double2(x.x - pa.x, x.y - pa.y);

        double c1 = dot(u, v);
        if (c1 <= 0.0f)
        {
            *pt = pa;
            *t = 0.0f;
            return length(make_double2(x.x - pt->x, x.y - pt->y));
        }

        double c2 = dot(u, u);
        if (c2 <= c1)
        {
            *pt = pb;
            *t = 1.0f;
            return length(make_double2(x.x - pt->x, x.y - pt->y));
        }

        *t = c1 / c2;
        *pt = make_double2(pa.x + u.x * (*t), pa.y + u.y * (*t));
        return length(make_double2(x.x - pt->x, x.y - pt->y));
    }

    __device__ __host__ inline float find_closest_point_line_segment(const float3 &pa, const float3 &pb,
                                                                     const float3 &x, float3 *pt, float *t)
    {
        float3 u = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        float3 v = make_float3(x.x - pa.x, x.y - pa.y, x.z - pa.z);

        float c1 = dot(u, v);
        if (c1 <= 0.0f)
        {
            *pt = pa;
            *t = 0.0f;
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        float c2 = dot(u, u);
        if (c2 <= c1)
        {
            *pt = pb;
            *t = 1.0f;
            return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        *t = c1 / c2;
        *pt = make_float3(pa.x + u.x * (*t), pa.y + u.y * (*t), pa.z + u.z * (*t));
        return length(make_float3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
    }

    __device__ __host__ inline double find_closest_point_line_segment(const double3 &pa, const double3 &pb,
                                                                      const double3 &x, double3 *pt, double *t)
    {
        double3 u = make_double3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
        double3 v = make_double3(x.x - pa.x, x.y - pa.y, x.z - pa.z);

        double c1 = dot(u, v);
        if (c1 <= 0.0f)
        {
            *pt = pa;
            *t = 0.0f;
            return length(make_double3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        double c2 = dot(u, u);
        if (c2 <= c1)
        {
            *pt = pb;
            *t = 1.0f;
            return length(make_double3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
        }

        *t = c1 / c2;
        *pt = make_double3(pa.x + u.x * (*t), pa.y + u.y * (*t), pa.z + u.z * (*t));
        return length(make_double3(x.x - pt->x, x.y - pt->y, x.z - pt->z));
    }

    template <unsigned int dim>
    class scene;

    template <>
    class scene<2>
    {
    public:
        struct silhouette_vertex
        {
            int4 indices;
            const float2 *vertices;

            __device__ __host__ silhouette_vertex(const int4 indices, const thrust::device_vector<float2> &vertices_d)
                : indices(indices), vertices(vertices_d.data().get())
            {
            }
            __device__ __host__ silhouette_vertex(const thrust::device_vector<float2> &vertices_d)
                : indices(make_int4(-1, -1, -1, -1)), vertices(vertices_d.data().get())
            {
            }
            __device__ __host__ silhouette_vertex()
                : indices(make_int4(-1, -1, -1, -1)), vertices(nullptr)
            {
            }

            __device__ __host__ aabb<float, 2> bounding_box() const
            {
                const float2 &p = vertices[get(indices, 1)];
                aabb<float, 2> ret;
                ret.upper = make_float2(p.x + bvh_offset, p.y + bvh_offset);
                ret.lower = make_float2(p.x - bvh_offset, p.y - bvh_offset);
                return ret;
            }
            __device__ __host__ float2 centroid() const
            {
                const float2 &p = vertices[get(indices, 1)];
                return p;
            }
            __device__ __host__ bool has_face(int f_index) const
            {
                return f_index == 0 ? get(indices, 2) != -1 : get(indices, 0) != -1;
            }
            __device__ __host__ float2 normal(int f_index, bool do_normalize = true) const
            {
                int i = f_index == 0 ? 1 : 0;
                const float2 &pa = vertices[get(indices, i + 0)];
                const float2 &pb = vertices[get(indices, i + 1)];

                float2 s = make_float2(pb.x - pa.x, pb.y - pa.y);
                float2 n = make_float2(s.y, -s.x);

                return do_normalize ? normalize(n) : n;
            }
            __device__ __host__ float2 normal() const
            {
                // TODO: read normal from geometry data

                float2 n = make_float2(0.0f, 0.0f);
                if (has_face(0))
                {
                    auto n0 = normal(0, false);
                    n = make_float2(n.x + n0.x, n.y + n0.y);
                }
                if (has_face(1))
                {
                    auto n1 = normal(1, false);
                    n = make_float2(n.x + n1.x, n.y + n1.y);
                }
                return normalize(n);
            }
            __device__ __host__ bool find_closest_silhouette_point(const float2 origin, const float max_radius_squared, float &distance,
                                                                   const bool flip_normal_orientation, const float min_radius_squared) const
            {
                if (min_radius_squared >= max_radius_squared)
                {
                    return false;
                }

                const float2 &p = vertices[get(indices, 1)];
                float2 view_dir = make_float2(origin.x - p.x, origin.y - p.y);
                float d = length(view_dir);
                if (d * d > max_radius_squared)
                {
                    return false;
                }

                bool is_silhouette = !has_face(0) || !has_face(1);
                if (!is_silhouette)
                {
                    const float2 n0 = normal(0);
                    const float2 n1 = normal(1);
                    is_silhouette = is_silhouette_vertex(n0, n1, view_dir, d, flip_normal_orientation);
                }

                if (is_silhouette && d * d <= max_radius_squared)
                {
                    distance = d;

                    return true;
                }

                return false;
            }
        };

        struct line_segment
        {
            int2 vertex_indices;
            int2 silhouette_indices;
            const float2 *vertices;
            const silhouette_vertex *silhouettes;
            __device__ __host__ line_segment(const int2 vertex_indices, const int2 silhouette_indices, const thrust::device_vector<float2> &vertices_d, const thrust::device_vector<silhouette_vertex> &silhouettes_d)
                : vertex_indices(vertex_indices), silhouette_indices(silhouette_indices), vertices(vertices_d.data().get()), silhouettes(silhouettes_d.data().get())
            {
            }
        };

        struct aabb_getter
        {
            __device__ __host__ lbvh::aabb<float, 2> operator()(const line_segment &ls) const noexcept
            {
                lbvh::aabb<float, 2> ret;
                auto vertices = ls.vertices;

                const float2 p0 = vertices[ls.vertex_indices.x];
                const float2 p1 = vertices[ls.vertex_indices.y];
                ret.upper = make_float2(max(p0.x, p1.x) + bvh_offset, max(p0.y, p1.y) + bvh_offset);
                ret.lower = make_float2(min(p0.x, p1.x) - bvh_offset, min(p0.y, p1.y) - bvh_offset);
                return ret;
            }
        };

        struct cone_getter
        {
            __device__ __host__ lbvh::cone<float, 2> operator()(const line_segment &ls) const noexcept
            {
                auto aabb = aabb_getter()(ls);
                auto aabb_centroid = centroid(aabb);
                lbvh::cone<float, 2> ret;
                ret.axis = make_float2(0.0f, 0.0f);
                ret.half_angle = M_PI;
                ret.radius = 0.0f;
                bool any_silhouettes = false;
                bool silhouettes_all_have_two_adjacent_faces = true;
                for (int i = 0; i < 2; ++i)
                {
                    int silhouette_index = get(ls.silhouette_indices, i);
                    if (silhouette_index != -1)
                    {
                        const auto &silhouette = ls.silhouettes[silhouette_index];
                        auto silhouette_normal = silhouette.normal();
                        ret.axis.x += silhouette_normal.x;
                        ret.axis.y += silhouette_normal.y;
                        auto silhouette_centroid = silhouette.centroid();
                        ret.radius = std::max(ret.radius, length(make_float2(silhouette_centroid.x - aabb_centroid.x, silhouette_centroid.y - aabb_centroid.y)));
                        silhouettes_all_have_two_adjacent_faces = silhouettes_all_have_two_adjacent_faces && silhouette.has_face(0) && silhouette.has_face(1);
                        any_silhouettes = true;
                    }
                }

                if (!any_silhouettes)
                {
                    ret.half_angle = -M_PI;
                }
                else if (!silhouettes_all_have_two_adjacent_faces)
                {
                    ret.half_angle = M_PI;
                }
                else
                {
                    float axis_norm = length(ret.axis);
                    if (axis_norm > epsilon<float>())
                    {
                        ret.axis.x /= axis_norm;
                        ret.axis.y /= axis_norm;
                        ret.half_angle = 0.0f;
                        for (int i = 0; i < 2; ++i)
                        {
                            int silhouette_index = get(ls.silhouette_indices, i);
                            if (silhouette_index != -1)
                            {
                                const auto &silhouette = ls.silhouettes[silhouette_index];
                                bool has_face_0 = silhouette.has_face(0);
                                bool has_face_1 = silhouette.has_face(1);
                                float2 silhouette_face_normals[2];
                                silhouette_face_normals[0] = has_face_0 ? silhouette.normal(0, true) : make_float2(0.0f, 0.0f);
                                silhouette_face_normals[1] = has_face_1 ? silhouette.normal(1, true) : make_float2(0.0f, 0.0f);
                                for (int j = 0; j < 2; ++j)
                                {
                                    const float2 &n = silhouette_face_normals[j];
                                    float angle = std::acos(std::max(-1.0f, std::min(1.0f, dot(ret.axis, n))));
                                    ret.half_angle = std::max(ret.half_angle, angle);
                                }
                            }
                        }
                    }
                }
                return ret;
            }
        };

        struct distance_calculator
        {
            __device__ __host__ float operator()(const float2 point, const line_segment &object) const noexcept
            {
                auto vertices = object.vertices;
                const float2 pa = vertices[object.vertex_indices.x];
                const float2 pb = vertices[object.vertex_indices.y];
                float2 pt;
                float t;
                return find_closest_point_line_segment(pa, pb, point, &pt, &t);
            }
        };

        struct silhouette_distance_calculator
        {
            __device__ __host__ bool operator()(const float2 origin, const line_segment &object, const float max_radius_squared, float &distance,
                                                const bool flip_normal_orientation, const float min_radius_squared) const noexcept
            {
                const int2 silhouette_indices = object.silhouette_indices;
                float max_radius_squared_detached = max_radius_squared;
                bool ret = false;
                for (int i = 0; i < 2; ++i)
                {
                    const int silhouette_index = get(silhouette_indices, i);
                    if (silhouette_index != -1)
                    {
                        const silhouette_vertex &silhouette_vertex = object.silhouettes[silhouette_index];
                        bool found = silhouette_vertex.find_closest_silhouette_point(origin, max_radius_squared_detached, distance, flip_normal_orientation, min_radius_squared);
                        if (found)
                        {
                            ret = true;
                            max_radius_squared_detached = distance * distance;
                        }
                    }
                }
                return ret;
            }
        };

        struct intersect_test
        {
            __device__ __host__ thrust::pair<bool, float> operator()(const Line<float, 2> &line, const line_segment &object) const noexcept
            {
                float2 p0 = object.vertices[object.vertex_indices.x];
                float2 p1 = object.vertices[object.vertex_indices.y];

                float2 seg_dir = {p1.x - p0.x, p1.y - p0.y};

                float D = line.dir.x * (-seg_dir.y) + line.dir.y * seg_dir.x;

                if (fabs(D) < epsilon<float>())
                {
                    return thrust::make_pair(false, 0.0f);
                }

#ifdef __CUDA_ARCH__
                auto invD = __frcp_rn(D);
#else
                auto invD = 1.0f / D;
#endif

                float t = ((p0.x - line.origin.x) * (-seg_dir.y) - (p0.y - line.origin.y) * (-seg_dir.x)) * invD;
                float s = (line.dir.x * (p0.y - line.origin.y) - line.dir.y * (p0.x - line.origin.x)) * invD;

                if (s >= -1e-3f && s <= 1.0f + 1e-3f && t >= 0.0f) // TODO: standardize epsilon
                {
                    return thrust::make_pair(true, t);
                }
                else
                {
                    return thrust::make_pair(false, 0.0f);
                }
            }
        };

        scene<2>() = default;
        template <typename VerticesInputIterator, typename IndicesInputIterator>
        scene<2>(VerticesInputIterator vertices_first, VerticesInputIterator vertices_last, IndicesInputIterator indices_first, IndicesInputIterator indices_last)
            : vertices_h(vertices_first, vertices_last), vertices_d(vertices_h), indices_h(indices_first, indices_last)
        {
        }

        void compute_silhouettes()
        {
            silhouettes_h.clear();
            silhouette_vertex default_silhouette_vertex(vertices_d);
            silhouettes_h.resize(vertices_h.size(), default_silhouette_vertex);

            for (const auto &line_indices : indices_h)
            {
                silhouette_vertex &sv1 = silhouettes_h[line_indices.x];
                get(sv1.indices, 1) = line_indices.x;
                get(sv1.indices, 2) = line_indices.y;
                sv1.vertices = vertices_d.data().get();

                silhouette_vertex &sv2 = silhouettes_h[line_indices.y];
                get(sv2.indices, 0) = line_indices.x;
                get(sv2.indices, 1) = line_indices.y;
                sv2.vertices = vertices_d.data().get();
            }

            silhouettes_d.resize(silhouettes_h.size());
            silhouettes_d = silhouettes_h;
        }
        void build_bvh()
        {
            std::unordered_map<int, bool> seen_silhouettes;
            for (auto it = indices_h.begin(); it != indices_h.end(); std::advance(it, 1))
            {
                const auto &line_indices = *it;
                int2 silhouette_indices = make_int2(-1, -1);
                int silhouette_indices_ptr = 0;
                for (int i = 0; i < 2; ++i)
                {
                    int v_index = get(line_indices, i);
                    if (!seen_silhouettes[v_index])
                    {
                        seen_silhouettes[v_index] = true;
                        get(silhouette_indices, silhouette_indices_ptr++) = v_index;
                    }
                }
                lines.push_back(line_segment((*it), silhouette_indices, vertices_d, silhouettes_d));
            }

            p_bvh = std::make_unique<lbvh::bvh<float, 2, line_segment, aabb_getter, cone_getter>>(lines.begin(), lines.end(), true); // TODO: expose query_host_enabled.
            bvh_dev = p_bvh->get_device_repr();
        }
        const auto &get_bvh_device_ptr() const
        {
            if (p_bvh)
            {
                return bvh_dev;
            }
            else
            {
                throw std::runtime_error("BVH is not built yet.");
            }
        }

    private:
        thrust::host_vector<float2> vertices_h;
        thrust::device_vector<float2> vertices_d;

        thrust::host_vector<int2> indices_h;

        std::vector<line_segment> lines;

        thrust::host_vector<silhouette_vertex> silhouettes_h;
        thrust::device_vector<silhouette_vertex> silhouettes_d;

        std::unique_ptr<lbvh::bvh<float, 2, line_segment, aabb_getter, cone_getter>> p_bvh;
        lbvh::bvh_device<float, 2, line_segment> bvh_dev;
    };

    template <>
    class scene<3>
    {
    public:
        struct silhouette_edge
        {
            int4 indices;
            const float3 *vertices;

            __device__ __host__ silhouette_edge(const int4 indices, const thrust::device_vector<float3> &vertices_d)
                : indices(indices), vertices(vertices_d.data().get())
            {
            }
            __device__ __host__ silhouette_edge(const thrust::device_vector<float3> &vertices_d)
                : indices(make_int4(-1, -1, -1, -1)), vertices(vertices_d.data().get())
            {
            }
            __device__ __host__ silhouette_edge()
                : indices(make_int4(-1, -1, -1, -1)), vertices(nullptr)
            {
            }

            __device__ __host__ aabb<float, 3> bounding_box() const
            {
                const float4 &pa = vec3_to_vec4(vertices[get(indices, 1)]);
                const float3 &pb = vertices[get(indices, 2)];

                aabb<float, 3> box(pa);
                expand_to_include(&box, pb);
                return box;
            }
            __device__ __host__ float3 centroid() const
            {
                const float3 &pa = vertices[get(indices, 1)];
                const float3 &pb = vertices[get(indices, 2)];

                return make_float3((pa.x + pb.x) / 2, (pa.y + pb.y) / 2, (pa.z + pb.z) / 2);
            }
            __device__ __host__ bool has_face(int f_index) const
            {
                return f_index == 0 ? get(indices, 3) != -1 : get(indices, 0) != -1;
            }
            __device__ __host__ float3 normal(int f_index, bool do_normalize = true) const
            {
                int i, j, k;
                if (f_index == 0)
                {
                    i = 3;
                    j = 1;
                    k = 2;
                }
                else
                {
                    i = 0;
                    j = 2;
                    k = 1;
                }

                const float3 &pa = vertices[get(indices, j)];
                const float3 &pb = vertices[get(indices, k)];
                const float3 &pc = vertices[get(indices, i)];

                float3 v1 = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
                float3 v2 = make_float3(pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);

                float3 n = cross(v1, v2);
                return do_normalize ? normalize(n) : n;
            }
            __device__ __host__ float3 normal() const
            {
                float3 n = make_float3(0.0f, 0.0f, 0.0f);
                if (has_face(0))
                {
                    auto n0 = normal(0, false);
                    n = make_float3(n.x + n0.x, n.y + n0.y, n.z + n0.z);
                }
                if (has_face(1))
                {
                    auto n1 = normal(1, false);
                    n = make_float3(n.x + n1.x, n.y + n1.y, n.z + n1.z);
                }
                return normalize(n);
            }
            __device__ __host__ bool find_closest_silhouette_point(const float3 origin, const float max_radius_squared, float &distance,
                                                                   const bool flip_normal_orientation, const float min_radius_squared) const
            {
                if (min_radius_squared >= max_radius_squared)
                {
                    return false;
                }

                const float3 &pa = vertices[get(indices, 1)];
                const float3 &pb = vertices[get(indices, 2)];
                float3 closest_pos;
                float t;
                float d = find_closest_point_line_segment(pa, pb, origin, &closest_pos, &t);

                if (d * d > max_radius_squared)
                {
                    return false;
                }

                bool is_silhouette = !has_face(0) || !has_face(1);
                if (!is_silhouette)
                {
                    float3 n0 = normal(0);
                    float3 n1 = normal(1);
                    float3 view_dir = make_float3(origin.x - closest_pos.x, origin.y - closest_pos.y, origin.z - closest_pos.z);
                    is_silhouette = is_silhouette_edge(pa, pb, n0, n1, view_dir, d, flip_normal_orientation);
                }

                if (is_silhouette && d * d <= max_radius_squared)
                {
                    distance = d;

                    return true;
                }

                return false;
            }
        };

        struct triangle
        {
            int3 vertex_indices;
            int3 silhouette_indices;
            const float3 *vertices;
            const silhouette_edge *silhouettes;

            __device__ __host__ triangle(const int3 vertex_indices,
                                         const int3 silhouette_indices,
                                         const thrust::device_vector<float3> &vertices_d,
                                         const thrust::device_vector<silhouette_edge> &silhouettes_d)
                : vertex_indices(vertex_indices), silhouette_indices(silhouette_indices), vertices(vertices_d.data().get()), silhouettes(silhouettes_d.data().get())
            {
            }
        };

        struct aabb_getter
        {
            __device__ __host__ lbvh::aabb<float, 3> operator()(const triangle &object) const noexcept
            {
                // TODO: use float4 for all member variables.
                const float4 &pa = vec3_to_vec4(object.vertices[get(object.vertex_indices, 0)]);
                const float3 &pb = object.vertices[get(object.vertex_indices, 1)];
                const float3 &pc = object.vertices[get(object.vertex_indices, 2)];

                lbvh::aabb<float, 3> box(pa);
                expand_to_include(&box, pb);
                expand_to_include(&box, pc);

                return box;
            }
        };

        struct cone_getter
        {
            __device__ __host__ lbvh::cone<float, 3> operator()(const triangle &object) const noexcept
            {
                auto aabb = aabb_getter()(object);
                auto aabb_centroid = centroid(aabb);
                lbvh::cone<float, 3> ret;
                ret.axis = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                ret.half_angle = M_PI;
                ret.radius = 0.0f;
                bool any_silhouettes = false;
                bool silhouettes_all_have_two_adjacent_faces = true;
                for (int i = 0; i < 3; ++i)
                {
                    int silhouette_index = get(object.silhouette_indices, i);
                    if (silhouette_index != -1)
                    {
                        const auto &silhouette = object.silhouettes[silhouette_index];
                        auto silhouette_normal = silhouette.normal();
                        ret.axis.x += silhouette_normal.x;
                        ret.axis.y += silhouette_normal.y;
                        ret.axis.z += silhouette_normal.z;
                        auto silhouette_centroid = silhouette.centroid();
                        ret.radius = std::max(
                            ret.radius,
                            length(make_float3(
                                silhouette_centroid.x - aabb_centroid.x,
                                silhouette_centroid.y - aabb_centroid.y,
                                silhouette_centroid.z - aabb_centroid.z)));
                        silhouettes_all_have_two_adjacent_faces = silhouettes_all_have_two_adjacent_faces && silhouette.has_face(0) && silhouette.has_face(1);
                        any_silhouettes = true;
                    }
                }

                if (!any_silhouettes)
                {
                    ret.half_angle = -M_PI;
                }
                else if (!silhouettes_all_have_two_adjacent_faces)
                {
                    ret.half_angle = M_PI;
                }
                else
                {
                    float axis_norm = length(ret.axis);
                    if (axis_norm > epsilon<float>())
                    {
                        ret.axis.x /= axis_norm;
                        ret.axis.y /= axis_norm;
                        ret.axis.z /= axis_norm;
                        ret.half_angle = 0.0f;
                        for (int i = 0; i < 3; ++i)
                        {
                            int silhouette_index = get(object.silhouette_indices, i);
                            if (silhouette_index != -1)
                            {
                                const auto &silhouette = object.silhouettes[silhouette_index];
                                bool has_face_0 = silhouette.has_face(0);
                                bool has_face_1 = silhouette.has_face(1);
                                float3 silhouette_face_normals[2];
                                silhouette_face_normals[0] = has_face_0 ? silhouette.normal(0, true) : make_float3(0.0f, 0.0f, 0.0f);
                                silhouette_face_normals[1] = has_face_1 ? silhouette.normal(1, true) : make_float3(0.0f, 0.0f, 0.0f);
                                for (int j = 0; j < 2; ++j)
                                {
                                    const float3 &n = silhouette_face_normals[j];
                                    float angle = std::acos(std::max(-1.0f, std::min(1.0f, dot(vec4_to_vec3(ret.axis), n))));
                                    ret.half_angle = std::max(ret.half_angle, angle);
                                }
                            }
                        }
                    }
                }
                return ret;
            }
        };

        struct distance_calculator
        {
            __device__ __host__ float operator()(const float4 point, const triangle &object) const noexcept
            {
                const float3 &pa = object.vertices[object.vertex_indices.x];
                const float3 &pb = object.vertices[object.vertex_indices.y];
                const float3 &pc = object.vertices[object.vertex_indices.z];

                float3 pt;
                float2 t;
                float d = find_closest_point_triangle(pa, pb, pc, vec4_to_vec3(point), &pt, &t);
                return d;
            }
        };

        struct silhouette_distance_calculator
        {
            __device__ __host__ bool operator()(const float4 origin, const triangle &object, const float max_radius_squared, float &distance,
                                                const bool flip_normal_orientation, const float min_radius_squared) const noexcept
            {
                // TODO: Implement
                const int3 silhouette_indices = object.silhouette_indices;
                float max_radius_squared_detached = max_radius_squared;
                bool ret = false;
                for (int i = 0; i < 3; ++i)
                {
                    const int silhouette_index = get(silhouette_indices, i);
                    if (silhouette_index != -1)
                    {
                        const silhouette_edge &se = object.silhouettes[silhouette_index];
                        bool found = se.find_closest_silhouette_point(vec4_to_vec3(origin), max_radius_squared_detached, distance, flip_normal_orientation, min_radius_squared);
                        if (found)
                        {
                            ret = true;
                            max_radius_squared_detached = distance * distance;
                        }
                    }
                }
                return ret;
            }
        };

        struct intersect_test
        {
            __device__ __host__ thrust::pair<bool, float> operator()(const Line<float, 3> &line, const triangle &object) const noexcept
            {
                // Retrieve triangle vertices from `object`
                float3 v0 = object.vertices[object.vertex_indices.x];
                float3 v1 = object.vertices[object.vertex_indices.y];
                float3 v2 = object.vertices[object.vertex_indices.z];

                // Compute the two edge vectors of the triangle
                float3 edge1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
                float3 edge2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};

                // Calculate the determinant using the cross product
                float3 h = {line.dir.y * edge2.z - line.dir.z * edge2.y,
                            line.dir.z * edge2.x - line.dir.x * edge2.z,
                            line.dir.x * edge2.y - line.dir.y * edge2.x};
                float det = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

                // If the determinant is close to zero, the line and triangle are parallel
                if (fabs(det) < epsilon<float>())
                {
                    return thrust::make_pair(false, 0.0f);
                }

#ifdef __CUDA_ARCH__
                auto inv_det = __frcp_rn(det);
#else
                auto inv_det = 1.0f / det;
#endif

                // Calculate the distance from v0 to the line origin
                float3 s = {line.origin.x - v0.x, line.origin.y - v0.y, line.origin.z - v0.z};

                // Calculate the first barycentric coordinate u and check bounds
                float u = (s.x * h.x + s.y * h.y + s.z * h.z) * inv_det;
                if (u < 0.0f || u > 1.0f)
                {
                    return thrust::make_pair(false, 0.0f);
                }

                // Calculate the second barycentric coordinate v and check bounds
                float3 q = {s.y * edge1.z - s.z * edge1.y,
                            s.z * edge1.x - s.x * edge1.z,
                            s.x * edge1.y - s.y * edge1.x};
                float v = (line.dir.x * q.x + line.dir.y * q.y + line.dir.z * q.z) * inv_det;
                if (v < 0.0f || u + v > 1.0f)
                {
                    return thrust::make_pair(false, 0.0f);
                }

                // Calculate the intersection distance t along the line
                float t = (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z) * inv_det;

                // Check if the intersection point lies in the positive direction of the line (t >= 0)
                if (t >= 0.0f)
                {
                    return thrust::make_pair(true, t);
                }

                return thrust::make_pair(false, 0.0f);
            }
        };

        scene<3>() = default;
        template <typename VerticesInputIterator, typename IndicesInputIterator>
        scene<3>(VerticesInputIterator vertices_first, VerticesInputIterator vertices_last, IndicesInputIterator indices_first, IndicesInputIterator indices_last)
            : vertices_h(vertices_first, vertices_last), vertices_d(vertices_h), indices_h(indices_first, indices_last)
        {
        }

        int assign_edge_indices()
        {
            int E = 0;
            int N = (int)indices_h.size();
            std::map<std::pair<int, int>, int> index_map;
            edge_indices_h.clear();

            for (int i = 0; i < N; ++i)
            {
                int3 edge_indices_per_triangle;
                for (int j = 0; j < 3; ++j)
                {
                    int k = (j + 1) % 3;
                    int I = get(indices_h[i], j);
                    int J = get(indices_h[i], k);
                    if (I > J)
                    {
                        std::swap(I, J);
                    }
                    std::pair<int, int> e(I, J);

                    if (index_map.find(e) == index_map.end())
                    {
                        index_map[e] = E++;
                    }
                    get(edge_indices_per_triangle, j) = index_map[e];
                }
                edge_indices_h.push_back(edge_indices_per_triangle);
            }

            return E;
        }
        void compute_silhouettes()
        {
            int number_silhouette_edges = assign_edge_indices();

            silhouettes_h.clear();
            silhouette_edge default_silhouette_edge(vertices_d);
            silhouettes_h.resize(number_silhouette_edges, default_silhouette_edge);

            // for (const auto &triangle_indices : indices_h)
            for (int i = 0; i < indices_h.size(); ++i)
            {
                int3 vertex_indices_per_triangle = indices_h[i];
                int3 edge_indices_per_triangle = edge_indices_h[i];
                for (int j = 0; j < 3; j++)
                {
                    int I = j - 1 < 0 ? 2 : j - 1;
                    int J = j + 0;
                    int K = j + 1 > 2 ? 0 : j + 1;
                    int e_index = get(edge_indices_per_triangle, j);

                    int orientation = 1;
                    if (get(vertex_indices_per_triangle, J) > get(vertex_indices_per_triangle, K))
                    {
                        std::swap(J, K);
                        orientation *= -1;
                    }

                    silhouette_edge &se = silhouettes_h[e_index];
                    get(se.indices, orientation == 1 ? 0 : 3) = get(vertex_indices_per_triangle, I);
                    get(se.indices, 1) = get(vertex_indices_per_triangle, J);
                    get(se.indices, 2) = get(vertex_indices_per_triangle, K);
                    se.vertices = vertices_d.data().get();
                }
            }

            silhouettes_d.resize(silhouettes_h.size());
            silhouettes_d = silhouettes_h;
        }
        void build_bvh()
        {
            std::unordered_map<int, bool> seen_silhouettes;
            // for(auto it = indices_h.begin(); it != indices_h.end(); std::advance(it, 1))
            for (int i = 0; i < indices_h.size(); ++i) // Traverse through triangles.
            {
                int3 vertex_indices_per_triangle = indices_h[i];
                int3 edge_indices_per_triangle = edge_indices_h[i];
                int3 silhouette_indices = make_int3(-1, -1, -1);
                int silhouette_indices_ptr = 0;
                for (int j = 0; j < 3; ++j)
                {
                    int e_index = get(edge_indices_per_triangle, j);
                    if (!seen_silhouettes[e_index])
                    {
                        seen_silhouettes[e_index] = true;
                        get(silhouette_indices, silhouette_indices_ptr++) = e_index;
                    }
                }
                triangles.push_back(triangle(vertex_indices_per_triangle, silhouette_indices, vertices_d, silhouettes_d));
            }

            p_bvh = std::make_unique<lbvh::bvh<float, 3, triangle, aabb_getter, cone_getter>>(triangles.begin(), triangles.end(), true);
            bvh_dev = p_bvh->get_device_repr();
        }
        const auto &get_bvh_device_ptr() const
        {
            if (p_bvh)
            {
                return bvh_dev;
            }
            else
            {
                throw std::runtime_error("BVH is not built yet.");
            }
        }

    private:
        thrust::host_vector<float3> vertices_h;
        thrust::device_vector<float3> vertices_d;

        thrust::host_vector<int3> indices_h;
        thrust::host_vector<int3> edge_indices_h;

        std::vector<triangle> triangles;

        thrust::host_vector<silhouette_edge> silhouettes_h;
        thrust::device_vector<silhouette_edge> silhouettes_d;

        std::unique_ptr<lbvh::bvh<float, 3, triangle, aabb_getter, cone_getter>> p_bvh;
        lbvh::bvh_device<float, 3, triangle> bvh_dev;
    };
}
