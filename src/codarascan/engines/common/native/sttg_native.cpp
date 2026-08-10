// SPDX-License-Identifier: Apache-2.0

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kEpsilon = 1e-12;

struct Cell {
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;
    double cos2 = 0.0;
    double sin2 = 0.0;
    float occupancy = 0.0f;
    float coherence = 0.0f;
    int edge_count = 0;
    int positive_x = 0;
    int negative_x = 0;
    int positive_y = 0;
    int negative_y = 0;
    uint8_t state = 0;  // 0 background, 1 weak growth, 2 strong seed
};

struct ContextComponent {
    std::vector<int> members;
    int cells = 0;
    int seeds = 0;
    double weight = 0.0;
    double cos2 = 0.0;
    double sin2 = 0.0;
    double occupancy = 0.0;
    double coherence = 0.0;
    int positive_x = 0;
    int negative_x = 0;
    int positive_y = 0;
    int negative_y = 0;
    double theta = 0.0;
    double orientation_resultant = 0.0;
    double density = 0.0;
    double polarity_balance = 0.0;
    double score = 0.0;
    double min_along = 0.0;
    double max_along = 0.0;
    double min_across = 0.0;
    double max_across = 0.0;
    double profile_transitions = 0.0;
    double profile_contrast = 0.0;
    double profile_agreement = 0.0;
    double low_transition_rescue = 0.0;
};

struct UnionFind {
    explicit UnionFind(int size) : parent(size), rank(size, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int value) {
        int root = value;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[value] != value) {
            const int next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    void unite(int first, int second) {
        first = find(first);
        second = find(second);
        if (first == second) {
            return;
        }
        if (rank[first] < rank[second]) {
            std::swap(first, second);
        }
        parent[second] = first;
        if (rank[first] == rank[second]) {
            rank[first] += 1;
        }
    }

    std::vector<int> parent;
    std::vector<uint8_t> rank;
};

struct Component {
    int root = -1;
    int cells = 0;
    int seeds = 0;
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
    double weight = 0.0;
    double cos2 = 0.0;
    double sin2 = 0.0;
    double occupancy = 0.0;
    double coherence = 0.0;
    int positive_x = 0;
    int negative_x = 0;
    int positive_y = 0;
    int negative_y = 0;
    double theta = 0.0;
    double orientation_resultant = 0.0;
    double density = 0.0;
    double polarity_balance = 0.0;
    double score = 0.0;
    double min_along = 0.0;
    double max_along = 0.0;
    double min_across = 0.0;
    double max_across = 0.0;
};

inline bool orientation_compatible(const Cell& first, const Cell& second, double cosine_threshold) {
    return first.cos2 * second.cos2 + first.sin2 * second.sin2 >= cosine_threshold;
}

PyObject* propose(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* image_object = nullptr;
    int window = 5;
    double gradient_threshold = 48.0;
    double seed_occupancy = 0.42;
    double seed_coherence = 0.93;
    double grow_occupancy = 0.28;
    double grow_coherence = 0.82;
    double angle_tolerance_degrees = 10.0;
    int minimum_cells = 12;
    double minimum_density = 0.20;
    double minimum_orientation = 0.86;
    double minimum_polarity = 0.10;
    double long_padding = 0.75;
    double short_padding = 0.50;

    static const char* keywords[] = {
        "image",
        "window",
        "gradient_threshold",
        "seed_occupancy",
        "seed_coherence",
        "grow_occupancy",
        "grow_coherence",
        "angle_tolerance_degrees",
        "minimum_cells",
        "minimum_density",
        "minimum_orientation",
        "minimum_polarity",
        "long_padding",
        "short_padding",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|iddddddiddddd",
            const_cast<char**>(keywords),
            &image_object,
            &window,
            &gradient_threshold,
            &seed_occupancy,
            &seed_coherence,
            &grow_occupancy,
            &grow_coherence,
            &angle_tolerance_degrees,
            &minimum_cells,
            &minimum_density,
            &minimum_orientation,
            &minimum_polarity,
            &long_padding,
            &short_padding)) {
        return nullptr;
    }
    if (window < 2 || minimum_cells < 1) {
        PyErr_SetString(PyExc_ValueError, "window must be >=2 and minimum_cells must be positive");
        return nullptr;
    }

    PyArrayObject* image = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(image_object, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    if (image == nullptr) {
        return nullptr;
    }
    if (PyArray_NDIM(image) != 2) {
        Py_DECREF(image);
        PyErr_SetString(PyExc_ValueError, "image must be a two-dimensional uint8 array");
        return nullptr;
    }

    const int height = static_cast<int>(PyArray_DIM(image, 0));
    const int width = static_cast<int>(PyArray_DIM(image, 1));
    const int grid_height = height / window;
    const int grid_width = width / window;
    if (grid_height <= 0 || grid_width <= 0) {
        Py_DECREF(image);
        npy_intp quad_dimensions[] = {0, 4, 2};
        npy_intp metric_dimensions[] = {0, 8};
        PyObject* quads = PyArray_SimpleNew(3, quad_dimensions, NPY_FLOAT32);
        PyObject* metrics = PyArray_SimpleNew(2, metric_dimensions, NPY_FLOAT32);
        return Py_BuildValue("NN", quads, metrics);
    }

    const auto* pixels = static_cast<const uint8_t*>(PyArray_DATA(image));
    const npy_intp stride = PyArray_STRIDE(image, 0);
    std::vector<Cell> cells(static_cast<size_t>(grid_height) * grid_width);
    std::vector<Component> components;
    std::vector<int> accepted_components;
    const double threshold_squared = gradient_threshold * gradient_threshold;

    Py_BEGIN_ALLOW_THREADS
    for (int y = 1; y < std::min(height - 1, grid_height * window); ++y) {
        const auto* previous = pixels + static_cast<npy_intp>(y - 1) * stride;
        const auto* current = pixels + static_cast<npy_intp>(y) * stride;
        const auto* next = pixels + static_cast<npy_intp>(y + 1) * stride;
        const int cell_y = y / window;
        for (int x = 1; x < std::min(width - 1, grid_width * window); ++x) {
            const int gx =
                3 * (static_cast<int>(previous[x + 1]) - static_cast<int>(previous[x - 1]))
                + 10 * (static_cast<int>(current[x + 1]) - static_cast<int>(current[x - 1]))
                + 3 * (static_cast<int>(next[x + 1]) - static_cast<int>(next[x - 1]));
            const int gy =
                3 * (static_cast<int>(next[x - 1]) - static_cast<int>(previous[x - 1]))
                + 10 * (static_cast<int>(next[x]) - static_cast<int>(previous[x]))
                + 3 * (static_cast<int>(next[x + 1]) - static_cast<int>(previous[x + 1]));
            const double magnitude_squared =
                static_cast<double>(gx) * gx + static_cast<double>(gy) * gy;
            if (magnitude_squared < threshold_squared) {
                continue;
            }
            Cell& cell = cells[static_cast<size_t>(cell_y) * grid_width + x / window];
            cell.edge_count += 1;
            cell.xx += static_cast<double>(gx) * gx;
            cell.yy += static_cast<double>(gy) * gy;
            cell.xy += static_cast<double>(gx) * gy;
            cell.positive_x += gx > 0;
            cell.negative_x += gx < 0;
            cell.positive_y += gy > 0;
            cell.negative_y += gy < 0;
        }
    }

    for (Cell& cell : cells) {
        cell.occupancy = static_cast<float>(cell.edge_count) / static_cast<float>(window * window);
        const double trace = cell.xx + cell.yy;
        if (trace <= kEpsilon) {
            continue;
        }
        const double difference = cell.xx - cell.yy;
        const double twice_cross = 2.0 * cell.xy;
        const double discriminant = std::hypot(difference, twice_cross);
        cell.coherence = static_cast<float>(discriminant / trace);
        if (discriminant > kEpsilon) {
            cell.cos2 = difference / discriminant;
            cell.sin2 = twice_cross / discriminant;
        }
        if (cell.occupancy >= seed_occupancy && cell.coherence >= seed_coherence) {
            cell.state = 2;
        } else if (cell.occupancy >= grow_occupancy && cell.coherence >= grow_coherence) {
            cell.state = 1;
        }
    }

    // Remove isolated responses before graph construction.
    std::vector<uint8_t> retained(cells.size(), 0);
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int index = y * grid_width + x;
            if (cells[index].state == 0) {
                continue;
            }
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height
                        && cells[ny * grid_width + nx].state != 0) {
                        neighbors += 1;
                    }
                }
            }
            retained[index] = neighbors >= 2;
        }
    }
    for (size_t index = 0; index < cells.size(); ++index) {
        if (!retained[index]) {
            cells[index].state = 0;
        }
    }

    UnionFind union_find(static_cast<int>(cells.size()));
    const double orientation_cosine =
        std::cos(2.0 * angle_tolerance_degrees * 3.14159265358979323846 / 180.0);
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int index = y * grid_width + x;
            if (cells[index].state == 0) {
                continue;
            }
            // Immediate adjacency grows along the bars. A bounded gap in the
            // dominant gradient direction connects successive parallel bars
            // without generic morphology that would smear every orientation.
            for (int dy = -3; dy <= 0; ++dy) {
                for (int dx = -3; dx <= 3; ++dx) {
                    if (dy == 0 && dx >= 0) {
                        continue;
                    }
                    const double distance_squared = static_cast<double>(dx * dx + dy * dy);
                    if (distance_squared > 9.0) {
                        continue;
                    }
                    if (distance_squared > 2.0) {
                        const double displacement_cos2 =
                            static_cast<double>(dx * dx - dy * dy) / distance_squared;
                        const double displacement_sin2 =
                            static_cast<double>(2 * dx * dy) / distance_squared;
                        const double gradient_alignment =
                            displacement_cos2 * cells[index].cos2
                            + displacement_sin2 * cells[index].sin2;
                        if (gradient_alignment < 0.5) {
                            continue;
                        }
                    }
                    const int nx = x + dx;
                    const int ny = y + dy;
                if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
                    continue;
                }
                const int neighbor = ny * grid_width + nx;
                if (cells[neighbor].state != 0
                    && orientation_compatible(cells[index], cells[neighbor], orientation_cosine)) {
                    union_find.unite(index, neighbor);
                }
                }
            }
        }
    }

    std::unordered_map<int, int> component_index;
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int index = y * grid_width + x;
            const Cell& cell = cells[index];
            if (cell.state == 0) {
                continue;
            }
            const int root = union_find.find(index);
            auto iterator = component_index.find(root);
            int position;
            if (iterator == component_index.end()) {
                position = static_cast<int>(components.size());
                component_index.emplace(root, position);
                Component component;
                component.root = root;
                component.min_x = component.max_x = x;
                component.min_y = component.max_y = y;
                components.push_back(component);
            } else {
                position = iterator->second;
            }
            Component& component = components[position];
            component.cells += 1;
            component.seeds += cell.state == 2;
            component.min_x = std::min(component.min_x, x);
            component.max_x = std::max(component.max_x, x);
            component.min_y = std::min(component.min_y, y);
            component.max_y = std::max(component.max_y, y);
            const double weight = std::max(1, cell.edge_count);
            component.weight += weight;
            component.cos2 += weight * cell.cos2;
            component.sin2 += weight * cell.sin2;
            component.occupancy += cell.occupancy;
            component.coherence += cell.coherence;
            component.positive_x += cell.positive_x;
            component.negative_x += cell.negative_x;
            component.positive_y += cell.positive_y;
            component.negative_y += cell.negative_y;
        }
    }

    for (int index = 0; index < static_cast<int>(components.size()); ++index) {
        Component& component = components[index];
        if (component.cells < minimum_cells || component.seeds == 0) {
            continue;
        }
        const int box_area =
            (component.max_x - component.min_x + 1) * (component.max_y - component.min_y + 1);
        component.density = static_cast<double>(component.cells) / std::max(1, box_area);
        component.orientation_resultant =
            std::hypot(component.cos2, component.sin2) / std::max(kEpsilon, component.weight);
        if (component.density < minimum_density
            || component.orientation_resultant < minimum_orientation) {
            continue;
        }
        component.theta = 0.5 * std::atan2(component.sin2, component.cos2);
        const bool use_x = std::abs(std::cos(component.theta)) >= std::abs(std::sin(component.theta));
        const int positive = use_x ? component.positive_x : component.positive_y;
        const int negative = use_x ? component.negative_x : component.negative_y;
        component.polarity_balance =
            2.0 * std::min(positive, negative) / std::max(1.0, static_cast<double>(positive + negative));
        if (component.polarity_balance < minimum_polarity) {
            continue;
        }
        component.score =
            component.orientation_resultant
            * (component.occupancy / component.cells)
            * component.polarity_balance
            * std::sqrt(static_cast<double>(component.cells));
        component.min_along = component.min_across = std::numeric_limits<double>::infinity();
        component.max_along = component.max_across = -std::numeric_limits<double>::infinity();
        accepted_components.push_back(index);
    }

    std::unordered_map<int, int> accepted_by_root;
    for (int output_index = 0; output_index < static_cast<int>(accepted_components.size()); ++output_index) {
        accepted_by_root.emplace(components[accepted_components[output_index]].root, output_index);
    }
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int index = y * grid_width + x;
            if (cells[index].state == 0) {
                continue;
            }
            const int root = union_find.find(index);
            const auto iterator = accepted_by_root.find(root);
            if (iterator == accepted_by_root.end()) {
                continue;
            }
            Component& component = components[accepted_components[iterator->second]];
            const double cosine = std::cos(component.theta);
            const double sine = std::sin(component.theta);
            const double px = (x + 0.5) * window;
            const double py = (y + 0.5) * window;
            const double along = px * cosine + py * sine;
            const double across = -px * sine + py * cosine;
            component.min_along = std::min(component.min_along, along);
            component.max_along = std::max(component.max_along, along);
            component.min_across = std::min(component.min_across, across);
            component.max_across = std::max(component.max_across, across);
        }
    }
    Py_END_ALLOW_THREADS

    std::sort(
        accepted_components.begin(),
        accepted_components.end(),
        [&](int first, int second) { return components[first].score > components[second].score; });

    npy_intp quad_dimensions[] = {
        static_cast<npy_intp>(accepted_components.size()), 4, 2,
    };
    npy_intp metric_dimensions[] = {
        static_cast<npy_intp>(accepted_components.size()), 8,
    };
    PyObject* quad_object = PyArray_SimpleNew(3, quad_dimensions, NPY_FLOAT32);
    PyObject* metric_object = PyArray_SimpleNew(2, metric_dimensions, NPY_FLOAT32);
    if (quad_object == nullptr || metric_object == nullptr) {
        Py_XDECREF(quad_object);
        Py_XDECREF(metric_object);
        Py_DECREF(image);
        return nullptr;
    }
    auto* quads = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(quad_object)));
    auto* metrics = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(metric_object)));
    for (int output_index = 0; output_index < static_cast<int>(accepted_components.size()); ++output_index) {
        const Component& component = components[accepted_components[output_index]];
        const double cosine = std::cos(component.theta);
        const double sine = std::sin(component.theta);
        const double direction_x = cosine;
        const double direction_y = sine;
        const double normal_x = -sine;
        const double normal_y = cosine;
        const double along_low = component.min_along - long_padding * window;
        const double along_high = component.max_along + long_padding * window;
        const double across_low = component.min_across - short_padding * window;
        const double across_high = component.max_across + short_padding * window;
        const double corners[4][2] = {
            {direction_x * along_low + normal_x * across_low,
             direction_y * along_low + normal_y * across_low},
            {direction_x * along_high + normal_x * across_low,
             direction_y * along_high + normal_y * across_low},
            {direction_x * along_high + normal_x * across_high,
             direction_y * along_high + normal_y * across_high},
            {direction_x * along_low + normal_x * across_high,
             direction_y * along_low + normal_y * across_high},
        };
        for (int corner = 0; corner < 4; ++corner) {
            quads[(output_index * 4 + corner) * 2] = static_cast<float>(corners[corner][0]);
            quads[(output_index * 4 + corner) * 2 + 1] = static_cast<float>(corners[corner][1]);
        }
        metrics[output_index * 8] = static_cast<float>(component.score);
        metrics[output_index * 8 + 1] = static_cast<float>(component.cells);
        metrics[output_index * 8 + 2] = static_cast<float>(component.seeds);
        metrics[output_index * 8 + 3] = static_cast<float>(component.density);
        metrics[output_index * 8 + 4] = static_cast<float>(component.orientation_resultant);
        metrics[output_index * 8 + 5] = static_cast<float>(component.polarity_balance);
        metrics[output_index * 8 + 6] = static_cast<float>(component.occupancy / component.cells);
        metrics[output_index * 8 + 7] = static_cast<float>(component.coherence / component.cells);
    }

    Py_DECREF(image);
    return Py_BuildValue("NN", quad_object, metric_object);
}

double dictionary_double(PyObject* settings, const char* key, double fallback) {
    if (settings == nullptr || settings == Py_None || !PyDict_Check(settings)) {
        return fallback;
    }
    PyObject* value = PyDict_GetItemString(settings, key);
    if (value == nullptr) {
        return fallback;
    }
    const double parsed = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return fallback;
    }
    return parsed;
}

int dictionary_int(PyObject* settings, const char* key, int fallback) {
    if (settings == nullptr || settings == Py_None || !PyDict_Check(settings)) {
        return fallback;
    }
    PyObject* value = PyDict_GetItemString(settings, key);
    if (value == nullptr) {
        return fallback;
    }
    const long parsed = PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return fallback;
    }
    return static_cast<int>(parsed);
}

PyObject* propose_context(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* image_object = nullptr;
    PyObject* settings = Py_None;
    static const char* keywords[] = {"image", "settings", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|O",
            const_cast<char**>(keywords),
            &image_object,
            &settings)) {
        return nullptr;
    }
    if (settings != Py_None && !PyDict_Check(settings)) {
        PyErr_SetString(PyExc_TypeError, "settings must be a dictionary");
        return nullptr;
    }

    const int tile = dictionary_int(settings, "tile", 12);
    const double gradient_threshold = dictionary_double(settings, "gradient_threshold", 48.0);
    const double energy_percentile = dictionary_double(settings, "energy_percentile", 35.0);
    const double seed_occupancy = dictionary_double(settings, "seed_occupancy", 0.18);
    const double seed_coherence = dictionary_double(settings, "seed_coherence", 0.92);
    const double seed_context = dictionary_double(settings, "seed_context", 0.82);
    const double grow_occupancy = dictionary_double(settings, "grow_occupancy", 0.07);
    const double grow_coherence = dictionary_double(settings, "grow_coherence", 0.80);
    const double grow_context = dictionary_double(settings, "grow_context", 0.67);
    const double grow_energy_factor = dictionary_double(settings, "grow_energy_factor", 0.55);
    const int orientation_bins = dictionary_int(settings, "orientation_bins", 12);
    const double fine_tolerance = dictionary_double(settings, "fine_tolerance_degrees", 13.0);
    const double context_tolerance = dictionary_double(settings, "context_tolerance_degrees", 16.9);
    const int minimum_cells = dictionary_int(settings, "minimum_cells", 3);
    const double minimum_density = dictionary_double(settings, "minimum_density", 0.0);
    const double minimum_orientation = dictionary_double(settings, "minimum_orientation", 0.0);
    const double minimum_polarity = dictionary_double(settings, "minimum_polarity", 0.0);
    const double long_padding = dictionary_double(settings, "long_padding", 1.0);
    const double short_padding = dictionary_double(settings, "short_padding", 1.0);
    const int profile_filter = dictionary_int(settings, "profile_filter", 1);
    const int minimum_profile_transitions =
        dictionary_int(settings, "minimum_profile_transitions", 8);
    const double minimum_profile_contrast =
        dictionary_double(settings, "minimum_profile_contrast", 20.0);
    const double minimum_profile_agreement =
        dictionary_double(settings, "minimum_profile_agreement", 0.45);
    const double maximum_profile_transition_rate =
        dictionary_double(settings, "maximum_profile_transition_rate", 0.72);
    const int retain_low_transition_rescue =
        dictionary_int(settings, "retain_low_transition_rescue", 0);
    const int rescue_minimum_profile_transitions =
        dictionary_int(settings, "rescue_minimum_profile_transitions", 4);
    const double rescue_minimum_aspect =
        dictionary_double(settings, "rescue_minimum_aspect", 1.2);
    const double rescue_maximum_aspect =
        dictionary_double(settings, "rescue_maximum_aspect", 3.0);
    const double rescue_minimum_orientation =
        dictionary_double(settings, "rescue_minimum_orientation", 0.995);
    const double rescue_minimum_contrast =
        dictionary_double(settings, "rescue_minimum_contrast", 150.0);
    const double rescue_minimum_agreement =
        dictionary_double(settings, "rescue_minimum_agreement", 0.65);
    const int rescue_minimum_cells =
        dictionary_int(settings, "rescue_minimum_cells", 40);
    const int rescue_maximum_cells =
        dictionary_int(settings, "rescue_maximum_cells", 500);
    if (tile < 2 || orientation_bins < 1 || minimum_cells < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid context STTG settings");
        return nullptr;
    }

    PyArrayObject* image = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(image_object, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    if (image == nullptr) {
        return nullptr;
    }
    if (PyArray_NDIM(image) != 2) {
        Py_DECREF(image);
        PyErr_SetString(PyExc_ValueError, "image must be a two-dimensional uint8 array");
        return nullptr;
    }

    const int height = static_cast<int>(PyArray_DIM(image, 0));
    const int width = static_cast<int>(PyArray_DIM(image, 1));
    const int grid_height = height / tile;
    const int grid_width = width / tile;
    if (grid_height <= 0 || grid_width <= 0) {
        Py_DECREF(image);
        npy_intp quad_dimensions[] = {0, 4, 2};
        npy_intp metric_dimensions[] = {0, 12};
        return Py_BuildValue(
            "NN",
            PyArray_SimpleNew(3, quad_dimensions, NPY_FLOAT32),
            PyArray_SimpleNew(2, metric_dimensions, NPY_FLOAT32));
    }

    const auto* pixels = static_cast<const uint8_t*>(PyArray_DATA(image));
    const npy_intp stride = PyArray_STRIDE(image, 0);
    std::vector<Cell> cells(static_cast<size_t>(grid_height) * grid_width);
    std::vector<double> energy(cells.size(), 0.0);
    std::vector<double> context_cos2(cells.size(), 0.0);
    std::vector<double> context_sin2(cells.size(), 0.0);
    std::vector<double> context_coherence(cells.size(), 0.0);
    std::vector<ContextComponent> components;
    const double threshold_squared = gradient_threshold * gradient_threshold;

    Py_BEGIN_ALLOW_THREADS
    for (int y = 1; y < std::min(height - 1, grid_height * tile); ++y) {
        const auto* previous = pixels + static_cast<npy_intp>(y - 1) * stride;
        const auto* current = pixels + static_cast<npy_intp>(y) * stride;
        const auto* next = pixels + static_cast<npy_intp>(y + 1) * stride;
        const int cell_y = y / tile;
        for (int x = 1; x < std::min(width - 1, grid_width * tile); ++x) {
            const int gx =
                3 * (static_cast<int>(previous[x + 1]) - static_cast<int>(previous[x - 1]))
                + 10 * (static_cast<int>(current[x + 1]) - static_cast<int>(current[x - 1]))
                + 3 * (static_cast<int>(next[x + 1]) - static_cast<int>(next[x - 1]));
            const int gy =
                3 * (static_cast<int>(next[x - 1]) - static_cast<int>(previous[x - 1]))
                + 10 * (static_cast<int>(next[x]) - static_cast<int>(previous[x]))
                + 3 * (static_cast<int>(next[x + 1]) - static_cast<int>(previous[x + 1]));
            const double magnitude_squared =
                static_cast<double>(gx) * gx + static_cast<double>(gy) * gy;
            if (magnitude_squared < threshold_squared) {
                continue;
            }
            Cell& cell = cells[static_cast<size_t>(cell_y) * grid_width + x / tile];
            cell.edge_count += 1;
            cell.xx += static_cast<double>(gx) * gx;
            cell.yy += static_cast<double>(gy) * gy;
            cell.xy += static_cast<double>(gx) * gy;
            cell.positive_x += gx > 0;
            cell.negative_x += gx < 0;
            cell.positive_y += gy > 0;
            cell.negative_y += gy < 0;
        }
    }

    std::vector<double> positive_energy;
    positive_energy.reserve(cells.size());
    for (size_t index = 0; index < cells.size(); ++index) {
        Cell& cell = cells[index];
        cell.occupancy = static_cast<float>(cell.edge_count) / static_cast<float>(tile * tile);
        const double trace = cell.xx + cell.yy;
        energy[index] = std::sqrt(std::max(0.0, trace));
        if (cell.occupancy > 0.01) {
            positive_energy.push_back(energy[index]);
        }
        if (trace <= kEpsilon) {
            continue;
        }
        const double difference = cell.xx - cell.yy;
        const double twice_cross = 2.0 * cell.xy;
        const double discriminant = std::hypot(difference, twice_cross);
        cell.coherence = static_cast<float>(discriminant / trace);
        if (discriminant > kEpsilon) {
            cell.cos2 = difference / discriminant;
            cell.sin2 = twice_cross / discriminant;
        }
    }
    double energy_gate = std::numeric_limits<double>::infinity();
    if (!positive_energy.empty()) {
        const size_t percentile_index = std::min(
            positive_energy.size() - 1,
            static_cast<size_t>(
                std::floor(std::clamp(energy_percentile, 0.0, 100.0) * 0.01
                           * static_cast<double>(positive_energy.size() - 1))));
        std::nth_element(
            positive_energy.begin(),
            positive_energy.begin() + percentile_index,
            positive_energy.end());
        energy_gate = positive_energy[percentile_index];
    }

    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            double xx = 0.0;
            double yy = 0.0;
            double xy = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
                        continue;
                    }
                    const Cell& neighbor = cells[ny * grid_width + nx];
                    xx += neighbor.xx;
                    yy += neighbor.yy;
                    xy += neighbor.xy;
                }
            }
            const int index = y * grid_width + x;
            const double trace = xx + yy;
            if (trace <= kEpsilon) {
                continue;
            }
            const double difference = xx - yy;
            const double twice_cross = 2.0 * xy;
            const double discriminant = std::hypot(difference, twice_cross);
            context_coherence[index] = discriminant / trace;
            if (discriminant > kEpsilon) {
                context_cos2[index] = difference / discriminant;
                context_sin2[index] = twice_cross / discriminant;
            }
            Cell& cell = cells[index];
            if (cell.occupancy >= seed_occupancy
                && cell.coherence >= seed_coherence
                && context_coherence[index] >= seed_context
                && energy[index] >= energy_gate) {
                cell.state = 2;
            } else if (
                cell.occupancy >= grow_occupancy
                && cell.coherence >= grow_coherence
                && context_coherence[index] >= grow_context
                && energy[index] >= grow_energy_factor * energy_gate) {
                cell.state = 1;
            }
        }
    }

    const double fine_cosine =
        std::cos(2.0 * fine_tolerance * 3.14159265358979323846 / 180.0);
    const double context_cosine =
        std::cos(2.0 * context_tolerance * 3.14159265358979323846 / 180.0);
    std::vector<uint8_t> visited(cells.size());
    std::vector<int> stack;
    std::vector<int> members;
    for (int orientation_bin = 0; orientation_bin < orientation_bins; ++orientation_bin) {
        std::fill(visited.begin(), visited.end(), 0);
        const double center =
            orientation_bin * 3.14159265358979323846 / orientation_bins;
        const double center_cos2 = std::cos(2.0 * center);
        const double center_sin2 = std::sin(2.0 * center);
        auto eligible = [&](int index) {
            const Cell& cell = cells[index];
            return cell.state != 0
                && cell.cos2 * center_cos2 + cell.sin2 * center_sin2 >= fine_cosine
                && context_cos2[index] * center_cos2 + context_sin2[index] * center_sin2
                    >= context_cosine;
        };
        for (int origin = 0; origin < static_cast<int>(cells.size()); ++origin) {
            if (visited[origin] || !eligible(origin)) {
                continue;
            }
            stack.clear();
            members.clear();
            stack.push_back(origin);
            visited[origin] = 1;
            while (!stack.empty()) {
                const int index = stack.back();
                stack.pop_back();
                members.push_back(index);
                const int x = index % grid_width;
                const int y = index / grid_width;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
                            continue;
                        }
                        const int neighbor = ny * grid_width + nx;
                        if (!visited[neighbor] && eligible(neighbor)) {
                            visited[neighbor] = 1;
                            stack.push_back(neighbor);
                        }
                    }
                }
            }
            if (static_cast<int>(members.size()) < minimum_cells) {
                continue;
            }
            ContextComponent component;
            int min_x = grid_width;
            int max_x = 0;
            int min_y = grid_height;
            int max_y = 0;
            for (int index : members) {
                const Cell& cell = cells[index];
                const int x = index % grid_width;
                const int y = index / grid_width;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
                component.seeds += cell.state == 2;
                const double weight = std::max(1, cell.edge_count);
                component.weight += weight;
                component.cos2 += weight * cell.cos2;
                component.sin2 += weight * cell.sin2;
                component.occupancy += cell.occupancy;
                component.coherence += cell.coherence;
                component.positive_x += cell.positive_x;
                component.negative_x += cell.negative_x;
                component.positive_y += cell.positive_y;
                component.negative_y += cell.negative_y;
            }
            if (component.seeds == 0) {
                continue;
            }
            component.members = members;
            component.cells = static_cast<int>(members.size());
            component.density = static_cast<double>(component.cells)
                / std::max(1, (max_x - min_x + 1) * (max_y - min_y + 1));
            component.orientation_resultant =
                std::hypot(component.cos2, component.sin2)
                / std::max(kEpsilon, component.weight);
            if (component.density < minimum_density
                || component.orientation_resultant < minimum_orientation) {
                continue;
            }
            component.theta = 0.5 * std::atan2(component.sin2, component.cos2);
            const bool use_x =
                std::abs(std::cos(component.theta)) >= std::abs(std::sin(component.theta));
            const int positive = use_x ? component.positive_x : component.positive_y;
            const int negative = use_x ? component.negative_x : component.negative_y;
            component.polarity_balance =
                2.0 * std::min(positive, negative)
                / std::max(1.0, static_cast<double>(positive + negative));
            if (component.polarity_balance < minimum_polarity) {
                continue;
            }
            component.score =
                component.orientation_resultant
                * (component.occupancy / component.cells)
                * (0.25 + 0.75 * component.polarity_balance)
                * std::sqrt(static_cast<double>(component.cells));
            component.min_along = component.min_across =
                std::numeric_limits<double>::infinity();
            component.max_along = component.max_across =
                -std::numeric_limits<double>::infinity();
            const double cosine = std::cos(component.theta);
            const double sine = std::sin(component.theta);
            for (int index : component.members) {
                const int x = index % grid_width;
                const int y = index / grid_width;
                const double px = (x + 0.5) * tile;
                const double py = (y + 0.5) * tile;
                const double along = px * cosine + py * sine;
                const double across = -px * sine + py * cosine;
                component.min_along = std::min(component.min_along, along);
                component.max_along = std::max(component.max_along, along);
                component.min_across = std::min(component.min_across, across);
                component.max_across = std::max(component.max_across, across);
            }
            if (profile_filter) {
                const double along_span = component.max_along - component.min_along;
                const double across_center =
                    0.5 * (component.min_across + component.max_across);
                const double across_span = component.max_across - component.min_across;
                const int sample_count = std::clamp(
                    static_cast<int>(std::lround(along_span)),
                    16,
                    2048);
                std::vector<std::vector<uint8_t>> binary(
                    3,
                    std::vector<uint8_t>(sample_count, 0));
                double contrast_sum = 0.0;
                int valid_lines = 0;
                for (int line = 0; line < 3; ++line) {
                    const double across = across_center
                        + (line - 1) * 0.22 * across_span;
                    std::vector<uint8_t> samples(sample_count, 255);
                    uint8_t minimum = 255;
                    uint8_t maximum = 0;
                    for (int sample = 0; sample < sample_count; ++sample) {
                        const double along = component.min_along
                            + (sample + 0.5) * along_span / sample_count;
                        const int px = static_cast<int>(
                            std::lround(std::cos(component.theta) * along
                                        - std::sin(component.theta) * across));
                        const int py = static_cast<int>(
                            std::lround(std::sin(component.theta) * along
                                        + std::cos(component.theta) * across));
                        if (px < 0 || px >= width || py < 0 || py >= height) {
                            continue;
                        }
                        const uint8_t value =
                            pixels[static_cast<npy_intp>(py) * stride + px];
                        samples[sample] = value;
                        minimum = std::min(minimum, value);
                        maximum = std::max(maximum, value);
                    }
                    const double contrast = static_cast<double>(maximum) - minimum;
                    if (contrast < minimum_profile_contrast) {
                        continue;
                    }
                    const double threshold = 0.5 * (minimum + maximum);
                    for (int sample = 0; sample < sample_count; ++sample) {
                        binary[line][sample] = samples[sample] < threshold;
                    }
                    // Suppress isolated one-sample flips before transition counting.
                    for (int sample = 1; sample + 1 < sample_count; ++sample) {
                        if (binary[line][sample - 1] == binary[line][sample + 1]
                            && binary[line][sample] != binary[line][sample - 1]) {
                            binary[line][sample] = binary[line][sample - 1];
                        }
                    }
                    contrast_sum += contrast;
                    valid_lines += 1;
                }
                if (valid_lines < 2) {
                    continue;
                }
                int transitions = 0;
                for (int sample = 1; sample < sample_count; ++sample) {
                    transitions += binary[1][sample] != binary[1][sample - 1];
                }
                if (transitions == 0 && valid_lines >= 2) {
                    for (int line : {0, 2}) {
                        int candidate_transitions = 0;
                        for (int sample = 1; sample < sample_count; ++sample) {
                            candidate_transitions +=
                                binary[line][sample] != binary[line][sample - 1];
                        }
                        transitions = std::max(transitions, candidate_transitions);
                    }
                }
                double agreement = 0.0;
                int agreement_lines = 0;
                for (int line : {0, 2}) {
                    int equal = 0;
                    for (int sample = 0; sample < sample_count; ++sample) {
                        equal += binary[line][sample] == binary[1][sample];
                    }
                    const double ratio =
                        static_cast<double>(equal) / sample_count;
                    agreement += std::max(ratio, 1.0 - ratio);
                    agreement_lines += 1;
                }
                agreement /= std::max(1, agreement_lines);
                const double transition_rate =
                    static_cast<double>(transitions) / sample_count;
                const double component_aspect =
                    along_span / std::max(kEpsilon, across_span);
                const double mean_contrast =
                    contrast_sum / std::max(1, valid_lines);
                const bool regular_profile =
                    transitions >= minimum_profile_transitions
                    && transition_rate <= maximum_profile_transition_rate
                    && agreement >= minimum_profile_agreement;
                const bool rescue_profile =
                    retain_low_transition_rescue
                    && transitions >= rescue_minimum_profile_transitions
                    && transitions < minimum_profile_transitions
                    && component_aspect >= rescue_minimum_aspect
                    && component_aspect <= rescue_maximum_aspect
                    && component.orientation_resultant
                        >= rescue_minimum_orientation
                    && mean_contrast >= rescue_minimum_contrast
                    && agreement >= rescue_minimum_agreement
                    && component.cells >= rescue_minimum_cells
                    && component.cells <= rescue_maximum_cells;
                if (!regular_profile && !rescue_profile) {
                    continue;
                }
                component.profile_transitions = transitions;
                component.profile_contrast = mean_contrast;
                component.profile_agreement = agreement;
                component.low_transition_rescue = rescue_profile ? 1.0 : 0.0;
            }
            components.push_back(std::move(component));
        }
    }
    Py_END_ALLOW_THREADS

    std::sort(
        components.begin(),
        components.end(),
        [](const ContextComponent& first, const ContextComponent& second) {
            return first.score > second.score;
        });
    npy_intp quad_dimensions[] = {static_cast<npy_intp>(components.size()), 4, 2};
    npy_intp metric_dimensions[] = {static_cast<npy_intp>(components.size()), 12};
    PyObject* quad_object = PyArray_SimpleNew(3, quad_dimensions, NPY_FLOAT32);
    PyObject* metric_object = PyArray_SimpleNew(2, metric_dimensions, NPY_FLOAT32);
    if (quad_object == nullptr || metric_object == nullptr) {
        Py_XDECREF(quad_object);
        Py_XDECREF(metric_object);
        Py_DECREF(image);
        return nullptr;
    }
    auto* quads = static_cast<float*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(quad_object)));
    auto* metrics = static_cast<float*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(metric_object)));
    for (int output_index = 0; output_index < static_cast<int>(components.size()); ++output_index) {
        const ContextComponent& component = components[output_index];
        const double cosine = std::cos(component.theta);
        const double sine = std::sin(component.theta);
        const double direction_x = cosine;
        const double direction_y = sine;
        const double normal_x = -sine;
        const double normal_y = cosine;
        const double along_low = component.min_along - long_padding * tile;
        const double along_high = component.max_along + long_padding * tile;
        const double across_low = component.min_across - short_padding * tile;
        const double across_high = component.max_across + short_padding * tile;
        const double corners[4][2] = {
            {direction_x * along_low + normal_x * across_low,
             direction_y * along_low + normal_y * across_low},
            {direction_x * along_high + normal_x * across_low,
             direction_y * along_high + normal_y * across_low},
            {direction_x * along_high + normal_x * across_high,
             direction_y * along_high + normal_y * across_high},
            {direction_x * along_low + normal_x * across_high,
             direction_y * along_low + normal_y * across_high},
        };
        for (int corner = 0; corner < 4; ++corner) {
            quads[(output_index * 4 + corner) * 2] = static_cast<float>(corners[corner][0]);
            quads[(output_index * 4 + corner) * 2 + 1] =
                static_cast<float>(corners[corner][1]);
        }
        metrics[output_index * 12] = static_cast<float>(component.score);
        metrics[output_index * 12 + 1] = static_cast<float>(component.cells);
        metrics[output_index * 12 + 2] = static_cast<float>(component.seeds);
        metrics[output_index * 12 + 3] = static_cast<float>(component.density);
        metrics[output_index * 12 + 4] =
            static_cast<float>(component.orientation_resultant);
        metrics[output_index * 12 + 5] =
            static_cast<float>(component.polarity_balance);
        metrics[output_index * 12 + 6] =
            static_cast<float>(component.occupancy / component.cells);
        metrics[output_index * 12 + 7] =
            static_cast<float>(component.coherence / component.cells);
        metrics[output_index * 12 + 8] =
            static_cast<float>(component.profile_transitions);
        metrics[output_index * 12 + 9] =
            static_cast<float>(component.profile_contrast);
        metrics[output_index * 12 + 10] =
            static_cast<float>(component.profile_agreement);
        metrics[output_index * 12 + 11] =
            static_cast<float>(component.low_transition_rescue);
    }
    Py_DECREF(image);
    return Py_BuildValue("NN", quad_object, metric_object);
}

PyMethodDef methods[] = {
    {
        "propose",
        reinterpret_cast<PyCFunction>(propose),
        METH_VARARGS | METH_KEYWORDS,
        "Return STTG oriented proposals and component metrics.",
    },
    {
        "propose_context",
        reinterpret_cast<PyCFunction>(propose_context),
        METH_VARARGS | METH_KEYWORDS,
        "Return context-aware binned STTG oriented proposals and component metrics.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_sttg_native",
    "Fused native Sparse Tensor Tile Graph proposal kernel.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__sttg_native() {
    import_array();
    return PyModule_Create(&module);
}
