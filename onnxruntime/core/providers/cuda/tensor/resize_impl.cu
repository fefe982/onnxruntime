// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/resize_impl.h"

namespace onnxruntime {
namespace cuda {

using onnxruntime::ResizeCoordinateTransformationMode;
using onnxruntime::ResizeNearestMode;
using onnxruntime::UpsampleMode;

struct NearestPixel_SIMPLE {
  __device__ __forceinline__ int operator()(float x_original, bool is_down_sampling) const {
    if (is_down_sampling) {
      return static_cast<int>(_Ceil(x_original));
    }
    return static_cast<int>(x_original);
  }
};

struct NearestPixel_ROUND_PREFER_FLOOR {
  __device__ __forceinline__ int operator()(float x_original, bool) const {
    if (x_original == static_cast<int>(x_original) + 0.5f) {
      return static_cast<int>(_Floor(x_original));
    }
    return static_cast<int>(roundf(x_original));
  }
};

struct NearestPixel_ROUND_PREFER_CEIL {
  __device__ __forceinline__ int operator()(float x_original, bool) const {
    return static_cast<int>(roundf(x_original));
  }
};

struct NearestPixel_FLOOR {
  __device__ __forceinline__ int operator()(float x_original, bool) const {
    return static_cast<int>(_Floor(x_original));
  }
};

struct NearestPixel_CEIL {
  __device__ __forceinline__ int operator()(float x_original, bool) const {
    return static_cast<int>(_Ceil(x_original));
  }
};

struct TransformCoordinate_ASYMMETRIC {
  __device__ __forceinline__ float operator()(float x_resized, float x_scale, float, float, float, float) const {
    return x_resized / x_scale;
  }
};

struct TransformCoordinate_HALF_PIXEL {
  __device__ __forceinline__ float operator()(float x_resized, float x_scale, float, float, float, float) const {
    return ((x_resized + 0.5f) / x_scale) - 0.5f;
  }
};

struct TransformCoordinate_PYTORCH_HALF_PIXEL {
  __device__ __forceinline__ float operator()(float x_resized, float x_scale, float length_resized, float,
                                              float, float) const {
    return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
  }
};

struct TransformCoordinate_TF_HALF_PIXEL_FOR_NN {
  __device__ __forceinline__ float operator()(float x_resized, float x_scale, float, float, float, float) const {
    return (x_resized + 0.5f) / x_scale;
  }
};

struct TransformCoordinate_ALIGN_CORNERS {
  __device__ __forceinline__ float operator()(float x_resized, float, float length_resized, float length_original,
                                              float, float) const {
    return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
  }
};

struct TransformCoordinate_TF_CROP_AND_RESIZE {
  __device__ __forceinline__ float operator()(float x_resized, float, float length_resized, float length_original,
                                              float roi_start, float roi_end) const {
    auto orig = length_resized > 1
                    ? roi_start * (length_original - 1) +
                          (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                    : 0.5 * (roi_start + roi_end) * (length_original - 1);
    return static_cast<float>(orig);
  }
};

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...) \
  case enum_type: {                                      \
    using HINT = type;                                   \
    return __VA_ARGS__();                                \
  }

#define CASE_TYPE_COORD(enum_type, type, ...) \
  CASE_TYPE_USING_HINT(enum_type, type, coord_t, __VA_ARGS__)

#define DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(TYPE, ...)                                                                      \
  [&] {                                                                                                                                \
    const auto& the_type = TYPE;                                                                                                       \
    /* don't use TYPE again in case it is an expensive or side-effect op */                                                            \
    switch (the_type) {                                                                                                                \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::HALF_PIXEL, TransformCoordinate_HALF_PIXEL, __VA_ARGS__)                     \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::ASYMMETRIC, TransformCoordinate_ASYMMETRIC, __VA_ARGS__)                     \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL, TransformCoordinate_PYTORCH_HALF_PIXEL, __VA_ARGS__)     \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::ALIGN_CORNERS, TransformCoordinate_ALIGN_CORNERS, __VA_ARGS__)               \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN, TransformCoordinate_TF_HALF_PIXEL_FOR_NN, __VA_ARGS__) \
      CASE_TYPE_COORD(ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE, TransformCoordinate_TF_CROP_AND_RESIZE, __VA_ARGS__)     \
      default:                                                                                                                         \
        ORT_THROW("unknown ResizeCoordinateTransformationMode");                                                                       \
    }                                                                                                                                  \
  }()

#define CASE_TYPE_NEAREST(enum_type, type, ...) \
  CASE_TYPE_USING_HINT(enum_type, type, nearest_t, __VA_ARGS__)

#define DISPATCH_RESIZE_NEAREST_MODE(TYPE, ...)                                                              \
  [&] {                                                                                                      \
    const auto& the_type = TYPE;                                                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */                                  \
    switch (the_type) {                                                                                      \
      CASE_TYPE_NEAREST(ResizeNearestMode::SIMPLE, NearestPixel_SIMPLE, __VA_ARGS__)                         \
      CASE_TYPE_NEAREST(ResizeNearestMode::ROUND_PREFER_FLOOR, NearestPixel_ROUND_PREFER_FLOOR, __VA_ARGS__) \
      CASE_TYPE_NEAREST(ResizeNearestMode::ROUND_PREFER_CEIL, NearestPixel_ROUND_PREFER_CEIL, __VA_ARGS__)   \
      CASE_TYPE_NEAREST(ResizeNearestMode::FLOOR, NearestPixel_FLOOR, __VA_ARGS__)                           \
      CASE_TYPE_NEAREST(ResizeNearestMode::CEIL, NearestPixel_CEIL, __VA_ARGS__)                             \
      default:                                                                                               \
        ORT_THROW("unknown ResizeNearestMode");                                                              \
    }                                                                                                        \
  }()

struct NearestMappingInfo {
  int origin_;
  int extrapolate_;
};

template <typename T, typename CudaFunctionOriginalCoordinate, typename CudaFunctionNearestPixel>
__global__ void _ResizeNearestMappingKernel2D(
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const float scales_height, const float scales_width,
    const float roi_start_height, const float roi_end_height,
    const float roi_start_width, const float roi_end_width,
    const bool extrapolation_enabled,
    const CudaFunctionOriginalCoordinate& transform_coordinate,
    const CudaFunctionNearestPixel& calc_nearest_pixel,
    NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, output_height + output_width);
  if (id >= 0 && id < output_height) {  // for Height
    int dim = id;

    // only apply co-ordinate transformation if scale != 1.0
    if (scales_height == 1.0f) {
      dims_mapping[id].extrapolate_ = 0;
    } else {
      float orig_coord = transform_coordinate(static_cast<float>(dim), scales_height, static_cast<float>(output_height),
                                              static_cast<float>(input_height), roi_start_height, roi_end_height);
      dims_mapping[id].extrapolate_ = static_cast<int>(
          extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_height - 1)));
      dim = calc_nearest_pixel(orig_coord, scales_height < 1);
      if (dim >= input_height) dim = input_height - 1;
      if (dim < 0) dim = 0;
    }

    dims_mapping[id].origin_ = dim;
  } else {
    int dim = id - output_height;

    // only apply co-ordinate transformation if scale != 1.0
    if (scales_width == 1.0f) {
      dims_mapping[id].extrapolate_ = 0;
    } else {
      float orig_coord = transform_coordinate(static_cast<float>(dim), scales_width, static_cast<float>(output_width),
                                              static_cast<float>(input_width), roi_start_width, roi_end_width);
      dims_mapping[id].extrapolate_ = static_cast<int>(
          extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_width - 1)));
      dim = calc_nearest_pixel(orig_coord, scales_width < 1);
      if (dim >= input_width) dim = input_width - 1;
      if (dim < 0) dim = 0;
    }

    dims_mapping[id].origin_ = dim;
    return;
  }
}

template <typename T, typename CudaFunctionOriginalCoordinate, typename CudaFunctionNearestPixel>
__global__ void _ResizeNearestMappingKernel(
    const size_t rank,
    const TArray<int64_t> input_shape,
    const TArray<int64_t> output_shape,
    const TArray<float> scales,
    const TArray<float, 10> roi,
    const size_t total_dim_sum,
    bool extrapolation_enabled,
    const CudaFunctionOriginalCoordinate& transform_coordinate,
    const CudaFunctionNearestPixel& calc_nearest_pixel,
    int64_t* prefix_dim_sum,
    NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, total_dim_sum);
  int64_t dim_sum = 0;
  for (int axis = 0; axis < rank; ++axis) {
    if (id == dim_sum) {
      prefix_dim_sum[axis] = dim_sum;
    }
    if (id >= dim_sum && id < dim_sum + output_shape[axis]) {
      int dim = id - dim_sum;

      // only apply co-ordinate transformation if scale != 1.0
      if (scales[axis] == 1.0f) {
        dims_mapping[id].extrapolate_ = 0;
      } else {
        float orig_coord = transform_coordinate(static_cast<float>(dim), scales[axis], static_cast<float>(output_shape[axis]),
                                                static_cast<float>(input_shape[axis]), roi[axis], roi[axis + rank]);
        dims_mapping[id].extrapolate_ = static_cast<int>(extrapolation_enabled && (orig_coord < 0.f || orig_coord > static_cast<float>(input_shape[axis] - 1)));
        dim = calc_nearest_pixel(orig_coord, scales[axis] < 1);
        if (dim >= input_shape[axis]) dim = input_shape[axis] - 1;
        if (dim < 0) dim = 0;
      }

      dims_mapping[id].origin_ = dim;
      return;
    }
    dim_sum += output_shape[axis];
  }
}

template <typename T, bool UseExtrapolation>
__global__ void _ResizeNearestKernel2D(
    const int64_t output_height, const int64_t output_width,
    const int64_t input_stride_image, const int input_stride_row,
    const fast_divmod output_stride_image, const fast_divmod output_stride_row,
    const T* input_data, T* output_data, const size_t N,
    const T extrapolation_value, const NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int imageid, h, w, output_index;
  output_stride_image.divmod(static_cast<int>(id), imageid, output_index);
  output_stride_row.divmod(output_index, h, w);
  if (UseExtrapolation) {
    if (dims_mapping[h].extrapolate_ + dims_mapping[output_height + w].extrapolate_) {
      output_data[id] = extrapolation_value;
      return;
    }
  }
  int input_index = input_stride_image * imageid +
                    input_stride_row * dims_mapping[h].origin_ +
                    dims_mapping[output_height + w].origin_;
  output_data[id] = input_data[input_index];
}

template <typename T>
__global__ void _ResizeNearestKernel(
    const int rank,
    const TArray<int64_t> input_strides,
    const TArray<fast_divmod> output_div_pitches,
    const T* input_data,
    T* output_data,
    const size_t N,
    const T extrapolation_value,
    const int64_t* prefix_dim_sum,
    const NearestMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int output_index = static_cast<int>(id);
  int input_index = 0;
  int extrapolation_occured = 0;
  for (int axis = 0; axis < rank; ++axis) {
    int dim = 0;
    output_div_pitches[axis].divmod(output_index, dim, output_index);
    const NearestMappingInfo& mi = dims_mapping[prefix_dim_sum[axis] + dim];
    extrapolation_occured += mi.extrapolate_;
    input_index += input_strides[axis] * mi.origin_;
  }
  output_data[id] = extrapolation_occured ? extrapolation_value : input_data[input_index];
}

struct LinearMappingInfo {
  int origin_;
  float weight_;
  int extrapolate_;
};

template <typename T, typename CudaFunctionOriginalCoordinate>
__global__ void _ResizeBilinearCoordinateMapping(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    float scale_height, float scale_width,
    float roi_height_start, float roi_height_end,
    float roi_width_start, float roi_width_end,
    const size_t SumHW, bool extrapolation_enabled,
    const CudaFunctionOriginalCoordinate& transform_coordinate,
    LinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);
  if (id < output_height) {  //  y = id
    float input_y = scale_height == 1 ? static_cast<float>(id) : transform_coordinate(static_cast<float>(id), scale_height, static_cast<float>(output_height), static_cast<float>(input_height), roi_height_start, roi_height_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_y < 0 || input_y > static_cast<float>(input_height - 1)));
    input_y = max(0.0f, min(input_y, static_cast<float>(input_height - 1)));
    int y_int = static_cast<int>(input_y);
    dims_mapping[id].origin_ = y_int;
    dims_mapping[id].weight_ = (y_int >= input_height - 1) ? 0.5f : input_y - y_int;
  } else {  // x = id - output_height
    float input_x = scale_width == 1 ? static_cast<float>(id - output_height) : transform_coordinate(static_cast<float>(id - output_height), scale_width, static_cast<float>(output_width), static_cast<float>(input_width), roi_width_start, roi_width_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_x < 0 || input_x > static_cast<float>(input_width - 1)));
    input_x = max(0.0f, min(input_x, static_cast<float>(input_width - 1)));
    int x_int = static_cast<int>(input_x);
    dims_mapping[id].origin_ = x_int;
    dims_mapping[id].weight_ = (x_int >= input_width - 1) ? 0.5f : input_x - x_int;
  }
}

// Antialiasing filters
struct BilinearFilter {
  __device__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

struct BiCubicFilter {
  __device__ float operator()(float x, float cubic_coeff_a) const {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return ((cubic_coeff_a + 2.0f) * x - (cubic_coeff_a + 3.0f)) * x * x + 1;
    }
    if (x < 2.0f) {
      return (((x - 5.0f) * x + 8.f) * x - 4.f) * cubic_coeff_a;
    }
    return 0.0f;
  }
};

struct TriLinearFilter {
  __device__ float operator()(float x, float /* cubic_coeff_a */) const {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

/// <summary>
/// This function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2]
/// 2. out_of_bounds: int64_t[output_size]
/// 3. scale_data: T[output_size * window_size]
///
/// Template parameter AccumType
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__device__ void SetupUpsampleFilterAnitAliasImpl(
    int id,
    int64_t input_size, int64_t output_size,
    float inv_scale,
    float roi_start, float roi_end,
    float scaled_support, int32_t window_size, bool exclude_outside,
    float cubic_coeff_a,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* scale_data) {
  CudaFunctionOriginalCoordinate transform_coordinate{};
  Filter filter{};

  const float scale = 1.0f / inv_scale;
  const float center = 0.5f + (scale == 1.0f) ? static_cast<float>(id)
                                              : get_original_coordinate(static_cast<float>(id), inv_scale,
                                                                        static_cast<float>(output_size),
                                                                        static_cast<float>(input_size),
                                                                        roi_start, roi_end);

  if (center - 0.5f < 0 || center - 0.5f > static_cast<float>(input_size - 1)) {
    out_of_bounds[id] = id;
  } else {
    out_of_bounds[id] = -1;
  }

  float total_weight = 0.0;

  auto fmin = _Floor(center - scaled_support + 0.5f);
  auto fmax = _Floor(center + scaled_support + 0.5f);

  int64_t min_real = static_cast<int64_t>(fmin);
  int64_t max_real = static_cast<int64_t>(fmax);
  int64_t min_cut = _Max(min_real, 0LL);
  int64_t max_cut = _Min(max_real, input_size);

  auto min_val = exclude_outside ? min_cut : min_real;
  auto max_val = exclude_outside ? max_cut : max_real;
  bounds[id * 2] = min_cut;
  bounds[id * 2 + 1] = max_cut;

  auto* scale_buffer = reinterpret_cast<float*>(&scale_data[id * window_size]);
  int64_t x = 0;
  max_val -= min_val;
  for (; x < max_val; x++) {
    float w = filter((x + min_val - center + 0.5f) * inv_scale, cubic_coeff_a);
    scale_buffer[x] = w;
    total_weight += w;
  }

  if (!exclude_outside) {
    int64_t neg_xsize = min_val < 0 ? -min_val : 0;
    for (x = 0; x < neg_xsize; x++) {
      scale_buffer[neg_xsize] += scale_buffer[x];
    }

    int64_t bound_size =
        max_val + min_val > input_size ? max_val + min_val - input_size : 0;
    for (x = max_val - bound_size; x < max_val; x++) {
      scale_buffer[max_val - bound_size - 1] +=
          scale_buffer[x];
    }

    for (x = 0; (neg_xsize | bound_size) > 0 && x < max_cut - min_cut; x++) {
      scale_buffer[x] = scale_buffer[x + neg_xsize];
    }
  }

  const float total_weight_inv = (total_weight == 0.0f) ? 1.f : (1.f / total_weight);
  auto* scale_buffer_int = reinterpret_cast<int32_t*>(scale_buffer);
  for (x = 0; x < max_cut - min_cut; x++) {
    scale_buffer[x] *= total_weight_inv;

    // normalize the scale to 1 << 22 for int8/uint8
    if constexpr (std::is_same<AccumType, int32_t>::value) {
      scale_buffer_int[x] = static_cast<int32_t>(_Round(scale_buffer[x] * ConstValue::mag_factor * 2.f));
    }
  }
}

/// This kernel computes antialias filter for bilinear or bicubic upsampling.
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the two dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the two dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the two dimensions
/// Buffers layout [h_data, w_data]
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupBilinearUpsampleFilterAntiAlias(
    std::tuple<int64_t, int64_t> input_dims,       // h, w
    std::tuple<int64_t, int64_t> output_dims,      // h, w
    std::tuple<float, float> inv_scale_vals,       // h, w
    std::tuple<float, float> roi_start_vals,       // h, w
    std::tuple<float, float> roi_end_vals,         // h, w
    std::tuple<float, float> dim_scaled_support,   // Pre-computed scaled support values h, w
    std::tuple<int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes h, w
    float cubic_coeff_a,
    bool exclude_outside,
    const size_t SumHW,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* weighted_coefficients) {  // computed weighted coefficients
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);

  // Setup for y
  int64_t input_size = std::get<0>(input_dims);
  int64_t output_size = std::get<0>(output_dims);
  float inv_scale = std::get<0>(inv_scale_vals);
  float roi_start = std::get<0>(roi_start_vals);
  float roi_end = std::get<0>(roi_end_vals);
  float scaled_support = std::get<0>(dim_scaled_support);
  int32_t window_size = std::get<0>(dim_window_size);

  // id >= output_height
  if (id >= std::get<0>(output_dims)) {
    // Setup for w
    // w = id - output_height
    id = id - std::get<0>(output_dims);
    input_size = std::get<1>(input_dims);
    output_size = std::get<1>(output_dims);
    inv_scale = std::get<1>(inv_scale_vals);
    roi_start = std::get<1>(roi_start_vals);
    roi_end = std::get<1>(roi_end_vals);

    scaled_support = std::get<1>(dim_scaled_support);
    window_size = std::get<1>(dim_window_size);

    // Adjust buffer positions
    bounds += (output_size * 2);
    out_of_bounds += output_size;
    weighted_coefficients += (output_size * window_size);
  }

  SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
      id,
      input_size, output_size,
      inv_scale,
      roi_start, roi_end,
      scaled_support, window_size,
      exclude_outside,
      cubic_coeff_a,
      bounds,
      out_of_bounds,
      weighted_coefficients);
}

/// <summary>
/// Compute AntiAlias filter for trilinear upsampling, all in one go
/// The function expects the following buffers to be pre-allocated on device
/// 1. bounds: int64_t[output_size * 2] for each of the three dimensions
/// 2. out_of_bounds: int64_t[output_size] for each of the three dimensions
/// 3. scale_data: AccumType[output_size * window_size] for each of the three dimensions
/// Each kind of buffer contains data for all 3 dims.
/// Buffers layout [d_data, h_data, w_data]
/// </summary>
template <typename AccumType, typename Filter, typename CudaFunctionOriginalCoordinate>
__global__ void _SetupTrilinerarUpsampleFilterAntiAlias(
    std::tuple<int64_t, int64_t, int64_t> input_dims,       // d, h, w
    std::tuple<int64_t, int64_t, int64_t> output_dims,      // d, h, w
    std::tuple<float, float, float> inv_scale_vals,         // d, h, w
    std::tuple<float, float, float> roi_start_vals,         // d, h, w
    std::tuple<float, float, float> roi_end_vals,           // d, h, w
    std::tuple<float, float, float> dim_scaled_support,     // Pre-computed scaled support values d, h, w
    std::tuple<int32_t, int32_t, int32_t> dim_window_size,  // Pre-computed windows sizes d, h, w
    const size_t SumDHW,
    int64_t* bounds,
    int64_t* out_of_bounds,
    AccumType* weighted_coefficients) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumDHW);

  const auto output_depth = std::get<0>(output_dims);

  // Setup for d by default (id < output_depth)
  int64_t input_size = std::get<0>(input_dims);
  int64_t output_size = std::get<0>(output_dims);
  float inv_scale = std::get<0>(inv_scale_vals);
  float roi_start = std::get<2>(roi_start_vals);
  float roi_end = std::get<0>(roi_end_vals);
  float scaled_support = std::get<0>(dim_scaled_support);
  int32_t window_size = std::get<0>(dim_window_size);

  if (id >= output_depth && id < (output_depth + std::get<1>(output_dims))) {
    // Setup for y - height

    // y = id - output_depth
    id = id - output_depth;
    input_size = std::get<1>(input_dims);
    output_size = std::get<1>(output_dims);
    inv_scale = std::get<1>(inv_scale_vals);
    roi_start = std::get<1>(roi_start_vals);
    roi_end = std::get<1>(roi_end_vals);

    // Adjust buffer positions
    scaled_support = std::get<1>(dim_scaled_support);
    window_size = std::get<1>(dim_window_size);

    bounds += output_size * 2;
    out_of_bounds += output_size;
    weighted_coefficients += (output_size * window_size);

  } else if (id > output_depth) {  // means we are out of bounds for the second for the first if on the right side
    // Setup for x

    // x = id - output_depth - output_height
    id = id - output_depth - std::get<1>(output_dims);
    input_size = std::get<2>(input_dims);
    output_size = std::get<2>(output_dims);
    inv_scale = std::get<2>(inv_scale_vals);
    roi_start = std::get<2>(roi_start_vals);
    roi_end = std::get<2>(roi_end_vals);

    // Adjust buffer positions
    scaled_support = std::get<2>(dim_scaled_support);
    window_size = std::get<2>(dim_window_size);

    bounds += (output_size * 4);
    out_of_bounds += (output_size * 2);
    weighted_coefficients += output_size * window_size * 2;
  }

  SetupUpsampleFilterAnitAliasImpl<AccumType, Filter, CudaFunctionOriginalCoordinate>(
      id,
      input_size, output_size,
      inv_scale,
      roi_start, roi_end,
      scaled_support, window_size,
      true,                       // exclude outside for trilinear
      onnxruntime::kCubicCoeffA,  // Default value for trilinear
      bounds,
      out_of_bounds,
      weighted_coefficients);
}

// The following method supports a 2-D or 4-D input in 'Linear mode'. Last two dimension is [H, W].
// the scale values for the outer dimensions except last two are 1.
template <typename T>
__global__ void _ResizeBilinearKernel(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    fast_divmod div_output_width, fast_divmod div_output_image,
    const T* input_data, T* output_data, const size_t N,
    const T extrapolation_value,
    LinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);
  CUDA_LONG input_index = bxc * input_height * input_width;
  int output_y, output_x;
  div_output_width.divmod(output_image_index, output_y, output_x);

  if (dims_mapping[output_y].extrapolate_ || dims_mapping[output_x + output_height].extrapolate_) {
    output_data[id] = extrapolation_value;
    return;
  }
  float y_offset_0 = dims_mapping[output_y].weight_;
  int y_int = dims_mapping[output_y].origin_;
  float x_offset_0 = dims_mapping[output_x + output_height].weight_;
  int x_int = dims_mapping[output_x + output_height].origin_;
  input_index += y_int * input_width + x_int;

  T x00 = input_data[input_index];
  bool end_of_h = (y_int >= input_height - 1);
  bool end_of_w = (x_int >= input_width - 1);
  T x10 = end_of_w ? x00 : input_data[input_index + 1];
  T x01 = end_of_h ? x00 : input_data[input_index + input_width];
  T x11 = end_of_w ? x01 : (end_of_h ? x10 : input_data[input_index + input_width + 1]);

  float y_offset_1 = 1.0f - y_offset_0;
  float x_offset_1 = 1.0f - x_offset_0;
  output_data[id] =
      x00 * static_cast<T>(y_offset_1 * x_offset_1) +
      x01 * static_cast<T>(y_offset_0 * x_offset_1) +
      x10 * static_cast<T>(y_offset_1 * x_offset_0) +
      x11 * static_cast<T>(y_offset_0 * x_offset_0);
}

template <typename T, typename CudaFunctionOriginalCoordinate>
__global__ void _ResizeTrilinearCoordinateMapping(
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t output_depth, int64_t output_height, int64_t output_width,
    float scale_depth, float scale_height, float scale_width,
    float roi_depth_start, float roi_depth_end,
    float roi_height_start, float roi_height_end,
    float roi_width_start, float roi_width_end,
    const size_t SumDHW, bool extrapolation_enabled,
    const CudaFunctionOriginalCoordinate& transform_coordinate,
    LinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumDHW);
  if (id < output_depth) {  //  z = id
    float input_z = scale_depth == 1 ? static_cast<float>(id) : transform_coordinate(static_cast<float>(id), scale_depth, static_cast<float>(output_depth), static_cast<float>(input_depth), roi_depth_start, roi_depth_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_z < 0 || input_z > static_cast<float>(input_depth - 1)));
    input_z = max(0.0f, min(input_z, static_cast<float>(input_depth - 1)));
    int z_int = static_cast<int>(input_z);
    dims_mapping[id].origin_ = z_int;
    dims_mapping[id].weight_ = (z_int >= input_depth - 1) ? 0.5f : input_z - z_int;
  } else if (id >= output_depth && id < (output_depth + output_height)) {  //  y = id - output_depth
    float input_y = scale_height == 1 ? static_cast<float>(id - output_depth) : transform_coordinate(static_cast<float>(id - output_depth), scale_height, static_cast<float>(output_height), static_cast<float>(input_height), roi_height_start, roi_height_end);

    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_y < 0 || input_y > static_cast<float>(input_height - 1)));
    input_y = max(0.0f, min(input_y, static_cast<float>(input_height - 1)));
    int y_int = static_cast<int>(input_y);
    dims_mapping[id].origin_ = y_int;
    dims_mapping[id].weight_ = (y_int >= input_height - 1) ? 0.5f : input_y - y_int;
  } else {  // x = id - output_depth - output_height
    float input_x = scale_width == 1 ? static_cast<float>(id - output_depth - output_height) : transform_coordinate(static_cast<float>(id - output_depth - output_height), scale_width, static_cast<float>(output_width), static_cast<float>(input_width), roi_width_start, roi_width_end);
    dims_mapping[id].extrapolate_ = (int)(extrapolation_enabled && (input_x < 0 || input_x > static_cast<float>(input_width - 1)));
    input_x = max(0.0f, min(input_x, static_cast<float>(input_width - 1)));
    int x_int = static_cast<int>(input_x);
    dims_mapping[id].origin_ = x_int;
    dims_mapping[id].weight_ = (x_int >= input_width - 1) ? 0.5f : input_x - x_int;
  }
}

// The following method supports a 3-D or 5-D input in 'Linear mode'. Last two dimension is [D, sH, W].
// the scale values for the outer dimensions except last two are 1.
template <typename T>
__global__ void _ResizeTrilinearKernel(
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t output_depth, int64_t output_height, int64_t output_width,
    fast_divmod div_output_height, fast_divmod div_output_width, fast_divmod div_output_image,
    const T* input_data, T* output_data, const size_t N,
    const T extrapolation_value,
    LinearMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int bxc, output_image_index;
  div_output_image.divmod(id, bxc, output_image_index);
  CUDA_LONG input_index = bxc * input_depth * input_height * input_width;
  int output_z, output_y, output_x, temp;

  div_output_height.divmod(output_image_index, output_z, temp);
  div_output_width.divmod(temp, output_y, output_x);

  if (dims_mapping[output_z].extrapolate_ ||
      dims_mapping[output_y + output_depth].extrapolate_ ||
      dims_mapping[output_x + output_depth + output_height].extrapolate_) {
    output_data[id] = extrapolation_value;
    return;
  }

  float z_offset_0 = dims_mapping[output_z].weight_;
  int z_int = dims_mapping[output_z].origin_;

  float y_offset_0 = dims_mapping[output_y + output_depth].weight_;
  int y_int = dims_mapping[output_y + output_depth].origin_;

  float x_offset_0 = dims_mapping[output_x + output_depth + output_height].weight_;
  int x_int = dims_mapping[output_x + output_depth + output_height].origin_;

  input_index += z_int * input_height * input_width + y_int * input_width + x_int;

  T x000 = input_data[input_index];

  bool end_of_h = (y_int >= input_height - 1);
  bool end_of_w = (x_int >= input_width - 1);

  T x100 = end_of_w ? x000 : input_data[input_index + 1];
  T x010 = end_of_h ? x000 : input_data[input_index + input_width];
  T x110 = end_of_w ? x010 : (end_of_h ? x100 : input_data[input_index + input_width + 1]);

  bool end_of_d = (z_int >= input_depth - 1);
  if (!end_of_d) {
    input_index = input_index + input_height * input_width;
  }

  T x001 = end_of_d ? x000 : input_data[input_index];

  T x101 = end_of_w ? x001 : input_data[input_index + 1];
  T x011 = end_of_h ? x001 : input_data[input_index + input_width];
  T x111 = end_of_w ? x011 : (end_of_h ? x101 : input_data[input_index + input_width + 1]);

  float z_offset_1 = 1.0f - z_offset_0;
  float y_offset_1 = 1.0f - y_offset_0;
  float x_offset_1 = 1.0f - x_offset_0;
  output_data[id] =
      x000 * static_cast<T>(z_offset_1 * y_offset_1 * x_offset_1) +
      x010 * static_cast<T>(z_offset_1 * y_offset_0 * x_offset_1) +
      x100 * static_cast<T>(z_offset_1 * y_offset_1 * x_offset_0) +
      x110 * static_cast<T>(z_offset_1 * y_offset_0 * x_offset_0) +

      x001 * static_cast<T>(z_offset_0 * y_offset_1 * x_offset_1) +
      x011 * static_cast<T>(z_offset_0 * y_offset_0 * x_offset_1) +
      x101 * static_cast<T>(z_offset_0 * y_offset_1 * x_offset_0) +
      x111 * static_cast<T>(z_offset_0 * y_offset_0 * x_offset_0);
}

template <typename T>
__device__ __forceinline__ float CubicInterpolationRowwise(
    const T* image, int x, int y, int input_height, int input_width,
    float coeff0, float coeff1, float coeff2, float coeff3) {
  int row_index = max(0, min(y, input_height - 1)) * input_width;
  return coeff0 * static_cast<float>(image[row_index + max(0, min(x - 1, input_width - 1))]) +
         coeff1 * static_cast<float>(image[row_index + max(0, min(x, input_width - 1))]) +
         coeff2 * static_cast<float>(image[row_index + max(0, min(x + 1, input_width - 1))]) +
         coeff3 * static_cast<float>(image[row_index + max(0, min(x + 2, input_width - 1))]);
}

struct CubicMappingInfo {
  int origin_;
  int extrapolate_;
  float coeff0_;
  float coeff1_;
  float coeff2_;
  float coeff3_;
};

template <typename T, typename CudaFunctionOriginalCoordinate>
__global__ void _ResizeCubicCoordinateMapping(
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width,
    float scale_height, float scale_width,
    float roi_height_start, float roi_height_end,
    float roi_width_start, float roi_width_end,
    const size_t SumHW, bool extrapolation_enabled,
    float cubic_coeff_a, bool exclude_outside,
    const CudaFunctionOriginalCoordinate& transform_coordinate,
    CubicMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, SumHW);
  auto& dm = dims_mapping[id];
  bool is_y_axis = (id < output_height);
  int max_input_coord = static_cast<int>(is_y_axis ? input_height : input_width);

  float scale = is_y_axis ? scale_height : scale_width;
  float input_coordinat = scale == 1 ? (is_y_axis ? id : id - output_height) : transform_coordinate(static_cast<float>(is_y_axis ? id : id - output_height), scale, static_cast<float>(is_y_axis ? output_height : output_width), static_cast<float>(max_input_coord), (is_y_axis ? roi_height_start : roi_width_start), (is_y_axis ? roi_height_end : roi_width_end));
  int coord_int = static_cast<int>(_Floor(input_coordinat));
  float s_coord = abs(input_coordinat - coord_int);
  float coeff_sum = 1.0f;
  float coeff_0 = static_cast<float>(((cubic_coeff_a * (s_coord + 1) - 5 * cubic_coeff_a) * (s_coord + 1) + 8 * cubic_coeff_a) * (s_coord + 1) - 4 * cubic_coeff_a);
  float coeff_1 = static_cast<float>(((cubic_coeff_a + 2) * s_coord - (cubic_coeff_a + 3)) * s_coord * s_coord + 1);
  float coeff_2 = static_cast<float>(((cubic_coeff_a + 2) * (1 - s_coord) - (cubic_coeff_a + 3)) * (1 - s_coord) * (1 - s_coord) + 1);
  float coeff_3 = static_cast<float>(((cubic_coeff_a * (2 - s_coord) - 5 * cubic_coeff_a) * (2 - s_coord) + 8 * cubic_coeff_a) * (2 - s_coord) - 4 * cubic_coeff_a);
  if (exclude_outside) {
    coeff_0 = (coord_int - 1 < 0 || coord_int - 1 >= max_input_coord) ? 0.0 : coeff_0;
    coeff_1 = (coord_int + 0 < 0 || coord_int + 0 >= max_input_coord) ? 0.0 : coeff_1;
    coeff_2 = (coord_int + 1 < 0 || coord_int + 1 >= max_input_coord) ? 0.0 : coeff_2;
    coeff_3 = (coord_int + 2 < 0 || coord_int + 2 >= max_input_coord) ? 0.0 : coeff_3;
    coeff_sum = coeff_0 + coeff_1 + coeff_2 + coeff_3;
  }
  dm.origin_ = coord_int;
  dm.coeff0_ = coeff_0 / coeff_sum;
  dm.coeff1_ = coeff_1 / coeff_sum;
  dm.coeff2_ = coeff_2 / coeff_sum;
  dm.coeff3_ = coeff_3 / coeff_sum;
  dm.extrapolate_ = (int)(extrapolation_enabled && (input_coordinat < 0 || input_coordinat > static_cast<float>(max_input_coord - 1)));
}

template <typename T>
__global__ void _ResizeBiCubicKernel(
    int64_t input_height, int64_t input_width, int64_t output_height, int64_t output_width,
    fast_divmod div_output_width, fast_divmod div_output_image,
    const T* input_data, T* output_data, const size_t N, const T extrapolation_value,
    CubicMappingInfo* dims_mapping) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int bxc, output_image_index, output_x, output_y;
  div_output_image.divmod(id, bxc, output_image_index);
  CUDA_LONG input_index = bxc * input_height * input_width;
  div_output_width.divmod(output_image_index, output_y, output_x);

  CubicMappingInfo& y_info = dims_mapping[output_y];
  CubicMappingInfo& x_info = dims_mapping[output_x + output_height];
  if (y_info.extrapolate_ || x_info.extrapolate_) {
    output_data[id] = extrapolation_value;
    return;
  }

  float w0 = x_info.coeff0_;
  float w1 = x_info.coeff1_;
  float w2 = x_info.coeff2_;
  float w3 = x_info.coeff3_;
  int x_int = x_info.origin_;
  int y_int = y_info.origin_;
  const T* image = input_data + input_index;
  output_data[id] = y_info.coeff0_ * CubicInterpolationRowwise(image, x_int, y_int - 1, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff1_ * CubicInterpolationRowwise(image, x_int, y_int, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff2_ * CubicInterpolationRowwise(image, x_int, y_int + 1, input_height, input_width, w0, w1, w2, w3) +
                    y_info.coeff3_ * CubicInterpolationRowwise(image, x_int, y_int + 2, input_height, input_width, w0, w1, w2, w3);
}

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const gsl::span<const int64_t>& output_dims) {
  switch (upsample_mode) {
    case UpsampleMode::NN:
      return sizeof(int64_t) * output_dims.size() + sizeof(NearestMappingInfo) * static_cast<size_t>(std::accumulate(output_dims.begin(), output_dims.end(), (int64_t)0));
    case UpsampleMode::LINEAR:
      return sizeof(LinearMappingInfo) * static_cast<size_t>(std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, (int64_t)0));
    case UpsampleMode::CUBIC:
      return sizeof(CubicMappingInfo) * static_cast<size_t>(std::accumulate(output_dims.rbegin(), output_dims.rbegin() + 2, (int64_t)0));
  }
  return 0;
}

template <typename T>
void ResizeNearestImpl(
    cudaStream_t stream,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi_vals,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    ResizeCoordinateTransformationMode transform_coordinate,
    ResizeNearestMode calc_nearest_pixel,
    int64_t* /* prefix_dim_sum */,
    NearestMappingInfo* dims_mapping) {
  unsigned int blocksPerGrid = static_cast<unsigned int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  bool could2d = rank >= 2 &&
                 transform_coordinate != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE &&
                 std::all_of(scales_vals.Data(), scales_vals.Data() + (rank - 2), [](float v) { return v == 1.0; });
  if (could2d) {
    int64_t output_height = output_shape[rank - 2];
    int64_t output_width = output_shape[rank - 1];
    fast_divmod div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(static_cast<int>(output_height * output_width));
    int blocksPerDimsMappingGrid = static_cast<int>(ceil((output_height + output_width) / 32.0));

    DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(transform_coordinate, [&]() {
      DISPATCH_RESIZE_NEAREST_MODE(calc_nearest_pixel, [&]() {
        _ResizeNearestMappingKernel2D<T><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
            static_cast<int>(input_shape[rank - 2]), static_cast<int>(input_shape[rank - 1]),
            static_cast<int>(output_height), static_cast<int>(output_width),
            scales_vals[rank - 2], scales_vals[rank - 1],
            roi_vals[rank - 2], roi_vals[rank - 2 + rank],
            roi_vals[rank - 1], roi_vals[rank - 1 + rank],
            extrapolation_enabled, coord_t(), nearest_t(),
            dims_mapping);
      });
    });
    if (extrapolation_enabled) {
      _ResizeNearestKernel2D<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_height, output_width,
          input_shape[rank - 2] * input_shape[rank - 1], static_cast<int>(input_shape[rank - 1]),
          div_output_image, output_div_pitches[rank - 2],
          input_data, output_data, N,
          extrapolation_value,
          dims_mapping);
    } else {
      _ResizeNearestKernel2D<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_height, output_width,
          input_shape[rank - 2] * input_shape[rank - 1], static_cast<int>(input_shape[rank - 1]),
          div_output_image, output_div_pitches[rank - 2],
          input_data, output_data, N,
          extrapolation_value,
          dims_mapping);
    }
    return;
  }

  int64_t total_dim_sum = std::accumulate(output_shape.Data(), output_shape.Data() + rank, (int64_t)0);
  int blocksPerDimsMappingGrid = (int)(ceil(static_cast<double>(total_dim_sum) / 32));
  DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(transform_coordinate, [&]() {
    DISPATCH_RESIZE_NEAREST_MODE(calc_nearest_pixel, [&]() {
      _ResizeNearestMappingKernel<T><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
          rank, input_shape, output_shape,
          scales_vals, roi_vals,
          total_dim_sum, extrapolation_enabled,
          coord_t(), nearest_t(),
          reinterpret_cast<int64_t*>(dims_mapping),
          reinterpret_cast<NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
    });
  });
  _ResizeNearestKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      rank, input_strides, output_div_pitches,
      input_data, output_data, N,
      extrapolation_value,
      reinterpret_cast<const int64_t*>(dims_mapping),
      reinterpret_cast<const NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
  return;
}

template <typename T>
void ResizeImpl(
    cudaStream_t stream,
    const UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi_vals,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    ResizeNearestMode nearest_mode,
    void* dims_mapping) {
  bool isSame = std::all_of(scales_vals.Data(), scales_vals.Data() + rank, [](float v) { return v == 1.0f; }) &&
                (coordinate_transform_mode != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE);
  if (isSame) {
    CUDA_CALL_THROW(cudaMemcpyAsync(output_data, input_data, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    return;
  }

  if (upsample_mode == UpsampleMode::NN) {
    ResizeNearestImpl(
        stream, rank, input_shape, output_shape, input_strides, output_div_pitches,
        scales_vals, roi_vals, input_data, output_data, N,
        extrapolation_enabled, extrapolation_value, cubic_coeff_a,
        coordinate_transform_mode, nearest_mode,
        reinterpret_cast<int64_t*>(dims_mapping),
        reinterpret_cast<NearestMappingInfo*>(reinterpret_cast<int64_t*>(dims_mapping) + rank));
    return;
  }

  // We support a special case of bilinear or bicubic if the input data is 4D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  bool is_2D = (rank == 2 || rank == 4);

  // We support a special case of trilinear or tricubic if the input data is 5D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  bool is_3D = (rank == 3 || rank == 5);

  // Should not hit this as we have already validated input rank/scales and we provide verbose error messages
  // to the user.
  ORT_ENFORCE(is_2D || is_3D, "Only bilinear/trilinear and bicubic modes are supported in Resize");

  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fast_divmod div_output_image;
  if (is_2D) {
    div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(gsl::narrow_cast<int>(N));
  } else if (is_3D) {
    div_output_image = (rank > 3) ? output_div_pitches[rank - 4] : fast_divmod(gsl::narrow_cast<int>(N));
  }

  int64_t output_depth = is_3D ? output_shape[rank - 3] : 0;
  int64_t output_height = output_shape[rank - 2];
  int64_t output_width = output_shape[rank - 1];
  int blocksPerDimsMappingGrid =
      static_cast<int>(ceil((output_depth + output_height + output_width) / 32.0));

  switch (upsample_mode) {
    case UpsampleMode::LINEAR:
      if (is_2D) {
        DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(coordinate_transform_mode, [&]() {
          _ResizeBilinearCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              input_shape[rank - 2], input_shape[rank - 1],
              output_height, output_width,
              scales_vals[rank - 2], scales_vals[rank - 1],
              roi_vals[rank - 2], roi_vals[rank - 2 + rank],
              roi_vals[rank - 1], roi_vals[rank - 1 + rank],
              output_height + output_width, extrapolation_enabled, coord_t(),
              reinterpret_cast<LinearMappingInfo*>(dims_mapping));
        });
        _ResizeBilinearKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input_shape[rank - 2], input_shape[rank - 1],
            output_height, output_width,
            output_div_pitches[rank - 2], div_output_image,
            input_data, output_data, N, extrapolation_value,
            reinterpret_cast<LinearMappingInfo*>(dims_mapping));
        return;
      } else if (is_3D) {
        DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(coordinate_transform_mode, [&]() {
          _ResizeTrilinearCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              input_shape[rank - 3], input_shape[rank - 2], input_shape[rank - 1],
              output_depth, output_height, output_width,
              scales_vals[rank - 3], scales_vals[rank - 2], scales_vals[rank - 1],
              roi_vals[rank - 3], roi_vals[rank - 3 + rank],
              roi_vals[rank - 2], roi_vals[rank - 2 + rank],
              roi_vals[rank - 1], roi_vals[rank - 1 + rank],
              output_depth + output_height + output_width, extrapolation_enabled, coord_t(),
              reinterpret_cast<LinearMappingInfo*>(dims_mapping));
        });
        _ResizeTrilinearKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input_shape[rank - 3], input_shape[rank - 2], input_shape[rank - 1],
            output_depth, output_height, output_width,
            output_div_pitches[rank - 3], output_div_pitches[rank - 2], div_output_image,
            input_data, output_data, N, extrapolation_value,
            reinterpret_cast<LinearMappingInfo*>(dims_mapping));
        return;
      }
      ORT_THROW("Resize support 2-D and 3-D dimensions in LINEAR mode.");
      break;
    case UpsampleMode::CUBIC:
      if (is_2D) {
        DISPATCH_RESIZE_COORDINATE_TRANSFORMATION_MODE(coordinate_transform_mode, [&]() {
          _ResizeCubicCoordinateMapping<T><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              input_shape[rank - 2], input_shape[rank - 1],
              output_height, output_width,
              scales_vals[rank - 2], scales_vals[rank - 1],
              roi_vals[rank - 2], roi_vals[rank - 2 + rank],
              roi_vals[rank - 1], roi_vals[rank - 1 + rank],
              output_height + output_width, extrapolation_enabled,
              cubic_coeff_a, exclude_outside, coord_t(),
              reinterpret_cast<CubicMappingInfo*>(dims_mapping));
        });
        _ResizeBiCubicKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            input_shape[rank - 2], input_shape[rank - 1],
            output_height, output_width,
            output_div_pitches[rank - 2], div_output_image,
            input_data, output_data, N, extrapolation_value,
            reinterpret_cast<CubicMappingInfo*>(dims_mapping));
        return;
      }
      ORT_THROW("Resize supports only 2-D in CUBIC mode.");
    case UpsampleMode::NN:
      ORT_THROW("Only bilinear/trilinear and bicubic modes are supported in Resize");
  }
}

#define CASEA_COORD_ANTIALIAS(coordinate_mode, TransformCoordType, ...) \
  case coordinate_mode: {                                               \
    using coord_t = TransformCoordType;                                 \
    return __VA_ARGS__();                                               \
    break;                                                              \
  }

#define DISPATCH_ANTIALIAS_FILTER_SETUP(coord_enum, ...)                                                                                     \
  [&] {                                                                                                                                      \
    const auto the_type = coord_enum;                                                                                                        \
    switch (the_type) {                                                                                                                      \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::HALF_PIXEL, TransformCoordinate_HALF_PIXEL, __VA_ARGS__)                     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ASYMMETRIC, TransformCoordinate_ASYMMETRIC, __VA_ARGS__)                     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL, TransformCoordinate_PYTORCH_HALF_PIXEL, __VA_ARGS__)     \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::ALIGN_CORNERS, TransformCoordinate_ALIGN_CORNERS, __VA_ARGS__)               \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN, TransformCoordinate_TF_HALF_PIXEL_FOR_NN, __VA_ARGS__) \
      CASEA_COORD_ANTIALIAS(ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE, TransformCoordinate_TF_CROP_AND_RESIZE, __VA_ARGS__)     \
      default:                                                                                                                               \
        ORT_THROW("unknown ResizeCoordinateTransformationMode");                                                                             \
    }                                                                                                                                        \
  }()

template <class T>
void ResizeAntiAliaceImpl(
    cudaStream_t stream,
    const ResizeAntiAliasParams& params,
    const UpsampleMode upsample_mode,
    ResizeCoordinateTransformationMode coordinate_transform_mode,
    std::function<IAllocatorUniquePtr<char>(size_t bytes)> allocate_scratch,
    const std::optional<T>& extrapolation_value,
    const T* input_data,
    T* output_data,
    const size_t N) {
  const int rank = params.input_shape.size();
  const auto& input_shape = params.input_shape;
  const auto& output_shape = params.output_shape;
  const auto& output_div_pitches = params.output_div_pitches;
  const auto& roi_vals = params.roi_vals;
  const auto& scales_vals = params.scales_vals;

  using AccumType = typename onnxruntime::AccumulateType<T>::type;

  // XXX: Move the below to the CPU code.
  const bool isSame = std::all_of(scales_vals.begin(), scales_vals.end(), [](float v) { return v == 1.0f; }) &&
                      (coordinate_transform_mode != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE);
  if (isSame) {
    CUDA_CALL_THROW(cudaMemcpyAsync(output_data, input_data, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    return;
  }

  // We support a special case of bilinear or bicubic if the input data is 4D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_2D = (rank == 2 || rank == 4);

  // We support a special case of trilinear or tricubic if the input data is 5D with the outer 2 scales being 1.0
  // We would have validated the outer scale values by the time execution reaches this
  const bool is_3D = (rank == 3 || rank == 5);

  assert(is_2D || is_3D);
  assert(!(is_2D && is_3D));

  // Should not hit this as we have already validated input rank/scales and we provide verbose error messages
  // to the user.
  ORT_ENFORCE(is_2D || is_3D, "Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");

  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fast_divmod div_output_image;
  if (is_2D) {
    div_output_image = (rank > 2) ? output_div_pitches[rank - 3] : fast_divmod(gsl::narrow_cast<int>(N));
  } else if (is_3D) {
    div_output_image = (rank > 3) ? output_div_pitches[rank - 4] : fast_divmod(gsl::narrow_cast<int>(N));
  }

  const int64_t input_depth = is_3D ? input_shape[rank - 3] : 0;
  const int64_t input_height = input_shape[rank - 2];
  const int64_t input_width = input_shape[rank - 1];

  const int64_t output_depth = is_3D ? output_shape[rank - 3] : 0;
  const int64_t output_height = output_shape[rank - 2];
  const int64_t output_width = output_shape[rank - 1];
  int blocksPerDimsMappingGrid =
      static_cast<int>(ceil((output_depth + output_height + output_width) / 32.0));

  /// UpsampleBase kernel instantiation code makes sure that NN mode is not enabled when we get here.

  switch (upsample_mode) {
    case UpsampleMode::LINEAR: {
      // Compute scaled support values and windows sizes for the bilinear kernel
      const float w_scaled_support = ComputeScaledSupportValue(params.support_value, scales_vals[rank - 1]);
      const float h_scaled_support = ComputeScaledSupportValue(params.support_value, scales_vals[rank - 2]);

      const int32_t w_window_size = ComputeWindowSize(w_scaled_support);
      const int32_t h_window_size = ComputeWindowSize(h_scaled_support);

      const auto h_weighted_buffer_size = ComputeWeightedCoeffBufferSize(output_height, h_window_size);
      const auto w_weighted_buffer_size = ComputeWeightedCoeffBufferSize(output_width, w_window_size);

      if (is_2D) {
        // Allocate bilinear buffers the layout used [y_data, x_data[, z_data]]

        // allocate in out/bounds buffer
        SafeInt<int64_t> bounds_buffer_size_bytes = (SafeInt<int64_t>(output_height) + output_width) * 2 * sizeof(int64_t);
        auto bounds_buffer = allocate_scratch(bounds_buffer_size_bytes);
        SafeInt<int64_t> out_of_bounds_buffer_size_bytes = (SafeInt<int64_t>(output_height) + output_width) * sizeof(int64_t);
        auto out_of_bounds_buffer = allocate_scratch(out_of_bounds_buffer_size_bytes);

        // allocate output of bounds buffer
        const size_t bilinear_weighted_buffer_size_bytes = (h_weighted_buffer_size + w_weighted_buffer_size) *
                                                           sizeof(AccumType);
        auto weighted_buffer = allocate_scratch(bilinear_weighted_buffer_size_bytes);

        DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
          //  Data is d, h, w in tuples
          _SetupBilinearUpsampleFilterAntiAlias<AccumType,
                                                BilinearFilter,
                                                coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              std::make_tuple(input_height, input_width),
              std::make_tuple(output_height, output_width),
              std::make_tuple(scales_vals[rank - 2], scales_vals[rank - 1]),
              std::make_tuple(roi_vals[rank - 2], roi_vals[rank - 1]),                // roi starts h, w
              std::make_tuple(roi_vals[rank - 2 + rank], roi_vals[rank - 1 + rank]),  // roi ends h, w
              std::make_tuple(h_scaled_support, w_scaled_support),
              std::make_tuple(h_window_size, w_window_size),
              params.cubic_coeff_a, params.exclude_outside,
              onnxruntime::narrow<size_t>(output_height + output_width),
              reinterpret_cast<int64_t*>(bounds_buffer.get()),
              reinterpret_cast<int64_t*>(out_of_bounds_buffer.get()),
              reinterpret_cast<AccumType*>(weighted_buffer.get()));
        });
      } else if (is_3D) {
        const float d_scaled_support = ComputeScaledSupportValue(params.support_value, scales_vals[rank - 3]);
        const int32_t d_window_size = ComputeWindowSize(d_scaled_support);
        const auto d_weighted_buffer_size = ComputeWeightedCoeffBufferSize(output_height, d_window_size);
        const size_t trilinear_weighted_buffer_size_bytes = (d_weighted_buffer_size + h_weighted_buffer_size +
                                                             w_weighted_buffer_size) *
                                                            sizeof(AccumType);

        SafeInt<int64_t> bounds_buffer_size_bytes = (SafeInt<int64_t>(output_depth) + output_height + output_width) *
                                                    2 * sizeof(int64_t);
        auto bounds_buffer = allocate_scratch(bounds_buffer_size_bytes);
        SafeInt<int64_t> out_of_bounds_buffer_size_bytes = (SafeInt<int64_t>(output_depth) + output_height + output_width) *
                                                           sizeof(int64_t);
        auto out_of_bounds_buffer = allocate_scratch(out_of_bounds_buffer_size_bytes);

        auto weighted_buffer = allocate_scratch(bilinear_weighted_buffer_size_bytes);
        DISPATCH_ANTIALIAS_FILTER_SETUP(coordinate_transform_mode, [&]() {
          _SetupTrilinerarUpsampleFilterAntiAlias<AccumType,
                                                  TriLinearFilter,
                                                  coord_t><<<blocksPerDimsMappingGrid, 32, 0, stream>>>(
              std::make_tuple(input_depth, input_height, input_width),
              std::make_tuple(output_depth, output_height, output_width),
              std::make_tuple(scales_vals[rank - 3], scales_vals[rank - 2], scales_vals[rank - 1]),
              std::make_tuple(roi_vals[rank - 3], roi_vals[rank - 2], roi_vals[rank - 1]),  // roi starts d, h, w
              std::make_tuple(roi_vals[rank - 3 + rank], roi_vals[rank - 2 + rank],         // roi ends d, h, w
                              roi_vals[rank - 1 + rank]),
              std::make_tuple(d_scaled_support, h_scaled_support, w_scaled_support),
              std::make_tuple(d_window_size, h_window_size, w_window_size),
              onnxruntime::narrow<size_t>(output_depth + output_height + output_width),
              reinterpret_cast<int64_t*>(bounds_buffer.get()),
              reinterpret_cast<int64_t*>(out_of_bounds_buffer.get()),
              reinterpret_cast<AccumType*>(weighted_buffer.get()));
        });
      }
    } break;
    default:
      ORT_THROW("Only bilinear/trilinear and bicubic modes are supported in Resize anti-alias mode");
      break;
  }
}

#define SPECIALIZED_IMPL(T)                                         \
  template void ResizeImpl<T>(                                      \
      cudaStream_t stream,                                          \
      const UpsampleMode upsample_mode,                             \
      const int rank,                                               \
      TArray<int64_t>& input_shape,                                 \
      TArray<int64_t>& output_shape,                                \
      TArray<int64_t>& input_strides,                               \
      TArray<fast_divmod>& output_div_pitches,                      \
      TArray<float>& scales_vals,                                   \
      TArray<float, 10>& roi_vals,                                  \
      const T* input_data,                                          \
      T* output_data,                                               \
      const size_t N,                                               \
      bool extrapolation_enabled,                                   \
      const T extrapolation_value,                                  \
      float cubic_coeff_a,                                          \
      bool exclude_outside,                                         \
      ResizeCoordinateTransformationMode coordinate_transform_mode, \
      ResizeNearestMode nearest_mode,                               \
      void* dims_mapping);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime
