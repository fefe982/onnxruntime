// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/common.h"
#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

size_t CalcResizeBufferSize(const onnxruntime::UpsampleMode upsample_mode,
                            const gsl::span<const int64_t>& output_dims);

template <typename T>
void ResizeImpl(
    cudaStream_t stream,
    const onnxruntime::UpsampleMode upsample_mode,
    const int rank,
    TArray<int64_t>& input_shape,
    TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides,
    TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals,
    TArray<float, 10>& roi,
    const T* input_data,
    T* output_data,
    const size_t N,
    bool extrapolation_enabled,
    const T extrapolation_value,
    float cubic_coeff_a,
    bool exclude_outside,
    onnxruntime::ResizeCoordinateTransformationMode coordinate_transform_mode,
    onnxruntime::ResizeNearestMode nearest_mode,
    void* dims_mapping);

// Cuda anti-alias parameters
struct ResizeAntiAliasParams {
  gsl::span<const int64_t> input_shape;
  gsl::span<const int64_t> output_shape;
  gsl::span<const int64_t> input_strides;
  const TArray<fast_divmod>& output_div_pitches;
  gsl::span<const float> roi_vals;     // Roi on CPU
  gsl::span<const float> scales_vals;  // Kernel input on CPU
  const float support_value;
  const float cubic_coeff_a;
  const bool exclude_outside;
  const AllocatorPtr& cuda_allocator;
};

/// <summary>
/// Compute scaled support value for a given dimension inverse scale
/// </summary>
/// <param name="support_value">Support value from parameters</param>
/// <param name="inv_scale">inverse scale value comes from input/attr for</param>
/// <returns></returns>
inline float ComputeScaledSupportValue(float support_value, float inv_scale) {
  const float scale = 1.0f / inv_scale;
  float scaled_support = (scale >= 1.0f) ? (support_value * 0.5f) * scale : support_value * 0.5f;
  return scaled_support;
}

/// <summary>
/// Compute window size for a given dimension scaled support value.
/// </summary>
/// <param name="scaled_support"></param>
/// <returns></returns>
inline int32_t ComputeWindowSize(float scaled_support) {
  SafeInt<int32_t> window_size(ceilf(scaled_support));
  return window_size * 2 + 1;
}

/// <summary>
/// Computes scale buffer size in number of elements for allocation purposes.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="output_size"></param>
/// <param name="window_size"></param>
/// <returns>Number of elements to fit in the buffer</returns>
inline SafeInt<int64_t> ComputeWeightedCoeffBufferSize(int64_t output_size, int32_t window_size) {
  SafeInt<int64_t> buffer_size(output_size);
  return buffer_size * window_size;
}

/// <summary>
/// Compute a buffer for bilinear data for CUDA antialias resizing.
/// </summary>
/// <param name="output_height">Image dim</param>
/// <param name="output_width">Image dim</param>
/// <param name="inv_height_scale"></param>
/// <param name="inv_width_scale"></param>
/// <param name="support_value">unscaled support value algo dependent</param>
/// <returns>Cumulative buffer size for y and x scales in number of elements</returns>
int64_t ComputeBilinearScaleBufferSize(int64_t output_height, int64_t output_width,
                                       float inv_height_scale, float inv_width_scale, float support_value);

/// <summary>
/// Computes a buffer for trilinear data for CUDA anti-alias resizing
/// </summary>
/// <param name="output_height">Image dim</param>
/// <param name="output_width">Image dim</param>
/// <param name="output_depth">Image dim</param>
/// <param name="inv_height_scale">comes from kernel input/attributes</param>
/// <param name="inv_width_scale">comes from kernel input/attributes</param>
/// <param name="inv_depth_scale">comes from kernel input/attributes</param>
/// <param name="support_value">unscaled support value algo dependent</param>
/// <returns></returns>
int64_t ComputeTrilinearScaleBufferSize(int64_t output_height, int64_t output_width, int64_t output_depth,
                                        float inv_height_scale, float inv_width_scale, float inv_depth_scale,
                                        float support_value);

}  // namespace cuda
}  // namespace onnxruntime
