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

// Cuda antialiase parameters
struct ResizeAntiAliasParams {
  const TArray<int64_t>& input_shape;
  const TArray<int64_t>& output_shape;
  const TArray<int64_t>& input_strides;
  const TArray<fast_divmod>& output_div_pitches;
  const TArray<float, 10>& roi_vals;          // Roi on CPU
  const TArray<float>& scales_vals;           // Kernel input on CPU
  gsl::span<int64_t> bounds;                  // On Device scratch buffer, must be pre-allocated
  gsl::span<int64_t> out_of_bounds;           // OnDevice scratch buffer, must be pre-allocated
  gsl::span<const float> weight_coeffcients;  //  OnDevice scratch buffer  must be pre-allocated
  const float cubic_coeff_a;
  bool exclude_outsize;
  void* dims_mapping;  // On Device
};

/// <summary>
/// Compute window size for a given dimension scaled support value.
/// </summary>
/// <param name="scaled_support"></param>
/// <returns></returns>
inline int32_t ComputeWindowSize(float scaled_support) {
  SafeInt<int32_t> window_size = narrow<int32_t>(ceilf(scaled_support));
  return window_size * 2 + 1;
}

}  // namespace cuda
}  // namespace onnxruntime
