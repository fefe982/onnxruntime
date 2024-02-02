// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ComputeContext, ProgramInfo} from '../types';

/*
template <typename T, unsigned TPB>
__global__ void FastGeluKernel(const T a, const T b, const T c, int input_length, int bias_length,
                               const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}
*/

const createFastGeluProgramInfo = (inputs: readonly TensorView[]): ProgramInfo => {
  const x = inputs[0];
  const bias = inputs[1];

  const getShaderSource = (shaderHelper: ShaderHelper): string => {};

  return {name: 'FastGelu', shaderCache: undefined, getShaderSource, getRunData: (inputs) => ({})};
};

export const fastGelu = (context: ComputeContext): void => {
  context.compute(createFastGeluProgramInfo(context.inputs));
};
