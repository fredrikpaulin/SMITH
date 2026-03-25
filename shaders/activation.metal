// smith/shaders/activation.metal
// Activation functions: relu, gelu, silu, sigmoid, tanh.
// Both forward and backward kernels.
// GELU uses the exact formula from TinyFormer:
//   GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#include <metal_stdlib>
using namespace metal;

// --- ReLU ---

kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = max(input[tid], 0.0f);
}

// relu backward: grad * (input > 0)
kernel void relu_backward(
    device const float* input     [[buffer(0)]],
    device const float* grad_out  [[buffer(1)]],
    device float* grad_in          [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    grad_in[tid] = input[tid] > 0.0f ? grad_out[tid] : 0.0f;
}

// --- GELU ---
// Forward: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

constant float SQRT_2_PI = 0.7978845608f;  // sqrt(2/pi)
constant float GELU_COEFF = 0.044715f;

kernel void gelu_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    float x = input[tid];
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + GELU_COEFF * x3);
    float t = tanh(inner);
    output[tid] = 0.5f * x * (1.0f + t);
}

// GELU backward (from TinyFormer):
// GELU'(x) = 0.5*(1+tanh(s)) + 0.5*x*(1-tanh(s)^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
// where s = sqrt(2/pi) * (x + 0.044715*x^3)
kernel void gelu_backward(
    device const float* input     [[buffer(0)]],
    device const float* grad_out  [[buffer(1)]],
    device float* grad_in          [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    float x = input[tid];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = SQRT_2_PI * (x + GELU_COEFF * x3);
    float t = tanh(inner);
    float t2 = t * t;
    float sech2 = 1.0f - t2;
    float d_inner = SQRT_2_PI * (1.0f + 3.0f * GELU_COEFF * x2);
    float term1 = 0.5f * (1.0f + t);
    float term2 = 0.5f * x * sech2 * d_inner;
    grad_in[tid] = grad_out[tid] * (term1 + term2);
}

// --- SiLU (Swish): x * sigmoid(x) ---

kernel void silu_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    float x = input[tid];
    float s = 1.0f / (1.0f + exp(-x));
    output[tid] = x * s;
}

kernel void silu_backward(
    device const float* input     [[buffer(0)]],
    device const float* grad_out  [[buffer(1)]],
    device float* grad_in          [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    float x = input[tid];
    float s = 1.0f / (1.0f + exp(-x));
    // d/dx(x*sigmoid(x)) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    //                     = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
    grad_in[tid] = grad_out[tid] * s * (1.0f + x * (1.0f - s));
}

// --- Sigmoid ---

kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = 1.0f / (1.0f + exp(-input[tid]));
}

kernel void sigmoid_backward(
    device const float* output    [[buffer(0)]],
    device const float* grad_out  [[buffer(1)]],
    device float* grad_in          [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    float s = output[tid];
    grad_in[tid] = grad_out[tid] * s * (1.0f - s);
}

// --- Tanh ---

kernel void tanh_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = tanh(input[tid]);
}

kernel void tanh_backward(
    device const float* output    [[buffer(0)]],
    device const float* grad_out  [[buffer(1)]],
    device float* grad_in          [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]])
{
    float t = output[tid];
    grad_in[tid] = grad_out[tid] * (1.0f - t * t);
}

// --- Exp / Log / Sqrt (element-wise, for autograd) ---

kernel void exp_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = exp(input[tid]);
}

kernel void log_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = log(input[tid]);
}

kernel void sqrt_forward(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint tid                   [[thread_position_in_grid]])
{
    output[tid] = sqrt(input[tid]);
}

// ============================================================
// f16 variants — half precision I/O, compute in half
// GELU/SiLU use f32 intermediates for the inner tanh/exp to avoid overflow
// ============================================================

kernel void relu_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = max(input[tid], half(0)); }
kernel void relu_backward_f16(device const half* input [[buffer(0)]], device const half* grad_out [[buffer(1)]], device half* grad_in [[buffer(2)]], uint tid [[thread_position_in_grid]]) { grad_in[tid] = input[tid] > half(0) ? grad_out[tid] : half(0); }

kernel void gelu_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float x = float(input[tid]);
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + GELU_COEFF * x3);
    output[tid] = half(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_backward_f16(device const half* input [[buffer(0)]], device const half* grad_out [[buffer(1)]], device half* grad_in [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    float x = float(input[tid]);
    float x2 = x * x; float x3 = x2 * x;
    float inner = SQRT_2_PI * (x + GELU_COEFF * x3);
    float t = tanh(inner); float sech2 = 1.0f - t * t;
    float d_inner = SQRT_2_PI * (1.0f + 3.0f * GELU_COEFF * x2);
    grad_in[tid] = half(float(grad_out[tid]) * (0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner));
}

kernel void silu_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float x = float(input[tid]);
    float s = 1.0f / (1.0f + exp(-x));
    output[tid] = half(x * s);
}

kernel void silu_backward_f16(device const half* input [[buffer(0)]], device const half* grad_out [[buffer(1)]], device half* grad_in [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
    float x = float(input[tid]);
    float s = 1.0f / (1.0f + exp(-x));
    grad_in[tid] = half(float(grad_out[tid]) * s * (1.0f + x * (1.0f - s)));
}

kernel void sigmoid_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = half(1.0f / (1.0f + exp(-float(input[tid])))); }
kernel void sigmoid_backward_f16(device const half* output [[buffer(0)]], device const half* grad_out [[buffer(1)]], device half* grad_in [[buffer(2)]], uint tid [[thread_position_in_grid]]) { float s = float(output[tid]); grad_in[tid] = half(float(grad_out[tid]) * s * (1.0f - s)); }
kernel void tanh_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = half(tanh(float(input[tid]))); }
kernel void tanh_backward_f16(device const half* output [[buffer(0)]], device const half* grad_out [[buffer(1)]], device half* grad_in [[buffer(2)]], uint tid [[thread_position_in_grid]]) { float t = float(output[tid]); grad_in[tid] = half(float(grad_out[tid]) * (1.0f - t * t)); }
kernel void exp_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = half(exp(float(input[tid]))); }
kernel void log_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = half(log(float(input[tid]))); }
kernel void sqrt_forward_f16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]], uint tid [[thread_position_in_grid]]) { output[tid] = half(sqrt(float(input[tid]))); }
