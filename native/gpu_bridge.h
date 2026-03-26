// smith/native/gpu_bridge.h
// Flat C API for the Metal GPU bridge. No Objective-C types cross this boundary.
// Every function takes and returns C primitives or opaque pointers.

#ifndef SMITH_GPU_BRIDGE_H
#define SMITH_GPU_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

// Storage modes for buffer allocation
#define SMITH_SHARED  0  // CPU + GPU coherent (unified memory, zero-copy)
#define SMITH_PRIVATE 1  // GPU-only (faster for intermediates)

// --- Lifecycle ---

// Initialize Metal device + command queue. Returns opaque device context.
void* smith_init(void);

// Release device context and all associated resources.
void smith_destroy(void* ctx);

// --- Device Info ---

// Returns the GPU name as a null-terminated string. Caller must free().
char* smith_device_name(void* ctx);

// Returns maximum threadgroup memory size in bytes.
uint64_t smith_max_threadgroup_memory(void* ctx);

// Returns maximum threads per threadgroup.
uint64_t smith_max_threads_per_threadgroup(void* ctx);

// --- Buffer Management ---

// Allocate a Metal buffer. mode: SMITH_SHARED or SMITH_PRIVATE.
void* smith_alloc(void* ctx, uint64_t bytes, uint32_t mode);

// Get raw pointer to shared buffer contents. Only valid for SMITH_SHARED buffers.
void* smith_buffer_contents(void* buffer);

// Get buffer length in bytes.
uint64_t smith_buffer_length(void* buffer);

// Release a buffer (decrements retain count).
void smith_release_buffer(void* buffer);

// --- Shader Library ---

// Load a precompiled .metallib from disk. Returns library pointer.
void* smith_load_library(void* ctx, const char* path);

// Load shader library from source string. Returns library pointer, or NULL on error.
// If error_out is non-NULL, writes error message (caller must free).
void* smith_compile_source(void* ctx, const char* source, char** error_out);

// Create a compute pipeline for a named kernel function. Returns pipeline pointer.
void* smith_create_pipeline(void* ctx, void* library, const char* fn_name);

// --- Compute Dispatch ---

// Opaque handle for a command encoder session.
typedef struct SmithEncoder SmithEncoder;

// Begin a new command buffer + compute encoder.
SmithEncoder* smith_begin(void* ctx);

// Bind a buffer at the given index.
void smith_set_buffer(SmithEncoder* enc, void* buffer, uint32_t index);

// Bind inline bytes at the given index. Data is copied into the command buffer.
void smith_set_bytes(SmithEncoder* enc, const void* data, uint32_t length, uint32_t index);

// Set the active compute pipeline.
void smith_set_pipeline(SmithEncoder* enc, void* pipeline);

// Set threadgroup memory size at the given index.
// Required for kernels using [[threadgroup(n)]] dynamic shared memory.
void smith_set_threadgroup_memory(SmithEncoder* enc, uint64_t length, uint32_t index);

// Dispatch with explicit grid and threadgroup dimensions.
void smith_dispatch(SmithEncoder* enc,
                    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                    uint64_t group_x, uint64_t group_y, uint64_t group_z);

// End encoding, submit, and block until GPU finishes.
void smith_end_sync(SmithEncoder* enc);

// End encoding and submit without blocking. Returns a completion token.
void* smith_end_async(SmithEncoder* enc);

// Block until an async command completes.
void smith_wait(void* token);

// --- Convenience: single-shot dispatch ---
// Combines begin + set_pipeline + set_buffers + dispatch + end_sync.
// buffers/indices arrays must have buffer_count elements.
void smith_dispatch_sync(void* ctx, void* pipeline,
                         void** buffers, uint32_t* indices, uint32_t buffer_count,
                         const void* params, uint32_t params_length, uint32_t params_index,
                         uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                         uint64_t group_x, uint64_t group_y, uint64_t group_z);

// --- Profiling ---

// Timing result from a timed dispatch.
typedef struct {
  double gpu_start;     // GPU start time in seconds (Mach absolute time)
  double gpu_end;       // GPU end time in seconds
  double gpu_ms;        // GPU duration in milliseconds
} SmithTiming;

// End encoding with timing. Blocks until complete, returns timing info.
// Caller must free the returned SmithTiming pointer.
SmithTiming* smith_end_timed(SmithEncoder* enc);

// Get current GPU memory allocation (allocated bytes on device).
uint64_t smith_allocated_size(void* ctx);

#endif // SMITH_GPU_BRIDGE_H
