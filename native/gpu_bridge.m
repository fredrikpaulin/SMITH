// smith/native/gpu_bridge.m
// Objective-C Metal bridge. Compiled as a shared library (dylib).
// Exposes only the C API declared in gpu_bridge.h.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gpu_bridge.h"
#include <stdlib.h>
#include <string.h>

// --- Internal context ---

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
} SmithContext;

struct SmithEncoder {
  id<MTLCommandBuffer> commandBuffer;
  id<MTLComputeCommandEncoder> encoder;
  SmithContext* ctx;
};

// --- Lifecycle ---

void* smith_init(void) {
  SmithContext* ctx = calloc(1, sizeof(SmithContext));
  ctx->device = MTLCreateSystemDefaultDevice();
  if (!ctx->device) {
    free(ctx);
    return NULL;
  }
  ctx->queue = [ctx->device newCommandQueue];
  return ctx;
}

void smith_destroy(void* ptr) {
  if (!ptr) return;
  SmithContext* ctx = (SmithContext*)ptr;
  // MTLCreateSystemDefaultDevice returns +1, newCommandQueue returns +1.
  // C struct fields are __unsafe_unretained so we must release manually.
  if (ctx->queue) CFRelease((__bridge CFTypeRef)ctx->queue);
  if (ctx->device) CFRelease((__bridge CFTypeRef)ctx->device);
  ctx->queue = nil;
  ctx->device = nil;
  free(ctx);
}

// --- Device Info ---

char* smith_device_name(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  NSString* name = ctx->device.name;
  const char* utf8 = [name UTF8String];
  char* copy = malloc(strlen(utf8) + 1);
  strcpy(copy, utf8);
  return copy;
}

uint64_t smith_max_threadgroup_memory(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  return ctx->device.maxThreadgroupMemoryLength;
}

uint64_t smith_max_threads_per_threadgroup(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  // Return 1D max. For compute kernels this is typically 1024 on Apple Silicon.
  return ctx->device.maxThreadsPerThreadgroup.width;
}

// --- Buffer Management ---

void* smith_alloc(void* ptr, uint64_t bytes, uint32_t mode) {
  SmithContext* ctx = (SmithContext*)ptr;
  MTLResourceOptions opts;
  if (mode == SMITH_PRIVATE) {
    opts = MTLResourceStorageModePrivate;
  } else {
    opts = MTLResourceStorageModeShared;
  }
  id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes options:opts];
  return (__bridge_retained void*)buffer;
}

void* smith_buffer_contents(void* buffer) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
  return [buf contents];
}

uint64_t smith_buffer_length(void* buffer) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
  return [buf length];
}

int64_t smith_buffer_retain_count(void* buffer) {
  if (!buffer) return -1;
  return (int64_t)CFGetRetainCount(buffer);
}

void smith_release_buffer(void* buffer) {
  if (!buffer) return;
  CFRelease(buffer);
}

// Self-test: uses same smith_alloc/smith_release_buffer code path.
// Returns bytes freed (should be >= 1MB if release works).
int64_t smith_test_release(void* ptr) {
  @autoreleasepool {
    SmithContext* ctx = (SmithContext*)ptr;
    uint64_t before = ctx->device.currentAllocatedSize;
    void* raw = smith_alloc(ptr, 1048576, 0);
    uint64_t after_alloc = ctx->device.currentAllocatedSize;
    smith_release_buffer(raw);
    uint64_t after_release = ctx->device.currentAllocatedSize;
    return (int64_t)(after_alloc - after_release);
  }
}

// Self-test 2: alloc a buffer, USE it in a Metal command, then release.
// Returns bytes freed. If 0, Metal commands add retains that prevent release.
int64_t smith_test_release_after_use(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  void* raw = smith_alloc(ptr, 1048576, 0);
  uint64_t after_alloc = ctx->device.currentAllocatedSize;

  // Use the buffer in a trivial Metal command inside @autoreleasepool.
  // The pool drains the autoreleased cmdBuf/enc, and ARC releases the local
  // variables when they go out of scope — so the command buffer is fully
  // deallocated BEFORE we measure currentAllocatedSize.
  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)raw;
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  } // cmdBuf deallocated here → releases its retain on our buffer

  smith_release_buffer(raw);
  uint64_t after_release = ctx->device.currentAllocatedSize;
  return (int64_t)(after_alloc - after_release);
}

// --- Shader Library ---

void* smith_load_library(void* ptr, const char* path) {
  SmithContext* ctx = (SmithContext*)ptr;
  NSString* nsPath = [NSString stringWithUTF8String:path];
  NSURL* url = [NSURL fileURLWithPath:nsPath];
  NSError* error = nil;
  id<MTLLibrary> library = [ctx->device newLibraryWithURL:url error:&error];
  if (error) {
    NSLog(@"smith: failed to load library at %@: %@", nsPath, error);
    return NULL;
  }
  return (__bridge_retained void*)library;
}

void* smith_compile_source(void* ptr, const char* source, char** error_out) {
  SmithContext* ctx = (SmithContext*)ptr;
  NSString* nsSource = [NSString stringWithUTF8String:source];
  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeFast;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
  }
  NSError* error = nil;
  id<MTLLibrary> library = [ctx->device newLibraryWithSource:nsSource options:opts error:&error];
  if (error) {
    if (error_out) {
      const char* msg = [[error localizedDescription] UTF8String];
      *error_out = malloc(strlen(msg) + 1);
      strcpy(*error_out, msg);
    }
    if (!library) return NULL;
  }
  return (__bridge_retained void*)library;
}

void* smith_create_pipeline(void* ptr, void* lib, const char* fn_name) {
  SmithContext* ctx = (SmithContext*)ptr;
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)lib;
  NSString* name = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) {
    NSLog(@"smith: kernel function '%@' not found in library", name);
    return NULL;
  }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
  if (error) {
    NSLog(@"smith: failed to create pipeline for '%@': %@", name, error);
    return NULL;
  }
  return (__bridge_retained void*)pipeline;
}

// --- Compute Dispatch ---

SmithEncoder* smith_begin(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  SmithEncoder* enc = calloc(1, sizeof(SmithEncoder));
  enc->ctx = ctx;
  // [commandBuffer] and [computeCommandEncoder] return autoreleased objects.
  // ARC does NOT manage id fields in C structs, so we must retain manually.
  // The @autoreleasepool drains the autorelease +1 immediately; CFRetain keeps
  // the objects alive past the pool. CFRelease in smith_end_* brings count to 0.
  @autoreleasepool {
    enc->commandBuffer = [ctx->queue commandBuffer];
    CFRetain((__bridge CFTypeRef)enc->commandBuffer);
    enc->encoder = [enc->commandBuffer computeCommandEncoder];
    CFRetain((__bridge CFTypeRef)enc->encoder);
  }
  return enc;
}

void smith_set_buffer(SmithEncoder* enc, void* buffer, uint32_t index) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
  [enc->encoder setBuffer:buf offset:0 atIndex:index];
}

void smith_set_bytes(SmithEncoder* enc, const void* data, uint32_t length, uint32_t index) {
  [enc->encoder setBytes:data length:length atIndex:index];
}

void smith_set_pipeline(SmithEncoder* enc, void* pipeline) {
  id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline;
  [enc->encoder setComputePipelineState:pso];
}

void smith_set_threadgroup_memory(SmithEncoder* enc, uint64_t length, uint32_t index) {
  [enc->encoder setThreadgroupMemoryLength:length atIndex:index];
}

void smith_dispatch(SmithEncoder* enc,
                    uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                    uint64_t group_x, uint64_t group_y, uint64_t group_z) {
  MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
  MTLSize group = MTLSizeMake(group_x, group_y, group_z);
  [enc->encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

void smith_end_sync(SmithEncoder* enc) {
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = enc->encoder;
    id<MTLCommandBuffer> cmdBuf = enc->commandBuffer;
    CFRelease((__bridge CFTypeRef)encoder);
    CFRelease((__bridge CFTypeRef)cmdBuf);
    enc->encoder = nil;
    enc->commandBuffer = nil;
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    // Explicitly nil locals INSIDE the pool so dealloc happens before
    // objc_autoreleasePoolPop(). The command buffer's dealloc autoreleases
    // its Metal buffer retains — those must land in THIS pool, not the outer
    // (never-draining) Bun FFI pool.
    encoder = nil;
    cmdBuf = nil;
  }
  free(enc);
}

void* smith_end_async(SmithEncoder* enc) {
  void* token;
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = enc->encoder;
    id<MTLCommandBuffer> cmdBuf = enc->commandBuffer;
    CFRelease((__bridge CFTypeRef)encoder);
    CFRelease((__bridge CFTypeRef)cmdBuf);
    enc->encoder = nil;
    enc->commandBuffer = nil;
    [encoder endEncoding];
    [cmdBuf commit];
    // __bridge_retained adds +1 for the raw pointer we return as token.
    token = (__bridge_retained void*)cmdBuf;
    encoder = nil;
    cmdBuf = nil;  // ARC release balances the __bridge_retained, leaving net +1 for caller
  }
  free(enc);
  return token;
}

void smith_wait(void* token) {
  if (!token) return;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)token;
    [cb waitUntilCompleted];
    cb = nil;  // dealloc inside pool so autoreleased buffer retains get drained
  }
}

// --- Convenience: single-shot dispatch ---

// --- Profiling ---

SmithTiming* smith_end_timed(SmithEncoder* enc) {
  SmithTiming* timing = calloc(1, sizeof(SmithTiming));
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = enc->encoder;
    id<MTLCommandBuffer> cmdBuf = enc->commandBuffer;
    CFRelease((__bridge CFTypeRef)encoder);
    CFRelease((__bridge CFTypeRef)cmdBuf);
    enc->encoder = nil;
    enc->commandBuffer = nil;
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    timing->gpu_start = cmdBuf.GPUStartTime;
    timing->gpu_end = cmdBuf.GPUEndTime;
    timing->gpu_ms = (timing->gpu_end - timing->gpu_start) * 1000.0;
    encoder = nil;
    cmdBuf = nil;
  }
  free(enc);
  return timing;
}

uint64_t smith_allocated_size(void* ptr) {
  SmithContext* ctx = (SmithContext*)ptr;
  return ctx->device.currentAllocatedSize;
}

// --- Convenience: single-shot dispatch ---

void smith_dispatch_sync(void* ptr, void* pipeline,
                         void** buffers, uint32_t* indices, uint32_t buffer_count,
                         const void* params, uint32_t params_length, uint32_t params_index,
                         uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                         uint64_t group_x, uint64_t group_y, uint64_t group_z) {
  SmithEncoder* enc = smith_begin(ptr);
  smith_set_pipeline(enc, pipeline);
  for (uint32_t i = 0; i < buffer_count; i++) {
    smith_set_buffer(enc, buffers[i], indices[i]);
  }
  if (params && params_length > 0) {
    smith_set_bytes(enc, params, params_length, params_index);
  }
  smith_dispatch(enc, grid_x, grid_y, grid_z, group_x, group_y, group_z);
  smith_end_sync(enc);
}
