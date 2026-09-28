// The completion token behind cb.rs's Seal. The last dispatch of a proving command
// buffer's last compute encoder writes the submission's epoch to its slot, and the host
// refuses the buffer's output unless it is there. One thread; the token is (slot, epoch).
//
// Standalone: no field arithmetic, so it concatenates anywhere.

#ifndef G16_SEAL_METAL
#define G16_SEAL_METAL

#include <metal_stdlib>
using namespace metal;

kernel void g16_seal(device uint* slots [[buffer(0)]],
                     constant uint2& token [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        slots[token.x] = token.y;
    }
}

#endif
