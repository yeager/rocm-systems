// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if CONSAN_WIDTH == 32
#define FIRST_SOURCE v2
#define SECOND_SOURCE v3
#define DUAL_RESULT v[8:9]
#define COLLISION_SHIFT 3
#else
#define FIRST_SOURCE v[2:3]
#define SECOND_SOURCE v[4:5]
#define DUAL_RESULT v[8:11]
#define COLLISION_SHIFT 4
#endif

#if CONSAN_COLLISION
  // Different lane bases make opposite ordinals address the same element.
  v_lshlrev_b32 v1, COLLISION_SHIFT, v12
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  // Preserve an independent output oracle without reading the colliding bytes.
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
#else
  // Separate waves' ranges so every result word has a deterministic value.
  v_lshlrev_b32 v1, 6, v6
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  consan_wait_all
  // The load overwrites its own address VGPR. Probes must capture it first.
  v_mov_b32 v8, v1
  CONSAN_DUAL_READ DUAL_RESULT, v8 offset0:3 offset1:5
#endif
