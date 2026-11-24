// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Each entry fits callable-probe branch reach alone. Preparing both entries
// exhausts return-branch reach and exercises typed composition fallback.
#ifndef CONSAN_COMPOSITION_SITES
#define CONSAN_COMPOSITION_SITES 200
#endif

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

.macro body name
.text
.globl \name
.type \name,@function
.p2align 8
\name:
  v_mov_b32 v1, 0
  v_mov_b32 v2, 7
  .rept CONSAN_COMPOSITION_SITES
    ds_store_b32 v1, v2
  .endr
  s_wait_dscnt 0
  s_endpgm
.Lend_\name:
.size \name, .Lend_\name-\name
.endm

body cap_a
body cap_b

.macro descriptor name
.rodata
.p2align 6
.amdhsa_kernel \name
  .amdhsa_group_segment_fixed_size 256
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 15
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel
.endm

descriptor cap_a
descriptor cap_b

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: "amdgcn-amd-amdhsa--gfx1250"
amdhsa.kernels:
  - .name: cap_a
    .symbol: cap_a.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 8
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
    .args:
      - .name: seed
        .size: 4
        .offset: 0
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
  - .name: cap_b
    .symbol: cap_b.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 8
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
    .args:
      - .name: seed
        .size: 4
        .offset: 0
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
