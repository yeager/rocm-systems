/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
#define RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_

// RCCL: one connect-time decision. multiSegOn is the core's fd -1 gate, so the CTS follows registration; GIN has no
// CTS. Resiliency replays a whole slot, which a send resumed across tests cannot do yet.
static inline bool IbMsegUseMultiSeg(bool multiSegOn, bool forGin, bool resilient) {
  return multiSegOn && !forGin && !resilient;
}

#endif // RCCL_TRANSPORT_NET_IB_IB_MULTISEG_MSEG_H_
