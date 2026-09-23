# How to program the Tensor Memory Accelerator (TMA)

- TMA is a separate DMA engine on each SM. It's designed to copy (large) tensor blocks between GMEM and SMEM.
  - There is exactly 1 TMA per SM.
  - Only 1 thread needs to issue a TMA transfer. This is **enqueued** onto a queue of TMA transfer requests!

- PROS: the SMs can remain busy and not have to waste instructions ld/store-ing data themselves.
- CONS: hardware-wise, TMA kernels have a deeper pipeline, so they have higher latency with an empty pipeline. Good TMA kernels need to have enough requests in-flight to avoid bubbles.

The hard part is just *knowing when the job is done and all the data has arrived in SMEM*.

## Implementation Details

- `MIOC arbiter` puts TMA operations into a dedicated TMA request queue.
- The `setup and request generator` processes 1 REQUEST PER CYCLE
  - It reads `{src, dst, transfer_size, mem_bar_object}`, and prepares memory transactions for the LST unit.

- Both the pointer AND data size must be multiples of 16 bytes! *Must be 16-byte aligned addresses*.

Loads (GMEM → SMEM) go through `cuda::device::memcpy_async_tx` + a shared `cuda::barrier`.
Stores (SMEM → GMEM) go through `cuda::ptx::cp_async_bulk` + `commit_group` / `wait_group`.

# Atomic Ops

- Initialize mbarrier (object in shared memory)
- TMALoad_Global2Shared
  - expect_bytes(X)
  - Wait until all N threads have issued expect_bytes, and N*X bytes are in my shared memory [PHASE FLIP]

> Under the hood, mbarrier object has 2 counters: arriveCounter + txCounter. When BOTH are 0, it "phase-shifts" from 0->1 or 1->0.
>
> We need the phasebit because when txCounter = 0, arriveCounter gets reset back to N like what it was originally. Phasebit is the only thing that changes.
>
> This lets your mbarrier object be small enough to live in SMEM + cheap enough to reuse in a hotloop. You need the phasebit so that the state can actually recycle.

- TMAStore_Shared2Global
  - commit_group: actually execute all of the STORE transactions you've enqueued.
  - wait until 0 transactions are outstanding. IDK if the code arrived in HBM, but you can reuse the SMEM send buffers.

# Doing TMA Loads

Doing a load means I know when the bytes are done arrive into SMEM (via a signal).

```cpp
#include <cuda/barrier>
#include <cuda/ptx>

// Shared state used by the CTA
alignas(16) __shared__ char s_buf0[256];
alignas(16) __shared__ char s_buf1[256];
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
//                                      ^^^^^^^^^^^^^^^^^^^^^
//                                      Scope of the barrier: this CTA only

// Global sources (16-byte aligned)
alignas(16) __device__ char g_src0[256];
alignas(16) __device__ char g_src1[256];

// 1. Initialize barrier: expect 1 thread arrival (the leader that issues TMA)
if (threadIdx.x == 0) {
  init(
      /* barrier = */ &bar,
      /* expected_arrival_count = */ 1
      // How many threads must arrive() before the phase can flip.
      // Here: only the leader arrives. Everyone else just waits.
  );
}
__syncthreads();  // make sure init is visible before anyone uses bar

if (threadIdx.x == 0) {
  // 2. First TMA load: 256 B from g_src0 → s_buf0
  //    On completion, hardware decrements bar's tx counter by 256.
  cuda::device::memcpy_async_tx(
      /* dest  = */ s_buf0,                          // shared-memory destination
      /* src   = */ g_src0,                          // global-memory source
      /* size  = */ cuda::aligned_size_t<16>{256},   // bytes to copy (must be multiple of 16)
      /* bar   = */ bar                              // shared barrier that tracks tx completion
  );

  // 3. Second TMA load: 256 B from g_src1 → s_buf1
  cuda::device::memcpy_async_tx(
      /* dest  = */ s_buf1,
      /* src   = */ g_src1,
      /* size  = */ cuda::aligned_size_t<16>{256},
      /* bar   = */ bar
  );

  // 4. Arrive + tell the barrier we expect 256 + 256 = 512 bytes total.
  //    Phase flips only when: (arrivals == expected) AND (tx bytes arrived == expected_tx).
  auto token = cuda::device::barrier_arrive_tx(
      /* barrier            = */ bar,
      /* arrive_count_update = */ 1,     // this thread contributes 1 arrival
      /* transaction_count   = */ 512    // expect_tx: total bytes from both loads
  );

  // 5. Wait until the mbarrier has observed all 512 bytes (phase flip).
  bar.wait(
      /* arrival_token = */ cuda::std::move(token)
      // Token encodes the phase we are waiting to leave.
  );
}
```

Equivalent low-level CCCL PTX mapping (same arguments, closer to the ISA):

```cpp
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__shared__ uint64_t s_mbar;          // raw mbarrier object in SMEM
alignas(16) __shared__ char s_buf0[256];
alignas(16) __shared__ char s_buf1[256];

// 1. mbarrier.init.shared::cta.b64 [s_mbar], 1;
ptx::mbarrier_init(
    /* addr                 = */ &s_mbar,   // pointer to mbarrier in shared memory
    /* expected_arrive_count = */ 1         // N threads that will arrive
);

// 2–3. cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
ptx::cp_async_bulk(
    /* dst_space = */ ptx::space_shared,   // destination address space (SMEM)
    /* src_space = */ ptx::space_global,   // source address space (GMEM)
    /* dstMem    = */ s_buf0,              // shared dst pointer (16B-aligned)
    /* srcMem    = */ g_src0,              // global src pointer (16B-aligned)
    /* size      = */ 256,                 // bytes (multiple of 16)
    /* smem_bar  = */ &s_mbar              // mbarrier that gets complete_tx updates
);
ptx::cp_async_bulk(
    /* dst_space = */ ptx::space_shared,
    /* src_space = */ ptx::space_global,
    /* dstMem    = */ s_buf1,
    /* srcMem    = */ g_src1,
    /* size      = */ 256,
    /* smem_bar  = */ &s_mbar
);

// 4. mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
uint64_t token = ptx::mbarrier_arrive_expect_tx(
    /* sem   = */ ptx::sem_release,   // release semantics on arrive
    /* scope = */ ptx::scope_cta,     // visibility: this CTA
    /* space = */ ptx::space_shared,  // barrier lives in shared memory
    /* addr  = */ &s_mbar,            // the mbarrier object
    /* tx_count = */ 512              // expect_tx bytes (256 + 256)
);

// 5. mbarrier.try_wait.acquire.cta.shared::cta.b64
while (!ptx::mbarrier_try_wait(
           /* addr  = */ &s_mbar,   // barrier to poll
           /* state = */ token      // arrival token / phase from step 4
           )) {
  // spin until phase flips (all arrivals + all tx bytes observed)
}
```

# DOING TMA STORES:

I can tell you when data has LEFT SMEM, but I can't tell you when it arrived in HBM

- Unlike loads, we need to do TMA stores in bulk groups. We enqueue a bunch of TMA stores,
- then we do one big `cp_async_bulk_commit_group` to flush to TMA
- `wait_group 0`: "0 out of the N TMA transactions can be outstanding at this point in the code"

```cpp
#include <cuda/ptx>
namespace ptx = cuda::ptx;

alignas(16) __shared__ char s_buf0[256];
alignas(16) __shared__ char s_buf1[256];
alignas(16) __device__ char g_dst0[256];
alignas(16) __device__ char g_dst1[256];

// 1. First TMA store: 256 B from s_buf0 → g_dst0
//    Enqueued into the current bulk async-group (not flushed yet).
ptx::cp_async_bulk(
    /* dst_space = */ ptx::space_global,  // destination address space (GMEM / HBM)
    /* src_space = */ ptx::space_shared,  // source address space (SMEM)
    /* dstMem    = */ g_dst0,             // global destination (16B-aligned)
    /* srcMem    = */ s_buf0,             // shared source (16B-aligned)
    /* size      = */ 256                 // bytes (multiple of 16)
    // No mbarrier here: store completion is tracked by the bulk async-group, not a barrier.
);

// 2. Second TMA store: 256 B from s_buf1 → g_dst1
ptx::cp_async_bulk(
    /* dst_space = */ ptx::space_global,
    /* src_space = */ ptx::space_shared,
    /* dstMem    = */ g_dst1,
    /* srcMem    = */ s_buf1,
    /* size      = */ 256
);

// 3. Close the current bulk async-group and flush the enqueued stores to the TMA
ptx::cp_async_bulk_commit_group();
//   no args — commits whatever stores this thread has enqueued since the last commit

// 4. Wait until ≤ N prior bulk-groups are still outstanding.
//    N = 0 means: wait until THIS group (and all older ones) have finished reading SMEM,
//    so s_buf0 / s_buf1 can be safely reused.
ptx::cp_async_bulk_wait_group_read(
    /* pending_groups = */ ptx::n32_t<0>{}
    // Compile-time constant: max number of still-outstanding committed bulk-groups allowed.
    // 0  → drain everything before continuing
    // 1  → allow 1 prior group still in flight (useful for double-buffering)
);
```
