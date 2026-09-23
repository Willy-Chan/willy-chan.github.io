libibverbs is how you program most RDMA NICs

Libfabric is just a wrapper around libibverbs

EFA has a libibverbs interface


CREATE A QP:
- EFA has a slightly extended one compared to infiniband/roce QPs
- Other guys have to exchange QPNs and GIDs out of band...
- Address handle: tells EFA NIC where to send the packet


Workhorse: RDMA write and write with immediate data

Also RDMA read/send/recv: but less commonly used. Usually you want sender to push data and send/recv is for very small messages.

No ordering guarantees or RDMA atomics over GPU memory.





INTERNODE RDMA - we just want to get bytes from one memory to another!
TCP : buffer to kernel to socket buffer to NIC to wire
- RDMA: you give a NIC a descriptor ("read 4 MB from X on machine Y and put it in Z").
- If you set EVERYTHING up in advance explicitly, RDMA is a lot faster than flexible TCP!

Issue 1: NIC doesn't know VA of other machine. So you must REGISTER some memory:
-     pin page so OS can't move them
-     program NIC translation table so we stick to physical for this region
-     return lkey (local key for own NIC) and rkey (remote key for peer)


RDMA apps preregister the buffers, then KEEPS REUSING THEM FOREVER! 
- For GPUs, it's the same idea; you register specific GPU HBM, and then GPUDirect RDMA pulls from HBM to NIC


QUEUE PAIRS
- How do we program NICs? With "queues": we have a SQ and RQ for you to post and receive work.
- THE SAME THING AS A SOCKET: we write to a ring buffer in memory (the "queues"). My NIC continuously is POLLING that region of memory!
- Post a WQE, ring a doorbell (MMIO write to the NIC), NIC is alerted and starts processing!
- Comppletion put in the CQ: you can poll the CQ as much as you want or have it interrupt your process!

"Kinds" of QPs (i.e. RDMA-type async sockets)
- RC, UD, UC, DC: reliable (TCP-like) sockets, UDP unreliable, etc....

RC is almost always used: p2p ordered reliable hardware retransmissions. Supports RDMA reads + atomics.
- However, you need ONE QP PER PEER: full-mesh on a 10000 node cluster == shit ton of QPs. So it's better to share them in practice.



CONNECTING TWO QPs (SOCKETS):
- you need the QPN (port) + GID (IP address)
- you need to exchange this QPN + GID + Starting seqno + rkeys thorugh some other channel like MPI/TCP sockets/shared files. "BOOTSTRAPPING" the RDMA.




Physically we have 2:
- IB fabric
- ROCE fabric: cheaper + existing ethernet, use diff kind of flow/congestion controls.





SOFTWARE RDMA PROGRAMMING:
- RDMA write, read (higher latency), write with immedaite (write + 32-bit value signal)
- two sided send/recv
- Atomics: fetch-and-add and compare-and-swap. THESE RDMA OPERATIONS WILL ALWAYS GO THROUGH AS ONE CONTIGUOUS OP and can't be split up by other RDMA ops.

- lets you do distributed locks/counters/flag WITHOUT CPU involvment, but it is much slower


SPECIFIC NOTE ON ORDERING
- If you do an RDMA write; the CQ entry means the NIC IS FINISHED SENDING: doesn't mean teh remote GPU knows about it.
- With a single RC QP you can guarantee ordering, but across differnt QPs tehre's no ordering
- That's why people WRITE A SIGNAL AFTER THE DATA: receivers can poll that signal.


VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
We get the hardware topology
We get the software "ideas"/interface

Now people have implemetned with specific APIs:
- libibverbs (proprietary infiniband spec): how you create a QP, postsend, pollCQ, etc.    Used for mellanox with libmlx5.
<wrapper around libibverbs>
- Libfabric (openfabrics interface spec) - EVEN HIGHER LEVEL of abstraction that is "provider agnostic". No QPs: just "endpoints", and you query for (verbs/efa/tcp/xci(HPE slingshot))
- UCX


each of these has a KERNEL DRIVER CONTROL PATH, and a NIC HARDWARE DATA PATH.


VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
These APIs are called within the applications MPI/NCCL/NVSHMEM. the NCCL net plugin layer is meant to talk to the IB verbs implementations others have written.


┌─────────────────────────────────────────────────────────────┐
│  APPLICATION            PyTorch, JAX, Megatron, DeepSpeed   │
├─────────────────────────────────────────────────────────────┤
│  COLLECTIVES / PGAS     NCCL, RCCL, MPI (OpenMPI, MPICH,    │
│                         MVAPICH), NVSHMEM, SHMEM            │
├─────────────────────────────────────────────────────────────┤
│  PLUGIN BOUNDARY        NCCL net plugin API (ncclNet_v*_t)  │  set of nccl provided APIs: init, devices, listen/connect, accept,..
│                         ← a thin ABI, not a transport       │
├─────────────────────────────────────────────────────────────┤
│  PORTABLE MIDDLEWARE    libfabric (OFI), UCX                │       HIDES calls to this stuff
│                         ← optional; provider/transport      │
│                           abstraction over many fabrics     │
├─────────────────────────────────────────────────────────────┤
│  VERBS / NATIVE API     libibverbs, mlx5dv (DirectVerbs),   │
│                         DOCA, libefa, CXI, libfabric's own  │
│                         provider-internal APIs              │   libefa doesn't use QPs
├─────────────────────────────────────────────────────────────┤
│  KERNEL (control only)  rdma-core / ib_core, mlx5_core,     │
│                         efa, cxi_ss1, nvidia-peermem/dma-buf│   <----THIS IS THE HARDWARE KERNEL MMIO DRIVER FOR THE DEVICE!!!!
├─────────────────────────────────────────────────────────────┤
│  HARDWARE               ConnectX-7, BlueField-3, AWS EFA,   │   efa nic doesn't use QPs
│                         Slingshot Cassini, Broadcom Thor    │
└─────────────────────────────────────────────────────────────┘

on a standard IB cluster there's no libfabric/UCX: there's a direct connection from PLUGIN -> infiniband.
-    We only use libfabric on NVIDA clusters when the NIC is like EFA or slingshot or something.
-   libefa uses SRD which is not RC QP: so we need aws-ofi-nccl todo this middlewayre translation using LIBFABRIC!


-   libfabric is the openfabric NON-NVIDIA middleware btwn NCCL PLUGIN <-> VERBS API. BETTER FOR PORTING
-     UCX     is the one favored by NVIDIA/IBM: has own features. BETTER FOR PERF


# Here are some common "stacks"
- NVIDIA IB cluster:      pytorch -> nccl -> ...<no middleware net_ib needed>... -> libibverbs -> mlx5_core driver ops -> NIC
- AWS:                    pytorhc -> nccl -> aws-ofi-nccl/libfabric middleware -> efa verbs -> efa driver -> efa NIC
- MPI on IB type cluster:                 App → OpenMPI → UCX (UCP → rc_mlx5) → mlx5dv/libibverbs → ConnectX


4 nodes, 8 GPUs/NICs per node
- every node's NIC plugs into the same switch: WITHIN RAIN i, all nodes are 1 hop apart. "leaf switch". You can also go up for a spine switch.
-      "CLOS/fat-tree"



Obviously if you're one hop away just do that
OPTIONS TO CROSS NIC:
- GO to leaf, then spine, then leaf. If rail-only not enforced.
- NVLink relay on the same intranode context. "hierarchical" 2-level type collectives.

- Multi-rail striping: If A to B has MULTIPLE RAILS (crazy topology), we can spread a large message across ALL of them.
- Restructure the algorithm so you never do cross-rail!!!! TP DP PP all can be rail aligned.... SOFTWARE IS USUALLY CHOSEN TO RAIL ALIGN!



Network switch = leaf (rail switch)/spine (cros-rail).        BTW: can have SHARP on a rail!

NVSwitch - intranode world

PCIe switch - you have to consider NUMA AFFINITY between GPUs and NICs: If the GPU + NIC is on the same PCIe switch that's better.




cross_nic = 0 means NEVER DO OPTION A: so we need to know about rails to do that!!!!! If you NEED to do a cross-nic, the system just fails because it's not physically possible!