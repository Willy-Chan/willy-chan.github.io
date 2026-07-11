---

## layout: post

title: Layman's Tuning of Distributed Networking Communication

---

Big thanks to Chris Fregly for giving me this book! Had the lovely privelege of meeting him at a Gimlet Labs x Cerebras meetup.

Important Libraries to know: NVIDIA Magnum I/O (NCCL, GPUDirect RDMA, GPUDirect Storage) for training, NIXL for disagg inference.



## PYTORCH NETWORKING OPTIMIZATION

### Idea #1: Pipelining (i.e. overlapping communication and computation on diff. streams)

- There is a H2D engine, Kernel engine, and D2H engine. Your goal is to overlap streams on these 3 things.
  - One stream does compute operations, another does communication operations
  - If you increase a compute event between two communication events, it minimizes communication overhead: can do more stuff before needing to stop and communicate.



- Stream is just a queue of operations.
  - You can pipeline stuff by simply **using nonblocking calls** that return immediately: you can sync correctly when needed from there.
  - Avoid `torch.cuda.synchronize()` as much as possible.
  - Look at the cascading pipeline of work! See if the GPU is busy at all times.



### Idea #1.5: Minibatching

```
for minibatch in batch:
    allreduce

vs.

for range(4):
    accumulate minibatch = batch/4
    allreduce
```

Trades off space for fewer sync points.



### Idea #2: Compression

- Reduces the volume of data that needs to be transferred: compressing gradients before sending them moves a smaller amount of data.

### Idea #3: Async Transfers

- Let's not wait for the entire gradients to accumulate. PyTorch auto divides the tensors into several buckets and transmits them ASAP. Portions can begin their allreduce naturally.





## NAIVE

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_no_overlap(rank, world_size):
    # have some data, and a target

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    model = MultiLayerNet(data.size(1)).to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size

    optimizer.step()
    dist.destroy_process_group()
```

- Unoverlapped time = Forward + backward + allreduce
- Overlap time = max(forward + backward + allreduce)

### OVERLAPPED

- Just wrap the `model` with `nn.parallel.DistributedDataParallel(model)`.
- In profiler you see saw tooth of fine grained compute communication overlap

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_no_overlap(rank, world_size):
    rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    model = MultiLayerNet(1024).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    dist.destroy_process_group()
```

- Unoverlapped time = Forward + backward + allreduce



# NVIDIA Magnum IO

- Bunch of technologies that are meant to speed up data movement and access between IO devices (GPUs, CPUs, storage, and NICs).
- There are 4 fundamental components:
  - Storage I/O: GPUDirect Storage (GDS) + BlueField SNAP. Lets you access NVMe SSDs without copying through CPU host memory.
  - Network I/O: GPUDirect RDMA + NCCL + NVSHMEM + UCX + HPC-X for GPU-GPU networking
  - In-network computing: SHARP lets you do reduction in IB switches. 
    - Ethernet-based clusters use RoCEv2 for RDMA, but can lack SHARP. Indiniband is preferred because it has SHARP + better interconnects + better features
  - I/O management
    - NETQ + UFM let you get diagnostics and telemetry on the I/O fabric



## RDMA

- RDMA lets you bypass the CPU and do GPU-GPU comms directly. Avoids the CPU kernel stack and lets a NIC directly read/write application memory!
- HOW TO CHECK:

```
check container has acccess to /dev/infiniband, otherwise you could go back to TCP sockets instead of GPUDirect RDMA.

lsmod | grep nvidia_peermem
dmesg should indiate initialization

NCCL_DEBUG=INFO should confirm NET/IB paths

RDMA perftests with --use_cuda
```



- GPUDirect RDMA is NVIDIA's IMPLEMENTATION of RDMA for GPUs. 
  - IB/RoCE NIC can bypass the CPU



Pitfalls:  
- make sure pytorch not using gloo (CPU, TCP fallback) and using NCCL. NCCL is better because of RDMA









