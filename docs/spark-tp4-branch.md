# Spark TP4 RPC branch

I retain the pinned bed0a856 runtime base with the existing GLM-5.3-Flash FFN TP4, CUDA/NCCL RPC transport and bounded packed weight writes. Attention remains mirrored. I require all participating RPC components to come from the same build.

I fixed client graph-cache invalidation after a remote buffer is freed. The server clears graphs for every device on that connection; the client now tracks that generation and the dispatcher identity before reusing a graph uid.

Validation: the real CPU RPC regression fails with a missing graph uid before this change and passes after it. ARM RPC compilation passes with the pinned CUDA image. I also split RDMA send and receive completion channels and activity timestamps. The old two-thread relay stalled under backpressure; the new relay completed three 512 MiB full-duplex SHA256 checks.

I completed one four-node GLM-5.3-Flash run: 38 cells, 82 requests, six basic answers, and 24 measured cells including 32K C1 and short/8K C2/C4. Four slots and total context147456 were used. Median short C1 engine decode was6.61 tok/s; short C4 aggregate end-to-end output was20.90 tok/s. 8K/32K C1 engine prefill was277.24/234.82 tok/s. Startup was372.99s. I have not established general quality or batch invariance; concurrent answers were not byte-identical to C1.

I expose weight-loading progress in HTTP503 health responses. This stage can reach100% before the API is ready. Both CPU and ARM server builds passed, and the dashboard displayed live progress.

Related work: [RPC tensor parallelism PR26610](https://github.com/ggml-org/llama.cpp/pull/26610) and [RPC graph issue20315](https://github.com/ggml-org/llama.cpp/issues/20315). These have different reproduction conditions.
