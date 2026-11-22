# Private depth-state parallel candidate

This candidate changes only how independent depth-state dot products are assigned to threads. It does not change the ascending feature FMA order inside any dot product, the RMS reduction, the materialized normalized values, or the ascending max/exp/sum/division and weighted output order.

The frozen original `mix.hip` is included with its depth export renamed to `ni2_depthmix_baseline`. All other mix exports remain original. The new `ni2_depthmix` retains the original geometry, span and alias guards and assigns state `d` to thread `d`. A block-wide barrier publishes all state logits before thread zero performs the original softmax. All threads reach every barrier. The existing typed32 quant, primitives, state and ns sources remain unchanged.

The motivation is measured: 144 depth calls took 78.30 ms in the all-family token-zero profile. This is not a prediction of candidate speed or end-to-end improvement. At most five lanes independently execute the state dots; memory access, register use and compiler lowering still need measurement.

Build only:

```sh
SOURCE_ROOT=/absolute/extracted/source OUT_DIR=/var/home/NEW-build \
  bash tests/wopr-llm/build-inkling-depth-parallel.sh
```

The output is `libnextrocm-inkling-depth-parallel.so`, with `depth.ll`, `depth.s`, compiler/source/object/runtime identities. It replaces only the mix TU in the five-TU typed runtime. No production runtime is changed.

The new validator keeps all 36 original CPU fixtures (54 numeric paths), original 63 rejected-call checks, recovery, finite checks, redzones, repeatability, input immutability and allocation/drop gates. It adds:

- Full original-kernel bit equality on all eight original depth cases.
- Full original-kernel bit equality for all 240 combinations of count 1–5, query count 1–4, and widths 1, 3, 16, 127, 128, 129, 255, 256, 257, 2304, 8191 and 8192.
- Five alternating timing pairs for each of the 20 width-2304 shapes, with complete output equality after every timed call.

Run `validate_depth_parallel.py` with `gpu_validation.py` beside it, externally pinned runtime and original fixture manifest. Python assertions must remain enabled. The controller owns the bounded GPU run and external memory watch.

Acceptance still requires compiled ordered-arithmetic inspection, actual operator gates, full 98-array reference comparison, 80-token teacher-forced comparison, lifecycle checks and a matched unprofiled benchmark. No tolerance change, precision change or full-model speed claim follows from source review alone.
