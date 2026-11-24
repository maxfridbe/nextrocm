# Explicit worker memory reservations

An administrator may set `memory_reservation_mb` and `memory_margin_mb` together in a local model registry entry. Both values are integer MiB. The reservation must be positive, the margin must be at least 256 MiB, and their sum must fit in an unsigned 64-bit integer. Partial, malformed or unsupported settings fail with an error rather than reverting to the default policy.

This policy is supported for non-spanning image, audio and model3d workers. It is not supported for streamed or split/spanning models, chat models or video models. It does not change a portable model card's `vram_gib`, precision, engine settings or request behavior. Without these two fields, admission retains the existing policy and 1536 MiB margin.

The reservation describes the administrator's expected full worker GPU allocation, including runtime overhead and working memory for the tested workload. Admission uses the larger of that reservation and any observed model peak, without the legacy card-size clamp, then adds the margin. A larger requirement learned from a native allocation refusal remains a lower bound. Static admission checks the selected GPU's reported capacity; scheduling checks available memory and may free eligible idle holders. A final check before worker launch uses the latest cached memory sample. Unknown capacity or free memory does not authorize a launch. Busy, pinned and held workers and external GPU reservations retain their existing protections.

These checks are not an atomic GPU reservation. Other allocations can occur after a sample. Native allocation guards, including Vox's existing 512 MiB reserve, remain essential. The explicit scheduler margin does not replace or reduce that native reserve.

For the WOPR deployment, the portable Vox INT8 card remains `vram_gib: 4.0`.
The local entry reserves 3456 MiB plus a 256 MiB margin (3712 MiB total), below
the 3840 MiB validation cap. Model/runtime/executable identities and hardware
results are recorded in the deployment record. The native 512 MiB allocation
reserve is unchanged. This setting is local to WOPR, not a generic promise that
all 4 GiB devices or workloads will fit.

An observed peak is evidence for tested requests, not a guarantee for every
request. A default output duration is not a hard ceiling. Oversized requests
remain subject to native allocation checks. Voice recordings are not cropped
to satisfy admission.
