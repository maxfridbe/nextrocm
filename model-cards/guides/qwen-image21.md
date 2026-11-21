# Qwen Image 2.1 Turbo (experimental)

Use the existing image interface with model `qwen-image-2.1-turbo-q4km`. The supported text-to-image configuration is 1024×1024, eight Euler steps and CFG 1. The card fixes these settings; it is not a general image-editing or arbitrary-resolution capability claim.

The native pipeline loads the text encoder, diffusion model and VAE in stages. Model files and host memory requirements are separate from the card's 7 GiB device reservation. Enable, load and unload it through the existing model controls.

**Known numerical limitation:** the retained full 1024×1024 comparison fails the original 0.001 relative-L2 gate at late denoising stages and the final image. Deployment was explicitly accepted with that limitation. Optimizations preserve the native baseline output; that does not turn the original accuracy result into a pass.

The selected weights are under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE). It permits research/evaluation use; commercial use requires a separate license. The software workspace's license declarations do not override the model terms.
