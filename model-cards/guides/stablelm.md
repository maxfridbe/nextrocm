# StableLM and Stable Code

These models use the existing chat API (`llm.chat` or `/v1/chat/completions`) and the model controls on the home page.

- **StableLM Zephyr 3B**: general instruction/chat prompts. State the task, input and desired answer format clearly.
- **Stable Code Instruct 3B**: coding instructions, such as requesting a function or explaining supplied code.
- **Stable Code 3B**: code completion. Supply the code prefix to continue, such as `def add(a, b):`; this base model does not use an instruction/chat template.

The model-specific prompt template is selected automatically. These entries have a 4096-token context and do not advertise tool calling. They generate ordinary answers rather than a separate thinking stream. A request loads the selected model through the existing service lifecycle.

The weights use the Stability AI Community License with conditional commercial terms; see each model card's pinned license link.
