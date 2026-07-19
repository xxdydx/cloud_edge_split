# current implementation

disaggregated inference — edge device will compute the forward pass of the LLM
up to first K layers. edge device will have its own KV cache. then, the hidden
states will be sent up to the cloud, and forward pass for remaining N-K layers
will be computed.

## improvements

using websockets

speculative decoding
