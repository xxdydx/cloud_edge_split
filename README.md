(not AI-generated)

# current implementation

disaggregated inference — edge device will compute the forward pass of the LLM
up to first K layers. edge device will have its own KV cache. then, the hidden
states will be sent up to the cloud, and forward pass for remaining N-K layers
will be computed.

## cons of current implementation
- every generated token costs one full round trip, edge device computes K layers -> network hop -> cloud computes N-K layers -> network hop back to edge device.
- 2 network hops per token generated. from my measurements using T4 GPU on cloud and Apple M3 CPU on edge, took 4s to generate 10 tokens with Qwen 0.5B, which is incredibly slow.
- using `Session` object; every call has the HTTP request overhead, with the usual sending headers and response. can be slow for every call, especially with autoregressive generation.

## improvements

**using websockets**
- after first TCP handshake, connection is kept alive and a persistent socket is created. for every call of sending the hidden state to cloud, HTTP request overhead is minimised, only the hidden state (or things that matter) is sent. this will increase performance. 


**speculative decoding** 
- use draft model, compute first K layers on edge device. draft up to N tokens each time, and send over to cloud device to verify.
- on cloud device, verify the full pass (Initial Prompt Tokens + N newly generated tokens). from logits generated, verify if the N tokens match with the verifier model's prediction probability distributions. at the point where the verifier model disagrees with draft model, stop the chain, take the accepted tokens + bonus token and get draft model to generate N tokens again. 
- faster than current implementation as this reduces the network hops done to the verifier model in cloud. potential con is if the draft model in edge device is bad and doesn't predict the tokens well as compared to verifier model. that will require calibration of parameters K and N.



