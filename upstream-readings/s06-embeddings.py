# =============================================================================
# Upstream reading for s06 — Embedding provider with batching
# =============================================================================
#
# Source: https://github.com/HKUDS/LightRAG (commit on 2026-05-09 main)
# File:   lightrag/llm/openai.py
# Lines:  733-895 (the `openai_embed` function plus its two decorators)
#
# License: MIT (HKUDS, 2025) — same as this repo. Excerpt is verbatim aside
#          from elision comments marked "[s06: omitted ...]".
#
# This is the function our `agents/s06-embeddings/embedder_openai.go` ports.
# Read this first, then open the Go file side-by-side. Three correspondences
# are non-obvious:
#
#   1. The two decorators (`@wrap_embedding_func_with_attrs` and `@retry`)
#      together describe everything that's interesting. The function body
#      itself just builds the API params and unpacks `response.data`.
#      → @wrap_embedding_func_with_attrs(embedding_dim=1536, ...)
#         maps to our `Dim() int` method on `OpenAIEmbedder`.
#      → @retry(stop_after_attempt(3), wait_exponential(...))
#         maps to our `withRetry(3, fn)` helper in embedder.go.
#
#   2. The `texts` parameter is `list[str]` — one HTTP call, many inputs.
#      That's what makes batching cheap.
#      → Our `Embed(ctx, texts)` slices texts into BatchSize-d chunks and
#        invokes embedOnce per chunk. Upstream batches ALL texts in one
#        call (no further slicing) because the wrapper enforces an upstream
#        BatchSize via a different layer not shown here.
#
#   3. response.data[i].embedding aligns with input[i].
#      → Our embedOnce sorts by dp.Index defensively; upstream relies on
#        list-comp ordering which is also index-aligned.
#
# Tests in our embedder_test.go assert all three properties.

# -----------------------------------------------------------------------------
# Verbatim excerpt (lightrag/llm/openai.py:733-895, ~50 LOC of body shown):
# -----------------------------------------------------------------------------

# @wrap_embedding_func_with_attrs(
#     embedding_dim=1536,
#     max_token_size=8192,
#     model_name="text-embedding-3-small",
#     supports_asymmetric=True,
# )
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=60),
#     retry=(
#         retry_if_exception_type(RateLimitError)
#         | retry_if_exception_type(APIConnectionError)
#         | retry_if_exception_type(APITimeoutError)
#     ),
# )
# async def openai_embed(
#     texts: list[str],
#     model: str = "text-embedding-3-small",
#     base_url: str | None = None,
#     api_key: str | None = None,
#     embedding_dim: int | None = None,
#     max_token_size: int | None = None,
#     client_configs: dict[str, Any] | None = None,
#     token_tracker: Any | None = None,
#     use_azure: bool = False,
#     azure_deployment: str | None = None,
#     api_version: str | None = None,
#     context: str = "document",
#     query_prefix: str | None = None,
#     document_prefix: str | None = None,
# ) -> np.ndarray:
#     """Generate embeddings for a list of texts using OpenAI's API
#     with automatic text truncation. ..."""
#
#     # [s06: omitted — context-prefix logic + max_token_size truncation]
#
#     openai_async_client = create_openai_async_client(
#         api_key=api_key, base_url=base_url, ...,
#     )
#
#     async with openai_async_client:
#         api_model = (
#             azure_deployment if use_azure and azure_deployment else model
#         )
#         api_params = {
#             "model": api_model,
#             "input": texts,
#         }
#         api_params["encoding_format"] = (
#             "base64" if EMBEDDING_USE_BASE64 else "float"
#         )
#         if embedding_dim is not None:
#             api_params["dimensions"] = embedding_dim
#
#         response = await openai_async_client.embeddings.create(**api_params)
#
#         # [s06: omitted — token_tracker.add_usage(...) when configured]
#
#         return np.array(
#             [
#                 np.array(dp.embedding, dtype=np.float32)
#                 if isinstance(dp.embedding, list)
#                 else np.frombuffer(
#                     base64.b64decode(dp.embedding), dtype=np.float32
#                 )
#                 for dp in response.data
#             ]
#         )

# -----------------------------------------------------------------------------
# Reading map — what to read AFTER s06
# -----------------------------------------------------------------------------
#
# Upstream files that consume the [][]float32 vectors produced here:
#
#   lightrag/kg/nano_vector_db_impl.py
#       The vector store that calls embedding_func during Insert and Query.
#       → Maps to s07 (Cosine-similarity vector store) in this repo. Three
#         independent indices share the same EmbeddingProvider — chunks,
#         entities, relations — so swapping providers globally is one line.
#
#   lightrag/lightrag.py:~384
#       Where `embedding_func` is bound onto the LightRAG dataclass field
#       and propagated into every storage that needs it.
#       → Our s11 main pipeline.go threads the EmbeddingProvider explicitly
#         (no global state). The Phase G addendum demonstrates a one-line
#         provider swap there.
#
#   lightrag/llm/anthropic.py    [Phase G]
#   lightrag/llm/bedrock.py      [Phase G]
#       Sibling files implementing the SAME embedding-func contract for
#       other providers. Only the body changes; decorators (@retry, @wrap)
#       are identical.
#       → Our Phase G addendum will land embedder_anthropic.go and
#         embedder_bedrock.go in s06's pattern: each is one new
#         OpenAIEmbedder-shaped struct, no signature change to
#         EmbeddingProvider.
#
# What s06 deliberately omits (and why):
#
#   - encoding_format=base64 path. Upstream sends base64-encoded float32 to
#     save ~25% wire size. Our Go port uses encoding_format=float for code
#     transparency (no extra base64 → []byte → []float32 step). One
#     exercise: add a `WithBase64()` option that switches the wire format.
#
#   - max_token_size truncation. Upstream truncates inputs that exceed the
#     model's per-input cap (8192 for text-embedding-3-small). s04
#     (token-based chunking) already enforces a 1200-token chunk size, so
#     in our pipeline truncation is moot. Phase G or a Phase H exercise can
#     add tiktoken-driven pre-flight truncation.
#
#   - token_tracker. Upstream records token usage to a shared object so
#     callers can cost-bound a session. Our Go port leaves this to the
#     caller (each pipeline observer can wrap Embed and count). Phase G
#     adds a tiny `TokenTracker` interface in s11.
#
#   - Azure / `use_azure` knobs. The Azure entrypoint is a separate
#     `azure_openai_embed` upstream (lightrag/llm/openai.py:986+) that
#     internally delegates back to this body. We omit Azure from s06 and
#     ship it as a Phase G example: WithAzureDeployment + WithAPIVersion.
#
# -----------------------------------------------------------------------------
# Glossary one-liners
# -----------------------------------------------------------------------------
#
#   embedding_func        — LightRAG's pluggable embedding callable. Our
#                           `EmbeddingProvider` interface in s06.
#   wrap_embedding_func_with_attrs — Decorator that pins (embedding_dim,
#                           max_token_size, model_name) onto the function
#                           object. Replaced by `Dim()` + struct fields.
#   stop_after_attempt(3) — tenacity policy: 1 initial + 2 retries. Our
#                           `MaxRetries=3` semantics.
#   wait_exponential(...) — tenacity backoff. Our retryDelay() in
#                           embedder.go.
#   encoding_format       — "base64" | "float" wire format for vectors.
#                           s06 uses "float"; "base64" is left as exercise.
