#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent device copy of one resident quantized tensor,
 * mirroring backend_cuda.h. On Strix Halo the "upload" writes into
 * HOST_VISIBLE|DEVICE_LOCAL memory — same physical RAM the iGPU reads,
 * so there is no PCIe copy, unlike the discrete-CUDA path. */
typedef struct ColiVkTensor ColiVkTensor;

/* Bring up instance/device/queue/pipeline. Returns 1 on success.
 * spv_path points at the compiled qmatmul.spv. */
int  coli_vk_init(const char *spv_path);
void coli_vk_shutdown(void);
int  coli_vk_available(void);
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count);
/* GLM-5.3 SwiGLU clamp for the fused gate_up kernel. 0 disables (GLM-5.2). */
void coli_vk_set_swiglu_limit(float limit);

/* VRAM pressure-proofing (both no-ops when the extension is absent):
 * alloc_priority sets the eviction-priority class of SUBSEQUENT weight uploads
 * (VK_EXT_memory_priority; scratches and the KV mirror pin themselves at 1.0) —
 * the engine brackets the bulk expert-tier fill at 0.4 so an oversubscribed heap
 * evicts cold experts, never the per-token attention working set.
 * mem_budget reports device-local usage/budget in GB (VK_EXT_memory_budget);
 * returns 0 if unavailable. */
void coli_vk_alloc_priority(float p);
int  coli_vk_mem_budget(double *used_gb, double *budget_gb);

/* 1 when resident weights go to plain DEVICE_LOCAL memory through a host staging buffer
 * (discrete card without Resizable BAR, or COLI_VK_STAGED=1); 0 = mapped host-visible
 * uploads as before. Decided once in coli_vk_init. */
int  coli_vk_staged(void);

/* Resolve the main shader path the way every engine does: COLI_VK_SHADERS (the qmatmul.spv
 * file or the directory holding it), then shaders/qmatmul.spv beside the executable
 * (Linux), then the CWD-relative default. buf must hold at least 1024 bytes. */
const char *coli_vk_default_spv(char *buf, size_t n);

/* y[S,O] = (x[S,I] @ dequant(W[O,I])^T) * scale[O].
 * fmt matches QT in glm.c: 1=int8, 2=int4. (0=f32,3=int2 fall back to CPU.)
 * First call uploads W+scales; later calls reuse the resident copy.
 * Returns 1 on success, 0 if unavailable / unsupported fmt. */
/* A whole gated-attention layer of a prefill block in one submit (see backend_vulkan.c). */
int  coli_vk_attn_block(ColiVkTensor *tq, ColiVkTensor *tk, ColiVkTensor *tv, ColiVkTensor *to,
                        const float *x, const float *qn, const float *kn, float *Kc, float *Vc, int ldt,
                        int S, int D, int H, int KV, int hd, int qdim, int rotary, int pos_base,
                        float eps, float theta, float scale, float *out, int layer);
int  coli_vk_attn_dec_ready(void);            /* decode core (attn_dec) built */
void coli_vk_kv_cut(int layer, int from);     /* host wrote K/V row `from`: resident copy ends there */
/* A whole DeltaNet layer of a prefill block in one submit (see backend_vulkan.c). */
int  coli_vk_dn_block(ColiVkTensor *proj, ColiVkTensor *outp, const float *x, const float *ba,
                      const float *wb, const float *wa,
                      const float *convw, const float *par, const float *normw, float *ring, float *state,
                      int S, int H, int conv_dim, int convk, int vh, int vk, int kdim, int vdim,
                      float eps, float qscale, float *y, int layer, int cap);
/* Resident DeltaNet state of the block (decode): on/off (-1 queries), host copy back
 * (1 if the GPU held it), forget the GPU copy, swap in the state after token cap. */
int  coli_vk_dn_resident(int on);
/* chained decode layer: DeltaNet block + its expert group in one submit, routed on the GPU */
/* decode graph: fixed command buffer per DeltaNet layer (block + tails + GPU routing + experts) */
typedef struct {
    int layer; ColiVkTensor *proj, *outp, *router, *shg, *shu, *shd;
    const float *wb, *wa, *convw, *par, *normw, *post_w, *next_w, *wsg;
    float *ring, *state, *x, *nrm, *tmp, *logits;
    int H, conv_dim, convk, vh, vk, kdim, vdim, E, K, I;
    float eps, qscale;
} ColiDecLayer;
int  coli_vk_dec_record(const ColiDecLayer *d);
int  coli_vk_dec_run(int n, const int *layers, const float *x, const float *nrm, const float *tmp, const float *logits, int H, int E);
int  coli_vk_moe_master(int layer, int E, ColiVkTensor *const *tg, ColiVkTensor *const *tu, ColiVkTensor *const *td);
int  coli_vk_moe_route(int layer, const float *logits, const float *nrm, const float *wsg, int E, int K, int D);
int  coli_vk_flush_deferred(void);
void coli_vk_defer_next_block(int on);
int  coli_vk_deferred(void);
int  coli_vk_dn_sync(int layer);
void coli_vk_dn_drop(int layer);
int  coli_vk_dn_rollback(int layer);
/* DeltaNet gated delta rule over a prefill block (see backend_vulkan.c); 0 -> CPU path. */
float *coli_vk_dn_stage(int which, size_t bytes);
/* persistent host-visible, host-cached buffer (slot 0..3) the prefill keeps its rows in;
 * blocks/groups called with exactly this pointer bind it instead of copying */
float *coli_vk_host_arena(int slot, size_t bytes);
/* one-shot tail for the next dn/attn block: x += out; n = RMSNorm(x)*(1+w); logits = n Wr^T
 * (all host-arena rows); _done() says whether it ran and clears the request */
void coli_vk_block_post(float *x, float *n, const float *w, float eps, ColiVkTensor *router, float *logits, int E);
int coli_vk_block_post_done(void);
int coli_vk_dn_ba_ready(void);
int coli_vk_fast_gemv(void);
void coli_vk_arena_cpu_wrote(const void *p);   /* the CPU (re)wrote this arena slot */
void coli_vk_expert_post(float *x, float *n, const float *w, float eps);   /* one-shot group tail */
int coli_vk_expert_post_done(void);   /* coli_vk_dn_block takes ba == NULL + wb/wa */
int  coli_vk_dn_recur(float *outv, float *state, const float *qn, const float *kn,
                      const float *vsrc, int vstride, int voff, const float *beta, const float *gexp,
                      int S, int vh, int vk, int kdim, int vdim);
/* Causal attention core of a prefill block (see backend_vulkan.c); 0 -> CPU path. */
int  coli_vk_attn_prefill(float *ctx, const float *q, const float *K, const float *V, int ldt,
                          int S, int H, int KV, int hd, int pos_base, float scale);
int  coli_vk_matmul(ColiVkTensor **tensor,
                    float *y, const float *x,
                    const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int gs);

/* Fused first half of the expert MLP in ONE dispatch (VK equivalent of
 * grouped_hidden_w4_dual): hidden[s,o] = silu(gate(x)) * up(x), reading x once for both
 * projections. D = input (hidden) dim, I = moe_inter. gate/up upload on first call.
 * Returns 0 if unavailable (no gate_up shader) / unsupported fmt so the caller falls back. */
int  coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up,
                     float *hidden, const float *x,
                     const void *gw, const float *gs,
                     const void *uw, const float *us,
                     int fmt, int S, int D, int I, int grp);

/* Full batched expert MLP for `count` experts in ONE submit, hidden staying on-device:
 * for each c, y_c = down_c(silu(gate_c(x_c)) * up_c(x_c)). x/y packed [sum(rows)*D];
 * experts are resident (gate/up: D->I, down: I->D). Mirrors coli_cuda_expert_group.
 * Returns 0 -> caller falls back to CPU. */
int  coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x);
/* Async form: _issue submits the group and returns immediately (one in flight max);
 * the caller computes its CPU share, then _take joins and reads back the packed y.
 * Both return 0 on failure (caller computes those experts on the CPU instead). */
int  coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x);
int  coli_vk_expert_group_take(float *y);

/* Prefill on device 0, S>1: x[S,D] is uploaded once, order[sum(rows)] maps
 * expert-packed rows to token pairs s*K+k. Missing pairs contribute zero.
 * weights[S,K] is reduced in ascending k order into y[S,D], using one submit
 * for up to 1024 experts. No float atomics. Returns 0 on failure; synchronous,
 * exclusive with group issue/take. The resident sum starts at zero, so adding
 * it to CPU/shared results as a block changes floating-point association. */
int coli_vk_expert_prefill(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           const int *order, const float *weights, int S, int K,
                           const float *x, float *y);

/* Upload a resident tensor without computing (expert tier: gate/up/down uploaded once,
 * then driven by coli_vk_expert_group). Returns 0 on failure/unsupported fmt. */
int  coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);

/* SECOND DEVICE (COLI_VK_DEV2): a self-contained context on another Vulkan GPU that
 * hosts ONLY tier experts and runs ONLY the async expert-group path. devidx: -1 =
 * auto (best real GPU that is not device 0), >=0 = enumeration index (the same
 * physical device is allowed with a warning — pre-hardware test mode). Its group
 * may be in flight simultaneously with device 0's. Tensors remember their device
 * (coli_vk_tensor_dev); free/bytes work on either. */
int  coli_vk_init_dev2(const char *spv_path, int devidx);
int  coli_vk_dev2_available(void);
int  coli_vk_tensor_dev(const ColiVkTensor *t);
int  coli_vk_mem_budget2(double *used_gb, double *budget_gb);
int  coli_vk_tensor_ensure2(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);
int  coli_vk_expert_group_issue2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                 ColiVkTensor *const *downs, const int *rows, int count,
                                 const float *x);
int  coli_vk_expert_group_take2(float *y);
int  coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           float *y, const float *x);

/* MLA absorb attention core (decode). The KV latent/rope caches live in persistent
 * per-layer device buffers: _ensure allocates a layer's cache at max_rows (once; resize
 * via _reset), _row mirrors one host row (absolute position), _reset drops all layers.
 * The caller keeps a valid-watermark and re-mirrors rows after any invalidation.
 * absorb runs S causal query rows over cache rows [st0, T) in one submit:
 * q [S,H*(Q+R)] roped, kv_b [H*(Q+V), K] uploads once (fmt 1=int8/2=int4),
 * ctx out [S,H*V]. Returns 0 -> caller falls back to CPU. */
int  coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd);
int  coli_vk_kv_row(int layer, int pos, const float *L, const float *R);
void coli_vk_kv_reset(void);
int  coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              float *ctx, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale);
/* Two resident matmuls sharing one input x in ONE submit (q_a + kv_a prologue pair).
 * Returns 0 -> caller falls back to single-matmul calls. */
/* q-prep chain: [q_a+kv_a pair] -> rmsnorm(q latent) -> q_b in ONE submit (needs
 * rmsnorm.spv next to the main shader; returns 0 without it -> 3-submit path).
 * lnw = the q-latent RMS-norm weights [Oqa], resident per layer after first call. */
int  coli_vk_attn_qprep(int layer,
                        ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                        ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                        ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                        int fmt, int grp, const float *lnw, float eps,
                        const float *x, int S, int I, float *q_out, float *kv_out,
                        float *lat_out /* normed q latent [S,Oqa], NULLable — DSA indexer input */);
int  coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                         ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                         int fmt, const float *x, int S, int I, int grp);

/* Fused variant: absorb + resident o-projection ([Dout, H*V]) in one submit; ctx stays
 * on-device, only out [S,Dout] is read back. Falls back like absorb (returns 0). */
int  coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                              float *out, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale, int Dout);

void   coli_vk_tensor_free(ColiVkTensor *t);
size_t coli_vk_tensor_bytes(const ColiVkTensor *t);

/* 1 if the selected device is an integrated GPU (shares physical memory with
 * the host), 0 otherwise or when no device is selected. */
int coli_vk_device_integrated(void);

#ifdef __cplusplus
}
#endif

#endif
