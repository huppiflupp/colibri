/* qwen36_fake_vulkan.h -- fake Vulkan backend for the qwen36 tier tests.
 *
 * The Vulkan twin of qwen36_fake_cuda.h: defines every coli_vk_* symbol the
 * tier's Vulkan shim (qwen36_tier.c, -DCOLI_VULKAN) links against, with the
 * signatures from backend_vulkan.h, and RECORDS what it receives. A test that
 * includes this and then ../qwen36_tier.c runs the shim's control flow with no
 * device, no libvulkan and no shaders -- in `make check` and under the
 * sanitizers on every platform -- while the numerics keep their own gate in
 * test_qwen36_tier_vk (VK=1, real device).
 *
 * Settable knobs:
 *   fake_vk_available     - what coli_vk_init returns and coli_vk_available
 *                           reports afterwards (default 1).
 *   fake_vk_budget_known  - whether coli_vk_mem_budget succeeds (default 1);
 *                           0 reproduces a driver without VK_EXT_memory_budget.
 *   fake_vk_budget_gb /
 *   fake_vk_used_gb       - the figures coli_vk_mem_budget reports.
 *   fake_vk_issue_hook    - called by coli_vk_expert_group_issue with the row
 *                           count and the input pointer; its return value is
 *                           what issue returns. NULL (the default) returns 1.
 *   fake_vk_take_ok       - coli_vk_expert_group_take returns this (default 1);
 *                           when 1 it fills row j of y with the value j+1 over
 *                           fake_vk_take_D floats, which the test sets to the
 *                           tier's D, so qt_take's accumulation is checkable.
 * Recorded:
 *   fake_vk_inits, fake_vk_shutdowns, fake_vk_uploads, fake_vk_frees,
 *   fake_vk_last_fmt, fake_vk_last_gs, fake_vk_last_bytes,
 *   fake_vk_last_issue_count. */
#ifndef QWEN36_FAKE_VULKAN_H
#define QWEN36_FAKE_VULKAN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* MinGW has no setenv: the tier reads its configuration from the environment
 * and the test must be able to set it on Windows too. */
#if defined(_WIN32)
static int test_setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
#define setenv test_setenv
#endif
#include "../backend_vulkan.h"

struct ColiVkTensor { int fmt, I, O, gs; const void *w; };

static int fake_vk_available = 1;
static int fake_vk_trunk_upload_ok;  /* inject dense upload failure by default */
static int fake_vk_budget_known = 1;
static double fake_vk_budget_gb = 2.0, fake_vk_used_gb = 0.0;
static int (*fake_vk_issue_hook)(int count, const float *x) = NULL;
static int fake_vk_take_ok = 1;
static int fake_vk_take_D = 0;

static int fake_vk_inits, fake_vk_shutdowns, fake_vk_uploads, fake_vk_frees;
static int fake_vk_last_fmt = -1, fake_vk_last_gs = -1;
static size_t fake_vk_last_bytes;
static size_t fake_vk_live_tensors, fake_vk_live_bytes;
static int fake_vk_last_issue_count;

int coli_vk_init(const char *spv_path) { (void)spv_path; fake_vk_inits++; return fake_vk_available; }
void coli_vk_shutdown(void) { fake_vk_shutdowns++; }
int coli_vk_available(void) { return fake_vk_available; }
const char *coli_vk_default_spv(char *buf, size_t n) { snprintf(buf, n, "fake.spv"); return buf; }
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count) {
    if (used_bytes) *used_bytes = fake_vk_live_bytes;
    if (tensor_count) *tensor_count = fake_vk_live_tensors;
}
int coli_vk_mem_budget(double *used_gb, double *budget_gb) {
    if (!fake_vk_budget_known) return 0;
    if (used_gb) *used_gb = fake_vk_used_gb;
    if (budget_gb) *budget_gb = fake_vk_budget_gb;
    return 1;
}
int coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales,
                          int fmt, int I, int O, int grp) {
    (void)scales;
    if (fmt == 1 && !fake_vk_trunk_upload_ok) return 0;
    if (*tensor) return 1;                       /* ensure: a second call is a no-op */
    ColiVkTensor *t = (ColiVkTensor *)calloc(1, sizeof *t);
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->gs = grp; t->w = weights;
    *tensor = t;
    fake_vk_uploads++;
    fake_vk_last_fmt = fmt; fake_vk_last_gs = grp;
    fake_vk_last_bytes = (size_t)I * O / (fmt == 1 ? 1 : 2);
    fake_vk_live_tensors++; fake_vk_live_bytes += fake_vk_last_bytes;
    return 1;
}
int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O, int gs) {
    (void)tensor; (void)y; (void)x; (void)weights; (void)scales;
    (void)fmt; (void)S; (void)I; (void)O; (void)gs;
    return 0;
}
void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    fake_vk_frees++;
    fake_vk_live_tensors--;
    fake_vk_live_bytes -= (size_t)t->I * t->O / (t->fmt == 1 ? 1 : 2);
    free(t);
}
int coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                               ColiVkTensor *const *downs, const int *rows, int count,
                               const float *x) {
    (void)gates; (void)ups; (void)downs; (void)rows;
    fake_vk_last_issue_count = count;
    if (fake_vk_issue_hook) return fake_vk_issue_hook(count, x);
    return 1;
}
int coli_vk_expert_group_take(float *y) {
    if (!fake_vk_take_ok) return 0;
    for (int j = 0; j < fake_vk_last_issue_count; j++)
        for (int d = 0; d < fake_vk_take_D; d++) y[(size_t)j * fake_vk_take_D + d] = (float)(j + 1);
    return 1;
}

/* Distinct per-expert output, scaled by token input: exposes pair permutation
 * and weight mistakes in the tier without reproducing the real shaders. */
static int fake_vk_prefill_calls;
int coli_vk_expert_prefill(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           const int *order, const float *weights, int S, int K,
                           const float *x, float *y) {
    (void)ups; (void)downs;
    fake_vk_prefill_calls++;
    if (!fake_vk_take_ok) return 0;
    int D = gates[0]->I;
    memset(y, 0, (size_t)S*D*sizeof(float));
    for (int c = 0, r = 0; c < count; c++)
        for (int j = 0; j < rows[c]; j++, r++) {
            int p = order[r], token = p/K;
            for (int d = 0; d < D; d++) y[(size_t)token*D+d] += weights[p]*(c+1)*x[(size_t)token*D+d];
        }
    return 1;
}


/* Stubs for the backend entry points the tier gained with the block / decode-graph work:
 * all report "unavailable", so the fake exercises the tier's CPU fallbacks. */
void coli_vk_arena_cpu_wrote(const void *p) { (void)p; }
int coli_vk_attn_block(ColiVkTensor *tq, ColiVkTensor *tk, ColiVkTensor *tv, ColiVkTensor *to, const float *x, const float *qn, const float *kn, float *Kc, float *Vc, int ldt, int S, int D, int H, int KV, int hd, int qdim, int rotary, int pos_base, float eps, float theta, float scale, float *out, int layer) { (void)tq; (void)tk; (void)tv; (void)to; (void)x; (void)qn; (void)kn; (void)Kc; (void)Vc; (void)ldt; (void)S; (void)D; (void)H; (void)KV; (void)hd; (void)qdim; (void)rotary; (void)pos_base; (void)eps; (void)theta; (void)scale; (void)out; (void)layer; return 0; }
int coli_vk_attn_dec_ready(void) { return 0; }
int coli_vk_attn_prefill(float *ctx, const float *q, const float *K, const float *V, int ldt, int S, int H, int KV, int hd, int pos_base, float scale) { (void)ctx; (void)q; (void)K; (void)V; (void)ldt; (void)S; (void)H; (void)KV; (void)hd; (void)pos_base; (void)scale; return 0; }
void coli_vk_block_post(float *x, float *n, const float *w, float eps, ColiVkTensor *router, float *logits, int E) { (void)x; (void)n; (void)w; (void)eps; (void)router; (void)logits; (void)E; }
int coli_vk_block_post_done(void) { return 0; }
int coli_vk_dec_record(const ColiDecLayer *d) { (void)d; return 0; }
int coli_vk_dec_run(int n, const int *layers, int S, const float *x, const float *nrm, const float *tmp, const float *logits, int H, int E) { (void)n; (void)layers; (void)S; (void)x; (void)nrm; (void)tmp; (void)logits; (void)H; (void)E; return 0; }
void coli_vk_defer_next_block(int on) { (void)on; }
int coli_vk_deferred(void) { return 0; }
int coli_vk_dn_ba_ready(void) { return 0; }
int coli_vk_dn_block(ColiVkTensor *proj, ColiVkTensor *outp, const float *x, const float *ba, const float *wb, const float *wa, const float *convw, const float *par, const float *normw, float *ring, float *state, int S, int H, int conv_dim, int convk, int vh, int vk, int kdim, int vdim, float eps, float qscale, float *y, int layer, int cap) { (void)proj; (void)outp; (void)x; (void)ba; (void)wb; (void)wa; (void)convw; (void)par; (void)normw; (void)ring; (void)state; (void)S; (void)H; (void)conv_dim; (void)convk; (void)vh; (void)vk; (void)kdim; (void)vdim; (void)eps; (void)qscale; (void)y; (void)layer; (void)cap; return 0; }
void coli_vk_dn_drop(int layer) { (void)layer; }
int coli_vk_dn_recur(float *outv, float *state, const float *qn, const float *kn, const float *vsrc, int vstride, int voff, const float *beta, const float *gexp, int S, int vh, int vk, int kdim, int vdim) { (void)outv; (void)state; (void)qn; (void)kn; (void)vsrc; (void)vstride; (void)voff; (void)beta; (void)gexp; (void)S; (void)vh; (void)vk; (void)kdim; (void)vdim; return 0; }
int coli_vk_dn_resident(int on) { (void)on; return 0; }
int coli_vk_dn_rollback(int layer) { (void)layer; return 0; }
float * coli_vk_dn_stage(int which, size_t bytes) { (void)which; (void)bytes; return NULL; }
int coli_vk_dn_sync(int layer) { (void)layer; return 0; }
void coli_vk_expert_post(float *x, float *n, const float *w, float eps) { (void)x; (void)n; (void)w; (void)eps; }
int coli_vk_expert_post_done(void) { return 0; }
int coli_vk_fast_gemv(void) { return 0; }
int coli_vk_flush_deferred(void) { return 1; }
float * coli_vk_host_arena(int slot, size_t bytes) { (void)slot; (void)bytes; return NULL; }
void coli_vk_kv_cut(int layer, int from) { (void)layer; (void)from; }
int coli_vk_moe_master(int layer, int E, ColiVkTensor *const *tg, ColiVkTensor *const *tu, ColiVkTensor *const *td) { (void)layer; (void)E; (void)tg; (void)tu; (void)td; return 0; }
int coli_vk_moe_route(int layer, const float *logits, const float *nrm, const float *wsg, int E, int K, int D) { (void)layer; (void)logits; (void)nrm; (void)wsg; (void)E; (void)K; (void)D; return 0; }

#endif /* QWEN36_FAKE_VULKAN_H */
