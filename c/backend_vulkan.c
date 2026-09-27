// Vulkan compute backend for colibri's quantized matmul, targeting the
// Strix Halo iGPU (RADV gfx1151). Mirrors backend_cuda.c's contract but
// exploits unified memory: weight "uploads" write into HOST_VISIBLE|
// DEVICE_LOCAL memory — the same physical RAM the iGPU reads — so there is
// no PCIe copy. That is what makes offloading *streamed experts* profitable
// here, which the discrete-CUDA path deliberately avoids.
//
// M2 scope: correctness + a standalone GPU-vs-CPU test harness. Synchronous
// submit/wait per call; async queues and zero-copy import come in M4.
#include "backend_vulkan.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <limits.h>
#ifdef __linux__
#include <unistd.h>
#endif

/* Thread safety (qwen36 tier): an upload thread may call coli_vk_tensor_ensure while the
 * decode thread submits expert groups. vkQueueSubmit on one VkQueue from two threads is a
 * spec violation, and the weight arena is a linked list. Every production submit on
 * G.queue goes through vk_submit; arena_suballoc takes g_arena_mx. Fence waits are never
 * taken under either lock. The VK_TEST harness is single-threaded and submits directly. */
static pthread_mutex_t g_submit_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_arena_mx  = PTHREAD_MUTEX_INITIALIZER;
static VkResult vk_submit(VkQueue q, const VkSubmitInfo *si, VkFence f) {
    pthread_mutex_lock(&g_submit_mx);
    VkResult r = vkQueueSubmit(q, 1, si, f);
    pthread_mutex_unlock(&g_submit_mx);
    return r;
}
static double vk_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

/* vk_fence_wait's spin budget (microseconds), set once by coli_vk_init from
 * COLI_VK_SPIN_US (default 300) so concurrent callers never race a lazy init. */
static long g_vk_spin_us = 300;

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s failed: %d\n", what, _r); return 0; } } while (0)

struct ColiVkTensor {
    VkBuffer wbuf, sbuf;
    VkDeviceMemory wmem, smem;
    size_t wbytes;
    int fmt, I, O, rowWords, gs;
    int dev;               /* 0 = primary device, 1 = the COLI_VK_DEV2 expert-tier device */
};

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

/* Persistent device-side KV latent/rope cache for one layer (MLA attention).
 * Host appends rows as tokens decode (absolute-position indexing); the absorb
 * kernel reads them in place. Allocated once at max_t rows, like the CUDA
 * kv_dev shadow. */
#define VK_KV_LAYERS 160
typedef struct {
    VkBuffer bl, br; VkDeviceMemory ml, mr; void *pl, *pr;
    int rows, K, R;
} VkKvLayer;

#define EG_MAX_EXPERTS 1024

static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;            // HOST_VISIBLE|HOST_COHERENT (prefer DEVICE_LOCAL) — for inputs/weights
    uint32_t memtype_cached;     // HOST_CACHED — for buffers the CPU reads back (outputs)
    /* Staged weight path: on a discrete card without Resizable BAR the HOST_VISIBLE|
     * DEVICE_LOCAL type is a ~256 MB window and everything past it silently lands in
     * system RAM. When staged==1, weight arenas use memtype_dl (DEVICE_LOCAL, not
     * host-visible) and are filled by vkCmdCopyBuffer from `stage` (memtype_stage,
     * host-visible, preferably NOT device-local so it never eats the BAR) on the
     * dedicated up_cmd/up_fence. Scratches, KV mirror and readbacks are unchanged. */
    int staged; uint32_t memtype_dl, memtype_stage;
    VkCommandPool up_cpool; VkCommandBuffer up_cmd; VkFence up_fence;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    /* fused dual gate+up+silu pipeline (6 bindings): x, Wg, gscale, Wu, uscale, hidden */
    VkShaderModule shader_gu; VkDescriptorSetLayout dsl_gu; VkPipelineLayout plyt_gu;
    VkPipeline pipe_gu; VkDescriptorPool dpool_gu; VkDescriptorSet dset_gu;
    /* Multi-row variants (qmatmul_mr / qmatmul_gate_up_mr): one workgroup per MR token
     * rows, each weight word read once for all of them. Same layouts as pipe / pipe_gu,
     * so the existing descriptor sets bind to either. Optional: absent -> one-row path. */
    VkShaderModule shader_mr, shader_gu_mr; VkPipeline pipe_mr, pipe_gu_mr; int mr;
    /* Tiled cooperative-matrix variants (qmatmul_coop / qmatmul_gate_up_coop), same
     * layouts again. coop = usable; coop_min = fewest token rows that take them. */
    VkShaderModule shader_co, shader_gu_co; VkPipeline pipe_co, pipe_gu_co; int coop, coop_min, coop_tt;
    /* Second-generation tiles (qmatmul_coop2 / qmatmul_gate_up_coop2): 128 outputs x
     * 16*TT tokens, 256 invocations. [0] = small token blocks (experts), [1] = large
     * (trunk). coop2 = usable; coop2_tt[] the TT of each; coop2_split = fewest rows
     * that take [1]. */
    VkShaderModule shader_co2, shader_gu_co2; VkPipeline pipe_co2[2], pipe_gu_co2[2];
    int coop2, coop2_tt[2], coop2_split;
    int has_coop_dev;    /* device created with cooperativeMatrix + shaderFloat16 + vulkanMemoryModel */
    /* MLA absorb attention core (7 bindings): q, W, scales, Lcache, Rcache, scores, ctx */
    /* causal prefill attention core (attn_prefill.comp): q, K, V, ctx */
    VkShaderModule shader_ap; VkDescriptorSetLayout dsl_ap; VkPipelineLayout plyt_ap;
    VkPipeline pipe_ap; VkDescriptorPool dpool_ap; VkDescriptorSet dset_ap;
    /* matrix-unit variant of the same core (attn_flash.comp, same layout/set),
     * built lazily per (head dim, query heads per kv head) */
    VkShaderModule shader_af; VkPipeline pipe_af; int af_hd, af_gq;
    Scratch ap_q, ap_k, ap_v, ap_o;
    /* DeltaNet prefill recurrence (dn_recur.comp): qn, kn, v source, beta, exp(g), state, out */
    VkShaderModule shader_dr; VkDescriptorSetLayout dsl_dr; VkPipelineLayout plyt_dr;
    VkPipeline pipe_dr; VkDescriptorPool dpool_dr; VkDescriptorSet dset_dr;
    Scratch dr_q, dr_k, dr_v, dr_b, dr_g, dr_s, dr_o;
    /* whole DeltaNet block of a prefill (coli_vk_dn_block): conv, prep, gated norm
     * pipelines, two matmul descriptor sets (projection, out_proj) and scratch */
    VkShaderModule shader_dc, shader_dp, shader_dg;
    VkDescriptorSetLayout dsl_dc, dsl_dp, dsl_dg; VkPipelineLayout plyt_dc, plyt_dp, plyt_dg;
    VkPipeline pipe_dc, pipe_dp, pipe_dg; VkDescriptorPool dpool_dc, dpool_dp, dpool_dg, db_pool;
    VkDescriptorSet dset_dc, dset_dp, dset_dg, db_mm[2];
    Scratch db_x, db_qz, db_ba, db_par, db_cw, db_ring, db_conv, db_bg, db_nw, db_or, db_y;
    int sgsize;
    int has_bda;         /* bufferDeviceAddress on: buffers usable by address (grouped expert shader) */
    /* grouped expert shaders (qmatmul_grp / qmatmul_gate_up_grp): [0] down, [1] gate_up */
    VkShaderModule shader_grp[2]; VkDescriptorSetLayout dsl_grp[2]; VkPipelineLayout plyt_grp[2];
    VkPipeline pipe_grp[2]; VkDescriptorPool dpool_grp[2]; VkDescriptorSet dset_grp[2]; int grp_tt;
    VkPipeline pipe_grp_s[2];   /* TT = 2 twin for the last (<= 32-row) tile of an expert */
    int grp_n[2], grp_ns[2], grp_mix[2];   /* full / small items per phase of the recorded group */
    Scratch grp_it[2], grp_et[2];
    /* COLI_VK_TS=1: GPU timestamps around every command buffer of the prefill
     * paths; per label the GPU time and the host wall time submit -> fence done */
    int ts_on; VkQueryPool tsq; double ts_period; double eg_th0;
    /* whole attention layer of a prefill (coli_vk_attn_block) */
    VkShaderModule shader_ab, shader_ag; VkDescriptorSetLayout dsl_ab, dsl_ag; VkPipelineLayout plyt_ab, plyt_ag;
    VkPipeline pipe_ab, pipe_ag; VkDescriptorPool dpool_ab, dpool_ag, ab_pool; VkDescriptorSet dset_ab, dset_ag, ab_mm[4];
    Scratch ab_x, ab_qkv, ab_nw, ab_qb, ab_gb, ab_k, ab_v, ab_cx, ab_ag, ab_y;
    VkShaderModule shader_att; VkDescriptorSetLayout dsl_att; VkPipelineLayout plyt_att;
    VkPipeline pipe_att; VkDescriptorPool dpool_att; VkDescriptorSet dset_att;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;
    Scratch x, y, h;   /* h = fused gate+up hidden output */
    Scratch stage;     /* host staging buffer for the staged weight path (TRANSFER_SRC) */
    /* full expert-group scratch: activations/hidden/output for K experts + per-expert
     * descriptor sets (gate_up: dsl_gu, down: dsl), so gate_up->down runs on-device in
     * one submit with hidden never leaving the GPU. */
    Scratch eg_x, eg_h, eg_y;
    VkDescriptorPool eg_pool; VkDescriptorSet eg_gu[EG_MAX_EXPERTS], eg_dn[EG_MAX_EXPERTS]; int eg_nsets;
    /* expert-group ASYNC state: its own command buffer + fence so an in-flight group
     * never collides with the main cmd/fence (dense matmuls, absorb) — issue() returns
     * immediately, the CPU computes its share, take() joins. */
    VkCommandBuffer eg_cmd; VkFence eg_fence; int eg_inflight; size_t eg_pending_yb;
    double eg_t0, eg_t1, eg_t2, eg_t3, eg_tin; int eg_prof;
    /* Prefill gather/reduce shares the expert command buffer, with persistent
     * metadata and token-sized host I/O. Packed x/h/y never leave the GPU. */
    Scratch ep_x, ep_order, ep_inverse, ep_weights, ep_y;
    VkShaderModule ep_shader[2]; VkDescriptorSetLayout ep_dsl[2];
    VkPipelineLayout ep_layout[2]; VkPipeline ep_pipe[2];
    VkDescriptorPool ep_pool[2]; VkDescriptorSet ep_set[2];
    int eg_reduced;
    VkBuffer ep_xa, ep_ya;      /* host-arena input/output of the current prefill group (or 0) */
    /* q-prep chain (pair -> rmsnorm -> q_b in ONE submit): norm pipeline (3 bindings),
     * a 3rd matmul set + norm set, GPU-only latent intermediates, per-layer resident
     * norm-weight buffers (tiny, uploaded once like the KV mirror). */
    VkShaderModule shader_nrm; VkDescriptorSetLayout dsl_nrm; VkPipelineLayout plyt_nrm;
    VkPipeline pipe_nrm; VkDescriptorPool dpool_nrm, qprep_pool; VkDescriptorSet dset_qp3, dset_nrm;
    Scratch qp1, qp2;
    VkBuffer lnbuf[VK_KV_LAYERS]; VkDeviceMemory lnmem[VK_KV_LAYERS]; int lnlen[VK_KV_LAYERS];
    Scratch att_sc;              /* attention score scratch (GPU-only) */
    Scratch att_ctx;             /* fused absorb+o: ctx stays on device (GPU-only) */
    Scratch y2;                  /* second output of the fused matmul pair (readback) */
    VkDescriptorPool pair_pool; VkDescriptorSet dset_pair;   /* 4-binding set for the pair's 2nd matmul */
    VkKvLayer kv[VK_KV_LAYERS];  /* per-layer resident KV latent/rope cache */
    /* resubmit cache: skip vkUpdateDescriptorSets + command re-record when the bound
     * tensor / shape / scratch buffers are unchanged from the previous call (the hot-
     * expert-called-repeatedly pattern). The synchronous fence wait each call means no
     * submission is ever in flight, so rebinding/re-recording only when something
     * actually changed is safe. */
    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, cmd_ready;
    VkBuffer bound_xbuf, bound_ybuf;
    size_t used_bytes, tensor_count;
    /* VRAM pressure-proofing: with VK_EXT_memory_priority the attention working set
     * (KV mirror, scratches) outranks bulk expert weights, so an oversubscribed heap
     * evicts cold tier experts instead of thrashing the per-token attention submits
     * (measured: decode attention 7.8s at 7.6 GB resident -> 17.8s at 15.2 GB).
     * VK_EXT_memory_budget lets the tier fill stop at a reserve instead of guessing. */
    int has_prio, has_budget, has_portability;
    float prio;                  /* priority applied to the NEXT allocations (class knob) */
} G;

struct PC { int fmt, S, I, O, rowWords, gs; };
/* gate_up shader only: extra SwiGLU clamp. limit<=0 keeps the unclamped path
 * (GLM-5.2 / colibri.c). Must ride per dispatch: the same pipeline serves
 * callers whose correct value is "no clamp". */
struct PCGU { int fmt, S, I, O, rowWords, gs; float limit; int pad; };
static float g_swiglu_limit = 0.f;
void coli_vk_set_swiglu_limit(float limit) { g_swiglu_limit = limit > 0.f ? limit : 0.f; }
static struct PCGU pcgu(int fmt, int S, int I, int O, int rowWords, int gs) {
    return (struct PCGU){fmt, S, I, O, rowWords, gs, g_swiglu_limit, 0};
}
struct PCEP { int S, D, K, rows; };
typedef struct { int S, K; const int *order; const float *weights; float *y; } ExpertPrefill;
struct PCN { int S, D; float eps; };
/* Push constants of the absorb attention kernel (must match attention_absorb.comp). */
struct PCAttn { int fmt, S, H, Q, R, V, K, st0, T, rowWords, cap; float scale; int gs; };
struct PCAP { int S, H, KV, hd, pos_base, ldt; float scale; };   /* attn_prefill.comp */
struct PCDR { int S, vh, vk, kdim, vdim, vstride, voff; };      /* dn_recur.comp */
struct PCDC { int S, conv_dim, convk, pstride; };               /* dn_conv.comp */
struct PCDP { int S, conv_dim, vk, vh, kdim; float scale; };    /* dn_prep.comp */
struct PCDG { int S, vh, vdim, pstride, zoff; float eps; };     /* dn_gnorm.comp */
struct PCAB { int S, H, KV, hd, qdim, gate_dim, rotary, pos_base, nt, has_qn, has_kn, q_off, k_off, v_off; float eps, theta; };  /* attn_prep.comp */
struct PCAG { int n, has_gate; };                               /* attn_gate.comp */
struct PCGRP { int I, O; float limit; int kgat; int ibase; };                        /* qmatmul_grp.comp */

static int pick_memtype(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    int best = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i; // ideal on APU
            if (best < 0) best = (int)i;
        }
    }
    return best;
}

/* Cached+coherent host-visible type for buffers the CPU READS BACK. pick_memtype prefers
 * DEVICE_LOCAL host-visible = write-combined VRAM over ReBAR, which the CPU writes fast but
 * reads catastrophically slowly (~40 MB/s). Outputs must be HOST_CACHED for cheap readback. */
static int pick_memtype_cached(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) return (int)i;
    }
    return pick_memtype(phys);   /* no cached type -> fall back (no worse than before) */
}

/* DEVICE_LOCAL and NOT host-visible, on the largest device-local heap: the target of the
 * staged weight path. -1 when no such type exists (unified-memory APUs, Lavapipe). */
static int pick_memtype_dl(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    int best = -1; VkDeviceSize bestheap = 0;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if (!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        VkDeviceSize hs = m.memoryHeaps[m.memoryTypes[i].heapIndex].size;
        if (hs > bestheap) { bestheap = hs; best = (int)i; }
    }
    return best;
}
/* Host-visible+coherent type for the staging buffer, preferring one that is NOT
 * device-local so staging never consumes the BAR window; falls back to G.memtype. */
static int pick_memtype_stage(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            !(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return (int)i;
    }
    return (int)G.memtype;
}

static int alloc_buf_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype, VkBufferUsageFlags usage) {
    if (G.has_bda) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
#ifdef VK_EXT_memory_priority
    VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .priority = G.prio};
    if (G.has_prio) ai.pNext = &pri;
#endif
    VkMemoryAllocateFlagsInfo maf = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    if (G.has_bda) { maf.pNext = (void *)ai.pNext; ai.pNext = &maf; }
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
    return 1;
}
static uint64_t vk_addr(VkBuffer b) {
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    return (uint64_t)vkGetBufferDeviceAddress(G.dev, &ai);
}
static int alloc_hostvis_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype) {
    return alloc_buf_mt(bytes, buf, mem, ptr, memtype, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}
/* Priority class of subsequent allocations (VK_EXT_memory_priority; no-op without it).
 * Scratches/KV force 1.0 internally; weight uploads take whatever is current — the
 * engine sets 0.4 around the bulk expert-tier fill, dense stays at the 0.75 default. */
void coli_vk_alloc_priority(float p) { G.prio = p < 0 ? 0 : p > 1 ? 1 : p; }

/* Device-local heap usage/budget in GB (VK_EXT_memory_budget). Returns 0 when the
 * extension is absent — callers then keep their count-based caps unchanged. */
int coli_vk_mem_budget(double *used_gb, double *budget_gb) {
#ifdef VK_EXT_memory_budget
    if (!G.has_budget || !G.phys) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
    vkGetPhysicalDeviceMemoryProperties2(G.phys, &mp2);
    double u = 0, b = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++)
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += (double)bud.heapUsage[i]; b += (double)bud.heapBudget[i];
        }
    if (used_gb) *used_gb = u / 1e9;
    if (budget_gb) *budget_gb = b / 1e9;
    return b > 0;
#else
    (void)used_gb; (void)budget_gb; return 0;
#endif
}
static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    return alloc_hostvis_mt(bytes, buf, mem, ptr, G.memtype);
}

static int scratch_reserve_mt(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G.dev, s->buf, NULL); vkFreeMemory(G.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    float p0 = G.prio; G.prio = 1.0f;            /* scratches ride every submit: never evict */
    int ok = alloc_hostvis_mt(bytes, &s->buf, &s->mem, &s->ptr, memtype);
    G.prio = p0;
    if (!ok) return 0;
    s->cap = bytes;
    return 1;
}
static int scratch_reserve(Scratch *s, size_t bytes) { return scratch_reserve_mt(s, bytes, G.memtype); }

static int rowwords(int fmt, int I) {
    size_t rb = fmt == 1 ? (size_t)I                         // bytes/row on CPU side
              : fmt == 5 ? ((size_t)I + 63) / 64 * 24        // int3-g64: 24B per 64-group
              : (size_t)(I + 1) / 2;
    return (int)((rb + 3) / 4);                              // padded to uint32 (24|4: exact)
}
/* Scale floats per tensor: per-row formats carry O, int3-g64 carries O*ceil(I/64)
 * (one f32 per 64-input group). upload_tensor and tensor_free must agree on this. */
static size_t scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (fmt == 4 || fmt == 7)
        return (size_t)O * (((size_t)I + gs - 1) / gs);   // per-group [O,ng]
    return (size_t)O;
}

static VkShaderModule load_spv(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VK] cannot open %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {   // SPIR-V is a stream of uint32; empty/non-seekable/odd size is invalid
        fprintf(stderr, "[VK] bad SPIR-V size %ld in %s\n", n, path); fclose(f); return VK_NULL_HANDLE; }
    uint32_t *code = malloc((size_t)n);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, n, f) != (size_t)n) { fclose(f); free(code); return VK_NULL_HANDLE; }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

/* Build a compute pipeline + descriptor pool/set for nbind storage buffers with a
 * pc_size-byte push constant. Used by the 4-binding matmul, 6-binding gate_up and
 * 7-binding absorb attention pipelines. */
static int build_pipeline(VkDevice dev, int nbind, size_t pc_size, VkShaderModule shader,
                          VkDescriptorSetLayout *dsl, VkPipelineLayout *plyt, VkPipeline *pipe,
                          VkDescriptorPool *dpool, VkDescriptorSet *dset) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)pc_size};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(dev, &pli, NULL, plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(dev, &dpi, NULL, dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(dev, &dsa, dset), "allocDescSet");
    return 1;
}

enum { TS_MATMUL, TS_GATEUP, TS_DNREC, TS_DNBLOCK, TS_ATTN, TS_EG, TS_N };
static const char *ts_name[TS_N] = {"matmul", "gate_up", "dn_recur", "dn_block", "attn_prefill", "expert_group"};
static struct { double gpu, host; long n; } g_ts[TS_N];
enum { TS_SLOT_CMD = 0, TS_SLOT_EG = 1 };
static void ts_begin(VkCommandBuffer cb, int slot) {
    if (!G.ts_on) return;
    vkCmdResetQueryPool(cb, G.tsq, 2u * slot, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, G.tsq, 2u * slot);
}
static void ts_end(VkCommandBuffer cb, int slot) {
    if (!G.ts_on) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.tsq, 2u * slot + 1);
}
static double vk_now(void);
static void ts_read(int slot, int label, double host_ms) {
    if (!G.ts_on) return;
    uint64_t v[2] = {0, 0};
    if (vkGetQueryPoolResults(G.dev, G.tsq, 2u * slot, 2, sizeof v, v, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    g_ts[label].gpu += (double)(v[1] - v[0]) * G.ts_period * 1e-6;
    g_ts[label].host += host_ms; g_ts[label].n++;
}
/* stage stamps inside one block (queries 4..19): ts_stage(cb, i) after stage i,
 * ts_stage_read(block, n) attributes the gaps to the stages of that block */
enum { TSB_DN, TSB_ATTN, TSB_EG, TSB_N };
static const char *tsb_name[TSB_N][8] = {{"dn:proj", "dn:conv", "dn:prep", "dn:recur", "dn:gnorm", "dn:outproj"},
                                          {"at:q", "at:k", "at:v", "at:prep", "at:core", "at:gate", "at:oproj"},
                                          {"eg:gather", "eg:gate_up", "eg:down", "eg:reduce"}};
static double g_tsb[TSB_N][8];
/* queries 4..11 for the G.cmd blocks (dn, attn), 12..19 for the expert group
 * (own command buffer, may be in flight at the same time) */
static uint32_t tsb_base(int b) { return b == TSB_EG ? 12u : 4u; }
static void ts_stage0b(VkCommandBuffer cb, int b) {
    if (!G.ts_on) return;
    vkCmdResetQueryPool(cb, G.tsq, tsb_base(b), 8);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, G.tsq, tsb_base(b));
}
static void ts_stageb(VkCommandBuffer cb, int b, int i) {
    if (G.ts_on) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, G.tsq, tsb_base(b) + 1u + i);
}
#define ts_stage0(cb) ts_stage0b(cb, TSB_DN)
#define ts_stage(cb, i) ts_stageb(cb, TSB_DN, i)
static void ts_stage_read(int b, int n) {
    if (!G.ts_on) return;
    uint64_t v[16] = {0};
    if (vkGetQueryPoolResults(G.dev, G.tsq, tsb_base(b), n + 1, sizeof v, v, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    for (int i = 0; i < n; i++) g_tsb[b][i] += (double)(v[i + 1] - v[i]) * G.ts_period * 1e-6;
}
static void ts_report(void) {
    if (!G.ts_on) return;
    for (int b = 0; b < TSB_N; b++) for (int i = 0; i < 8 && tsb_name[b][i]; i++)
        if (g_tsb[b][i] > 0) fprintf(stderr, "[VK_TS]   %-12s %10.1f ms\n", tsb_name[b][i], g_tsb[b][i]);
    double tg = 0, th = 0;
    fprintf(stderr, "[VK_TS] %-13s %7s %10s %10s %8s\n", "label", "calls", "gpu ms", "host ms", "gpu/host");
    for (int i = 0; i < TS_N; i++) if (g_ts[i].n) {
        fprintf(stderr, "[VK_TS] %-13s %7ld %10.1f %10.1f %7.0f%%\n", ts_name[i], g_ts[i].n, g_ts[i].gpu, g_ts[i].host,
                g_ts[i].host > 0 ? 100.0 * g_ts[i].gpu / g_ts[i].host : 0.0);
        tg += g_ts[i].gpu; th += g_ts[i].host;
    }
    fprintf(stderr, "[VK_TS] %-13s %7s %10.1f %10.1f %7.0f%%\n", "total", "", tg, th, th > 0 ? 100.0 * tg / th : 0.0);
}

/* Make compute-shader writes visible to the host before the fence the CPU waits
 * on: the fence orders execution, the barrier makes the writes available to a
 * HOST read (Khronos synchronization example "CPU read-back of data written by a
 * compute shader"). HOST_COHERENT spares the invalidate, not this dependency. */
/* Host arena: persistent host-visible, HOST_CACHED buffers the engine can keep its
 * prefill rows in (normed input, sublayer output). A block or group called with
 * exactly such a pointer binds the buffer itself instead of copying through its
 * own staging (UMA: the copies were ~1.5 ms per layer at 1011 tokens). Cached,
 * because the CPU reads these rows (residual add, norms) -- reads from the
 * write-combined staging type cost ~250 ns each. */
#define ARENA_N 4
static Scratch g_arena[ARENA_N];
float *coli_vk_host_arena(int slot, size_t bytes) {
    if (!G.ready || slot < 0 || slot >= ARENA_N || !scratch_reserve_mt(&g_arena[slot], bytes, G.memtype_cached)) return NULL;
    return (float *)g_arena[slot].ptr;
}
static VkBuffer arena_buf(const void *p, size_t n) {
    for (int i = 0; i < ARENA_N; i++)
        if (g_arena[i].buf && p == g_arena[i].ptr && n <= g_arena[i].cap) return g_arena[i].buf;
    return VK_NULL_HANDLE;
}
static void host_read_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier hb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
}
/* A second pipeline on an existing layout, with the multi-row count MR as
 * specialization constant 0 (qmatmul_mr.comp / qmatmul_gate_up_mr.comp). */
static int build_pipeline_mr(VkDevice dev, VkPipelineLayout plyt, VkShaderModule shader, int mr, VkPipeline *pipe) {
    VkSpecializationMapEntry me = {.constantID = 0, .offset = 0, .size = sizeof(int)};
    VkSpecializationInfo spi = {.mapEntryCount = 1, .pMapEntries = &me, .dataSize = sizeof(int), .pData = &mr};
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main",
                  .pSpecializationInfo = &spi},
        .layout = plyt};
    return vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe) == VK_SUCCESS;
}
/* Use the multi-row shader for this dispatch (S token rows over one I x O weight)?
 * Only formats it implements, and only where it measured faster on gfx1151 (RADV,
 * VK_MR_SWEEP in the harness): from 8 rows on, on weights of at least 4M elements
 * (2 MB int4 -- smaller ones stay cache-resident, so reading them once saves nothing
 * and the fewer workgroups cost occupancy), with I up to 8192 (at I = 16384 MR=4
 * measured slower). The harness sets g_mr_any to cover every shape. */
static int g_mr_any;
static int mr_use(int fmt, int S, int I, int O) {
    if (G.mr < 2 || S < 2 || !(fmt == 1 || fmt == 2 || fmt == 4)) return 0;
    if (g_mr_any) return 1;
    return S >= 8 && I <= 8192 && (size_t)I * (size_t)O >= ((size_t)4 << 20);
}
/* Pipeline and grid of a tiled dispatch (coop_use said yes): the second-generation
 * tiles when the output dimension fits their 128-row blocks, else the first. */
static VkPipeline coop_pick(int gate_up, int S, int O, uint32_t *gx, uint32_t *gy) {
    /* COLI_VK_GU64=1: gate_up shader with 64-output blocks (gate + up of the same 64
     * outputs share one 128-row tile; experiment) */
    static int gu64 = -1;
    if (gu64 < 0) { const char *e = getenv("COLI_VK_GU64"); gu64 = e && *e == '1'; }
    int bo = gate_up && gu64 ? 64 : 128;
    if (G.coop2 && G.sgsize == 64 && O % bo == 0) {   /* gen 2 hard-codes 4 x wave64 */
        int v = S >= G.coop2_split;
        int tm = 16 * G.coop2_tt[v];
        *gx = (uint32_t)((S + tm - 1) / tm); *gy = (uint32_t)(O / bo);
        return gate_up ? G.pipe_gu_co2[v] : G.pipe_co2[v];
    }
    int tm = 16 * G.coop_tt;
    *gx = (uint32_t)((S + tm - 1) / tm); *gy = (uint32_t)(O / 64);
    return gate_up ? G.pipe_gu_co : G.pipe_co;
}
/* Tiled cooperative-matrix shader for this dispatch? int8/int4 formats, both dims a
 * multiple of its 64-wide tiles, enough token rows. Takes precedence over the
 * multi-row shader. One workgroup = 64 outputs x 16*coop_tt token rows, so a
 * lone matmul launches only O/64 workgroups: measured on gfx1151 it loses to the
 * GEMV below ~64 rows, while inside an expert group (many experts in flight in
 * one submit) it wins from 8 rows on. `grouped` picks the threshold. */
static int g_coop_any;
static int coop_use(int fmt, int S, int I, int O, int gs, int grouped) {
    if (!G.coop || !(fmt == 1 || fmt == 2 || fmt == 4) || I % 64 || O % 64) return 0;
    if (fmt == 4 && (gs < 8 || gs % 8)) return 0;
    if (g_coop_any) return 1;
    return S >= (grouped ? G.coop_min : (G.coop_min > 64 ? G.coop_min : 64));
}

/* "…/qmatmul.spv" -> "…/qmatmul<suffix>" (sibling of the main shader). */
static void derive_sibling(const char *spv, const char *suffix, char *out, size_t n) {
    const char *dot = strstr(spv, ".spv");
    if (dot && (size_t)(dot - spv) + strlen(suffix) + 1 < n) {
        size_t pre = (size_t)(dot - spv);
        memcpy(out, spv, pre); strcpy(out + pre, suffix);
    } else snprintf(out, n, "%s", spv);
}
/* "…/qmatmul.spv" -> "…/attention_absorb.spv" (same directory). */
static void derive_dir_file(const char *spv, const char *fname, char *out, size_t n) {
    const char *sl = strrchr(spv, '/');
    size_t pre = sl ? (size_t)(sl - spv) + 1 : 0;
    if (pre + strlen(fname) + 1 < n) { memcpy(out, spv, pre); strcpy(out + pre, fname); }
    else snprintf(out, n, "%s", fname);
}

int coli_vk_init(const char *spv_path) {
    if (G.ready) return 1;
    {
        const char *e = getenv("COLI_VK_SPIN_US");
        g_vk_spin_us = e ? atol(e) : 300;
        if (g_vk_spin_us < 0) g_vk_spin_us = 0;
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &G.inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "[VK] no devices\n"); return 0; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    // Prefer a real GPU over a CPU/software device (llvmpipe) on multi-adapter hosts:
    // discrete > integrated > virtual > other/cpu. Falls back to devs[0] if all equal.
    G.phys = devs[0];
    int bestrank = -1;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER          ? 1 : 0; // CPU last
        if (rank > bestrank) { bestrank = rank; G.phys = devs[i]; }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, qf);
    G.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.qfam = i; break; }
    if (G.qfam == UINT32_MAX) { fprintf(stderr, "[VK] no compute queue\n"); return 0; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    /* Pressure-proofing extensions (both optional, detected at runtime):
     * memory_priority ranks allocations for the kernel's eviction order,
     * memory_budget exposes how much VRAM a new allocation can still take. */
    const char *dext[4]; uint32_t ndext = 0;
    {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) {
            vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, ep);
            for (uint32_t i = 0; i < ne; i++) {
#ifdef VK_EXT_memory_priority
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) G.has_prio = 1;
#endif
#ifdef VK_EXT_memory_budget
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) G.has_budget = 1;
#endif
                if (!strcmp(ep[i].extensionName, "VK_KHR_portability_subset")) G.has_portability = 1;
#ifdef VK_KHR_cooperative_matrix
                if (!strcmp(ep[i].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) G.has_coop_dev = 1;
#endif
            }
            free(ep);
        }
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
#ifdef VK_KHR_cooperative_matrix
    /* Optional tiled matmul on the matrix units: needs the extension, its feature,
     * f16 arithmetic and the Vulkan memory model, plus a 16x16x16 f16 x f16 -> f32
     * subgroup configuration. Missing any of them -> the GEMV shaders only. */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features v12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    if (G.has_coop_dev) {
        v12.pNext = &cmf;
        VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v12};
        vkGetPhysicalDeviceFeatures2(G.phys, &f2);
        int cfg = 0;
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR gp = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(G.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t nc = 0;
        if (gp && gp(G.phys, &nc, NULL) == VK_SUCCESS && nc) {
            VkCooperativeMatrixPropertiesKHR *cp = calloc(nc, sizeof(*cp));
            if (cp) {
                for (uint32_t i = 0; i < nc; i++) cp[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                if (gp(G.phys, &nc, cp) == VK_SUCCESS)
                    for (uint32_t i = 0; i < nc; i++)
                        if (cp[i].MSize == 16 && cp[i].NSize == 16 && cp[i].KSize == 16 &&
                            cp[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cp[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            cp[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cp[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            cp[i].scope == VK_SCOPE_SUBGROUP_KHR) cfg = 1;
                free(cp);
            }
        }
        G.has_coop_dev = cfg && cmf.cooperativeMatrix && v12.shaderFloat16 && v12.vulkanMemoryModel;
        int bda = v12.bufferDeviceAddress && !(getenv("COLI_VK_BDA") && getenv("COLI_VK_BDA")[0] == '0');
        if (G.has_coop_dev) {
            memset(&v12, 0, sizeof v12); v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            v12.shaderFloat16 = VK_TRUE; v12.vulkanMemoryModel = VK_TRUE;
            if (bda) { v12.bufferDeviceAddress = VK_TRUE; G.has_bda = 1; }
            memset(&cmf, 0, sizeof cmf); cmf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            cmf.cooperativeMatrix = VK_TRUE;
            v12.pNext = &cmf; cmf.pNext = (void *)di.pNext; di.pNext = &v12;
            dext[ndext++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        }
    }
#endif
#ifdef VK_EXT_memory_priority
    VkPhysicalDeviceMemoryPriorityFeaturesEXT prif = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        .memoryPriority = VK_TRUE};
    if (G.has_prio) { dext[ndext++] = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME; prif.pNext = (void *)di.pNext; di.pNext = &prif; }
#endif
#ifdef VK_EXT_memory_budget
    if (G.has_budget) dext[ndext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
#endif
    /* A portability implementation (MoltenVK) lists VK_KHR_portability_subset,
     * and the spec then requires it enabled (VUID-VkDeviceCreateInfo-pProperties-
     * 04451); conformant drivers never list it, so RADV, NVIDIA and Lavapipe are
     * untouched. Spelled out: the macro sits behind VK_ENABLE_BETA_EXTENSIONS. */
    if (G.has_portability) dext[ndext++] = "VK_KHR_portability_subset";
    di.enabledExtensionCount = ndext; di.ppEnabledExtensionNames = ndext ? dext : NULL;
    G.prio = 0.75f;                              /* default class: dense/resident weights */
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);
    if (G.has_prio || G.has_budget)
        fprintf(stderr, "[VK] VRAM pressure-proofing: memory_priority %s, memory_budget %s\n",
                G.has_prio ? "on" : "absent", G.has_budget ? "on" : "absent");

    int mt = pick_memtype(G.phys);
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;
    G.memtype_cached = (uint32_t)pick_memtype_cached(G.phys);

    /* Resizable-BAR sanity (#523): on discrete cards the weight tiers want
     * HOST_VISIBLE|DEVICE_LOCAL. With ReBAR disabled that combination exists only in a
     * ~256 MB BAR window (or not at all), so tier allocations silently land in system
     * RAM and every access crosses PCIe — measurably SLOWER than the CPU path, while
     * the resident-experts log still reports an apparently healthy VRAM tier. Compare
     * the chosen type's heap against the largest DEVICE_LOCAL heap and say so up front. */
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(G.phys, &mp);
        VkDeviceSize dl_max = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
            if ((mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                mp.memoryHeaps[i].size > dl_max) dl_max = mp.memoryHeaps[i].size;
        VkMemoryPropertyFlags cf = mp.memoryTypes[G.memtype].propertyFlags;
        VkDeviceSize hv_dl = mp.memoryHeaps[mp.memoryTypes[G.memtype].heapIndex].size;
        int small_bar = dl_max && (!(cf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || hv_dl * 4 < dl_max);
        /* Staged device-local weights: auto when the host-visible slice is small,
         * COLI_VK_STAGED=1/0 forces either way (1 lets ReBAR owners validate the path).
         * Exactly "1" forces on, exactly "0" forces off; anything else (unset, empty,
         * other text) is auto. */
        int dl = pick_memtype_dl(G.phys);
        const char *st = getenv("COLI_VK_STAGED");
        int staged_off_forced = st && !strcmp(st, "0");
        int want = st && !strcmp(st, "1") ? 1 : (staged_off_forced ? 0 : small_bar);
        if (want && dl < 0) {
            fprintf(stderr, "[VK] COLI_VK_STAGED requested but no device-local-only memory type "
                    "(unified memory?) — mapped uploads\n");
            want = 0;
        }
        if (want) {
            G.staged = 1; G.memtype_dl = (uint32_t)dl; G.memtype_stage = (uint32_t)pick_memtype_stage(G.phys);
            fprintf(stderr, "[VK] weights: staged device-local uploads (host-visible VRAM %llu of %llu MB; "
                    "memtype %u via staging memtype %u)\n", (unsigned long long)(hv_dl >> 20),
                    (unsigned long long)(dl_max >> 20), G.memtype_dl, G.memtype_stage);
        } else if (dl_max && !(cf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            fprintf(stderr, "[VK] warning: no host-visible+device-local memory type — weight tiers "
                    "will live in system RAM and every access crosses PCIe (expect slower than "
                    "CPU-only). On a discrete card, enable Resizable BAR in the BIOS or set "
                    "COLI_VK_STAGED=1.\n");
        else if (small_bar)
            fprintf(stderr, "[VK] warning: only %llu of %llu MB VRAM is host-visible (Resizable BAR "
                    "appears disabled) — allocations beyond the %llu MB window fall back to system "
                    "RAM and will be slow. Enable Resizable BAR / Smart Access Memory in the BIOS.%s\n",
                    (unsigned long long)(hv_dl >> 20), (unsigned long long)(dl_max >> 20),
                    (unsigned long long)(hv_dl >> 20),
                    staged_off_forced ? " (COLI_VK_STAGED=0 is set; unset it to use staged uploads)" : "");
    }

    G.shader = load_spv(G.dev, spv_path);
    if (!G.shader) return 0;
    if (!build_pipeline(G.dev, 4, sizeof(struct PC), G.shader, &G.dsl, &G.plyt, &G.pipe, &G.dpool, &G.dset)) return 0;

    /* Optional fused gate+up pipeline: skip gracefully if its shader isn't present
     * (single-matmul path keeps working). */
    char gu_path[512]; derive_sibling(spv_path, "_gate_up.spv", gu_path, sizeof(gu_path));
    G.shader_gu = load_spv(G.dev, gu_path);
    if (G.shader_gu && !build_pipeline(G.dev, 6, sizeof(struct PCGU), G.shader_gu, &G.dsl_gu, &G.plyt_gu, &G.pipe_gu, &G.dpool_gu, &G.dset_gu))
        return 0;

    /* Optional multi-row pipelines for S > 1. COLI_VK_MR=<rows per workgroup> (default
     * 4, 1 or 0 = off: every S > 1 dispatch takes the one-row shaders as before). Both
     * must load, or neither is used. */
    {
        const char *e = getenv("COLI_VK_MR");
        int mr = e ? atoi(e) : 4;
        if (mr > 8) mr = 8;
        G.mr = 0;
        VkPhysicalDeviceProperties dp; vkGetPhysicalDeviceProperties(G.phys, &dp);
        /* both shaders declare a fixed 8192-float staging array (32 KiB) */
        if (mr > 1 && G.shader_gu && dp.limits.maxComputeSharedMemorySize >= 8192 * sizeof(float)) {
            char p1[512], p2[512];
            derive_sibling(spv_path, "_mr.spv", p1, sizeof(p1));
            derive_sibling(spv_path, "_gate_up_mr.spv", p2, sizeof(p2));
            G.shader_mr = load_spv(G.dev, p1);
            G.shader_gu_mr = G.shader_mr ? load_spv(G.dev, p2) : VK_NULL_HANDLE;
            if (G.shader_mr && G.shader_gu_mr &&
                build_pipeline_mr(G.dev, G.plyt, G.shader_mr, mr, &G.pipe_mr) &&
                build_pipeline_mr(G.dev, G.plyt_gu, G.shader_gu_mr, mr, &G.pipe_gu_mr))
                G.mr = mr;
            /* COLI_VK_MR_ALL=1: multi-row for every S > 1, bypassing the size rule (tuning) */
            const char *a = getenv("COLI_VK_MR_ALL");
            if (a && *a == '1') g_mr_any = 1;
        }
    }

    /* Optional tiled cooperative-matrix pipelines. COLI_VK_COOP=0 turns them off,
     * COLI_VK_COOP_MIN=<rows> sets the fewest token rows per expert that take them in an
     * expert group (default 8; a lone matmul needs max(that, 64)). */
    {
        const char *e = getenv("COLI_VK_COOP"), *mn = getenv("COLI_VK_COOP_MIN"), *tt = getenv("COLI_VK_COOP_TT");
        G.coop = 0; G.coop_min = mn ? atoi(mn) : 8;
        if (G.coop_min < 2) G.coop_min = 2;
        /* 16-row token tiles per workgroup (1, 2 or 4; COLI_VK_COOP_TT) */
        G.coop_tt = tt ? atoi(tt) : 2;
        if (G.coop_tt != 1 && G.coop_tt != 4) G.coop_tt = 2;
        VkPhysicalDeviceProperties dp; vkGetPhysicalDeviceProperties(G.phys, &dp);
        if (G.has_coop_dev && !(e && *e == '0') && G.shader_gu &&
            dp.limits.maxComputeSharedMemorySize >= 49152) {   /* 64-row tiles: ~44 KiB shared */
            char p1[512], p2[512];
            derive_sibling(spv_path, "_coop.spv", p1, sizeof(p1));
            derive_sibling(spv_path, "_gate_up_coop.spv", p2, sizeof(p2));
            G.shader_co = load_spv(G.dev, p1);
            G.shader_gu_co = G.shader_co ? load_spv(G.dev, p2) : VK_NULL_HANDLE;
            if (G.shader_co && G.shader_gu_co &&
                build_pipeline_mr(G.dev, G.plyt, G.shader_co, G.coop_tt, &G.pipe_co) &&
                build_pipeline_mr(G.dev, G.plyt_gu, G.shader_gu_co, G.coop_tt, &G.pipe_gu_co))
                G.coop = 1;
        }
        /* second generation, on top: COLI_VK_COOP2=0 off, COLI_VK_COOP2_TT=<small>,<large>
         * (default 2,8), COLI_VK_COOP2_SPLIT=<rows> (default 64) */
        const char *e2 = getenv("COLI_VK_COOP2"), *t2 = getenv("COLI_VK_COOP2_TT"), *sp = getenv("COLI_VK_COOP2_SPLIT");
        G.coop2 = 0; G.coop2_tt[0] = 2; G.coop2_tt[1] = 8; G.coop2_split = sp ? atoi(sp) : 64;
        if (t2) sscanf(t2, "%d,%d", &G.coop2_tt[0], &G.coop2_tt[1]);
        /* gen 2 needs TT >= 2 (its x staging reads two token tiles per lane group;
         * TT=1 computed garbage in the harness) */
        for (int v = 0; v < 2; v++) if (G.coop2_tt[v] < 2 || G.coop2_tt[v] > 8) {
            fprintf(stderr, "[VK] COLI_VK_COOP2_TT: %d not supported (2..8), using %d\n", G.coop2_tt[v], v ? 8 : 2);
            G.coop2_tt[v] = v ? 8 : 2;
        }
        if (G.coop && !(e2 && *e2 == '0') && dp.limits.maxComputeSharedMemorySize >= 57344) {   /* TT=8 gate_up: 56 KiB */
            char p1[512], p2[512];
            derive_sibling(spv_path, "_coop2.spv", p1, sizeof(p1));
            derive_sibling(spv_path, "_gate_up_coop2.spv", p2, sizeof(p2));
            G.shader_co2 = load_spv(G.dev, p1);
            G.shader_gu_co2 = G.shader_co2 ? load_spv(G.dev, p2) : VK_NULL_HANDLE;
            int ok = G.shader_co2 && G.shader_gu_co2;
            for (int v = 0; ok && v < 2; v++)
                ok = build_pipeline_mr(G.dev, G.plyt, G.shader_co2, G.coop2_tt[v], &G.pipe_co2[v]) &&
                     build_pipeline_mr(G.dev, G.plyt_gu, G.shader_gu_co2, G.coop2_tt[v], &G.pipe_gu_co2[v]);
            G.coop2 = ok;
        }
        /* grouped expert shaders: need device addresses and gen 2 (wave64);
         * COLI_VK_GRP=0 off, COLI_VK_GRP_TT=<token tiles> (default 2) */
        const char *eg = getenv("COLI_VK_GRP"), *gt = getenv("COLI_VK_GRP_TT");
        G.grp_tt = gt ? atoi(gt) : 4; /* Allowed TT: 2, 4, 8. Powers of two only, because the double-buffered x fetch
         * splits 64 inputs into 4*TT per lane (XPL); other values compute garbage.
         * The upper bound is a hardware limit: TT 16 needs 68 KB of shared memory
         * for gate_up (64 KB per workgroup on RDNA3.5 / Radeon 8060S) and 64 f32
         * accumulator tiles per subgroup. On hardware with more shared memory and
         * registers per workgroup the limit can be raised (TT 16 is the ceiling of
         * the XPL scheme: 64 / XPL must stay >= 1). */
        if (G.grp_tt != 2 && G.grp_tt != 4 && G.grp_tt != 8) G.grp_tt = 4;
        if (G.coop2 && G.has_bda && !(eg && *eg == '0')) {
            const char *nm[2] = {"_grp.spv", "_gate_up_grp.spv"};
            int ok = 1;
            for (int v = 0; ok && v < 2; v++) {
                char pp[512]; derive_sibling(spv_path, nm[v], pp, sizeof(pp));
                G.shader_grp[v] = load_spv(G.dev, pp);
                VkPipeline tmp = VK_NULL_HANDLE;
                ok = G.shader_grp[v] &&
                     build_pipeline(G.dev, v ? 5 : 4, sizeof(struct PCGRP), G.shader_grp[v], &G.dsl_grp[v], &G.plyt_grp[v], &tmp, &G.dpool_grp[v], &G.dset_grp[v]) &&
                     build_pipeline_mr(G.dev, G.plyt_grp[v], G.shader_grp[v], G.grp_tt, &G.pipe_grp[v]);
                /* mixed tiles: full 16*TT-row tiles while more than 32 rows remain,
                 * the rest on a 32-row twin (COLI_VK_GRP_MIX=0: all full tiles) */
                const char *mx = getenv("COLI_VK_GRP_MIX");
                if (ok && G.grp_tt > 2 && !(mx && *mx == '0') &&
                    !build_pipeline_mr(G.dev, G.plyt_grp[v], G.shader_grp[v], 2, &G.pipe_grp_s[v])) G.pipe_grp_s[v] = VK_NULL_HANDLE;
                if (tmp) vkDestroyPipeline(G.dev, tmp, NULL);
            }
            if (!ok) G.pipe_grp[0] = G.pipe_grp[1] = VK_NULL_HANDLE;
            if (!G.pipe_grp_s[0] || !G.pipe_grp_s[1]) {
                for (int v = 0; v < 2; v++) if (G.pipe_grp_s[v]) vkDestroyPipeline(G.dev, G.pipe_grp_s[v], NULL);
                G.pipe_grp_s[0] = G.pipe_grp_s[1] = VK_NULL_HANDLE;
            }
        }
    }

    /* Optional MLA absorb attention pipeline (same directory as the main shader). */
    /* Optional rmsnorm pipeline: enables the pair->norm->q_b single-submit chain
     * (coli_vk_attn_qprep); absent -> callers keep the 3-submit path. */
    char nrm_path[512]; derive_dir_file(spv_path, "rmsnorm.spv", nrm_path, sizeof(nrm_path));
    G.shader_nrm = load_spv(G.dev, nrm_path);
    if (G.shader_nrm) {
        if (!build_pipeline(G.dev, 3, sizeof(struct PCN), G.shader_nrm, &G.dsl_nrm, &G.plyt_nrm, &G.pipe_nrm, &G.dpool_nrm, &G.dset_nrm))
            return 0;
        /* one extra 4-binding matmul set for the chain's 3rd matmul (dset+dset_pair serve 1+2) */
        VkDescriptorPoolSize ps3 = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps3};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi3, NULL, &G.qprep_pool), "qprep descPool");
        VkDescriptorSetAllocateInfo dsa3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.qprep_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa3, &G.dset_qp3), "qprep descSet");
    }
    const char *ep_names[2] = {"expert_gather.spv", "expert_reduce.spv"};
    for (int i = 0; i < 2; i++) {
        char path[512]; derive_dir_file(spv_path, ep_names[i], path, sizeof(path));
        G.ep_shader[i] = load_spv(G.dev, path);
        if (G.ep_shader[i] && !build_pipeline(G.dev, i ? 4 : 3, sizeof(struct PCEP),
                G.ep_shader[i], &G.ep_dsl[i], &G.ep_layout[i], &G.ep_pipe[i],
                &G.ep_pool[i], &G.ep_set[i])) return 0;
    }
    char att_path[512]; derive_dir_file(spv_path, "attention_absorb.spv", att_path, sizeof(att_path));
    G.shader_att = load_spv(G.dev, att_path);
    if (G.shader_att && !build_pipeline(G.dev, 7, sizeof(struct PCAttn), G.shader_att, &G.dsl_att, &G.plyt_att, &G.pipe_att, &G.dpool_att, &G.dset_att))
        return 0;
    /* The prefill kernels below reduce over one 64-lane subgroup per workgroup. */
    {
        VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 pp2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
        vkGetPhysicalDeviceProperties2(G.phys, &pp2);
        G.sgsize = (int)sgp.subgroupSize;
    }
    /* Optional whole-DeltaNet-block kernels (COLI_VK_DN_BLOCK=0 turns them off). */
    if (G.sgsize == 64 && !(getenv("COLI_VK_DN_BLOCK") && getenv("COLI_VK_DN_BLOCK")[0] == '0')) {
        char pa[512];
        derive_dir_file(spv_path, "dn_conv.spv", pa, sizeof(pa));  G.shader_dc = load_spv(G.dev, pa);
        derive_dir_file(spv_path, "dn_prep.spv", pa, sizeof(pa));  G.shader_dp = load_spv(G.dev, pa);
        derive_dir_file(spv_path, "dn_gnorm.spv", pa, sizeof(pa)); G.shader_dg = load_spv(G.dev, pa);
        int ok = G.shader_dc && G.shader_dp && G.shader_dg &&
            build_pipeline(G.dev, 4, sizeof(struct PCDC), G.shader_dc, &G.dsl_dc, &G.plyt_dc, &G.pipe_dc, &G.dpool_dc, &G.dset_dc) &&
            build_pipeline(G.dev, 6, sizeof(struct PCDP), G.shader_dp, &G.dsl_dp, &G.plyt_dp, &G.pipe_dp, &G.dpool_dp, &G.dset_dp) &&
            build_pipeline(G.dev, 4, sizeof(struct PCDG), G.shader_dg, &G.dsl_dg, &G.plyt_dg, &G.pipe_dg, &G.dpool_dg, &G.dset_dg);
        if (ok) {
            VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8};
            VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &ps};
            VkDescriptorSetLayout l2[2] = {G.dsl, G.dsl};
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorSetCount = 2, .pSetLayouts = l2};
            ok = vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.db_pool) == VK_SUCCESS;
            ai.descriptorPool = G.db_pool;
            ok = ok && vkAllocateDescriptorSets(G.dev, &ai, G.db_mm) == VK_SUCCESS;
        }
        if (!ok) { G.pipe_dc = G.pipe_dp = G.pipe_dg = VK_NULL_HANDLE; }
    }
    if (G.sgsize != 64) fprintf(stderr, "[VK] subgroup size %d != 64: prefill attention/DeltaNet kernels off\n", G.sgsize);
    /* Optional whole-attention-layer kernels (COLI_VK_ATTN_BLOCK=0 turns them off). */
    if (G.sgsize == 64 && !(getenv("COLI_VK_ATTN_BLOCK") && getenv("COLI_VK_ATTN_BLOCK")[0] == '0')) {
        char pa[512];
        derive_dir_file(spv_path, "attn_prep.spv", pa, sizeof(pa)); G.shader_ab = load_spv(G.dev, pa);
        derive_dir_file(spv_path, "attn_gate.spv", pa, sizeof(pa)); G.shader_ag = load_spv(G.dev, pa);
        int ok = G.shader_ab && G.shader_ag &&
            build_pipeline(G.dev, 6, sizeof(struct PCAB), G.shader_ab, &G.dsl_ab, &G.plyt_ab, &G.pipe_ab, &G.dpool_ab, &G.dset_ab) &&
            build_pipeline(G.dev, 3, sizeof(struct PCAG), G.shader_ag, &G.dsl_ag, &G.plyt_ag, &G.pipe_ag, &G.dpool_ag, &G.dset_ag);
        if (ok) {
            VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 16};
            VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 4, .poolSizeCount = 1, .pPoolSizes = &ps};
            VkDescriptorSetLayout l4[4] = {G.dsl, G.dsl, G.dsl, G.dsl};
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorSetCount = 4, .pSetLayouts = l4};
            ok = vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.ab_pool) == VK_SUCCESS;
            ai.descriptorPool = G.ab_pool;
            ok = ok && vkAllocateDescriptorSets(G.dev, &ai, G.ab_mm) == VK_SUCCESS;
        }
        if (!ok) { G.pipe_ab = G.pipe_ag = VK_NULL_HANDLE; }
    }
    /* Optional DeltaNet prefill recurrence (COLI_VK_DN_RECUR=0 turns it off). */
    {
        const char *e = getenv("COLI_VK_DN_RECUR");
        char dr_path[512]; derive_dir_file(spv_path, "dn_recur.spv", dr_path, sizeof(dr_path));
        if (!(e && *e == '0')) G.shader_dr = load_spv(G.dev, dr_path);
        if (G.shader_dr && !build_pipeline(G.dev, 7, sizeof(struct PCDR), G.shader_dr, &G.dsl_dr, &G.plyt_dr, &G.pipe_dr, &G.dpool_dr, &G.dset_dr)) {
            vkDestroyShaderModule(G.dev, G.shader_dr, NULL); G.shader_dr = VK_NULL_HANDLE; G.pipe_dr = VK_NULL_HANDLE;
        }
    }
    /* Optional causal prefill attention core (COLI_VK_ATTN_PREFILL=0 turns it off). */
    {
        const char *e = getenv("COLI_VK_ATTN_PREFILL");
        char ap_path[512]; derive_dir_file(spv_path, "attn_prefill.spv", ap_path, sizeof(ap_path));
        if (!(e && *e == '0')) G.shader_ap = load_spv(G.dev, ap_path);
        if (G.shader_ap && !build_pipeline(G.dev, 4, sizeof(struct PCAP), G.shader_ap, &G.dsl_ap, &G.plyt_ap, &G.pipe_ap, &G.dpool_ap, &G.dset_ap)) {
            vkDestroyShaderModule(G.dev, G.shader_ap, NULL); G.shader_ap = VK_NULL_HANDLE; G.pipe_ap = VK_NULL_HANDLE;
        }
        const char *fe = getenv("COLI_VK_ATTN_FLASH");
        char af_path[512]; derive_dir_file(spv_path, "attn_flash.spv", af_path, sizeof(af_path));
        if (G.pipe_ap && G.coop && !(fe && *fe == '0')) G.shader_af = load_spv(G.dev, af_path);
    }

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.eg_cmd), "eg cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.eg_fence), "eg fence");

    if (G.staged) {   /* own pool: command buffers from one pool need external sync too */
        VkCommandPoolCreateInfo upci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
        VKCHECK(vkCreateCommandPool(G.dev, &upci, NULL, &G.up_cpool), "up cmdPool");
        VkCommandBufferAllocateInfo ubi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = G.up_cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VKCHECK(vkAllocateCommandBuffers(G.dev, &ubi, &G.up_cmd), "up cmdBuf");
        VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.up_fence), "up fence");
    }

    G.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G.phys, &p);
    if (getenv("COLI_VK_TS") && getenv("COLI_VK_TS")[0] == '1' && p.limits.timestampComputeAndGraphics) {
        VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 4 + 16};
        if (vkCreateQueryPool(G.dev, &qpi, NULL, &G.tsq) == VK_SUCCESS) { G.ts_on = 1; G.ts_period = p.limits.timestampPeriod; }
    }
    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s%s%s, weights %s\n", p.deviceName, G.qfam, G.memtype,
            G.shader_gu ? ", fused gate+up" : "", G.shader_att ? ", absorb attention" : "",
            G.coop2 ? ", coopmat tiles (gen 2)" : G.coop ? ", coopmat tiles" : "", G.staged ? "staged" : "mapped");
    return 1;
}

int coli_vk_available(void) { return G.ready; }
int coli_vk_staged(void) { return G.ready && G.staged; }

const char *coli_vk_default_spv(char *buf, size_t n) {
    const char *env = getenv("COLI_VK_SHADERS");
    struct stat st;
    if (env && *env) {
        if (!stat(env, &st) && S_ISDIR(st.st_mode)) { snprintf(buf, n, "%s/qmatmul.spv", env); return buf; }
        return env;
    }
#ifdef __linux__
    ssize_t k = readlink("/proc/self/exe", buf, n - 1);
    if (k > 0) {
        buf[k] = 0;
        char *sl = strrchr(buf, '/');
        if (sl && (size_t)(sl + 1 - buf) + sizeof("shaders/qmatmul.spv") <= n) {
            strcpy(sl + 1, "shaders/qmatmul.spv");
            if (!stat(buf, &st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}

void coli_vk_mem_info(size_t *used, size_t *count) {
    if (used) *used = G.used_bytes;
    if (count) *count = G.tensor_count;
}

/* Weight-tensor suballocator: many tensors share a few big VkDeviceMemory blocks.
 * WHY: per-submit driver cost measures LINEAR in the number of distinct device-memory
 * objects the queue actively references (~0.35 ms/submit extra at a 950-expert tier's
 * ~5.7k allocations: decode attention 7.9s @420 -> 17.6s @950, flat once the GPU paths
 * moved to CPU; IDLE allocations cost nothing until first referenced — harness ballast
 * probe). Packing tier+dense uploads into 256 MB arenas keeps the referenced-BO count
 * in the dozens. Arena slices are never reclaimed per-tensor (registry/dense uploads
 * live for the process; the rare fill-failure free leaks its slice, bounded) — a
 * tensor's mem handle stays VK_NULL_HANDLE, which coli_vk_tensor_free's vkFreeMemory
 * treats as the documented no-op. The staged path keeps a second chain in
 * DEVICE_LOCAL-only memory (never mapped); slices there are filled by stage_copy. */
typedef struct VkWArena { VkDeviceMemory mem; uint8_t *base; size_t cap, off; int filled; struct VkWArena *next; } VkWArena;
static VkWArena *g_warena;      /* mapped host-visible chain (G.memtype) */
static VkWArena *g_warena_dl;   /* staged device-local chain (G.memtype_dl); base == NULL */
#define VK_WARENA_BLOCK ((size_t)256 << 20)
static VkResult vk_fence_wait(VkDevice dev, VkFence f);
static int arena_suballoc_locked(size_t bytes, VkBuffer *buf, void **ptr, int dl, VkWArena **ablk) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = (dl ? (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
                     : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
                 (G.has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    uint32_t mt = dl ? G.memtype_dl : G.memtype; VkWArena **head = dl ? &g_warena_dl : &g_warena;
    if (!(req.memoryTypeBits & (1u << mt))) { vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
    size_t align = req.alignment ? req.alignment : 256, off = 0;
    VkWArena *a = *head;
    for (; a; a = a->next) {
        off = (a->off + align - 1) & ~(align - 1);
        if (off + req.size <= a->cap) break;
    }
    if (!a) {
        size_t cap = req.size > VK_WARENA_BLOCK ? (req.size + 4095) & ~(size_t)4095 : VK_WARENA_BLOCK;
        a = calloc(1, sizeof(*a));
        if (!a) { vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = cap, .memoryTypeIndex = mt};
#ifdef VK_EXT_memory_priority
        VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
            .priority = G.prio};
        if (G.has_prio) ai.pNext = &pri;
#endif
        VkMemoryAllocateFlagsInfo maf = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        if (G.has_bda) { maf.pNext = (void *)ai.pNext; ai.pNext = &maf; }
        if (vkAllocateMemory(G.dev, &ai, NULL, &a->mem) != VK_SUCCESS ||
            (!dl && vkMapMemory(G.dev, a->mem, 0, cap, 0, (void **)&a->base) != VK_SUCCESS)) {
            if (a->mem) vkFreeMemory(G.dev, a->mem, NULL);
            free(a); vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
        }
        /* New dl blocks are filled once by the arena_suballoc wrapper, outside
         * g_arena_mx (fence waits are never taken under a lock — see the file-top
         * invariant); mapped blocks never need it. */
        a->filled = dl ? 0 : 1;
        a->cap = cap; a->next = *head; *head = a;
        off = 0;
    }
    VKCHECK(vkBindBufferMemory(G.dev, *buf, a->mem, off), "vkBindBufferMemory");
    if (ptr) *ptr = dl ? NULL : a->base + off;
    a->off = off + req.size;
    *ablk = a;
    return 1;
}
static int arena_suballoc(size_t bytes, VkBuffer *buf, void **ptr, int dl) {
    VkWArena *a = NULL;
    pthread_mutex_lock(&g_arena_mx);
    int r = arena_suballoc_locked(bytes, buf, ptr, dl, &a);
    pthread_mutex_unlock(&g_arena_mx);
    if (r && dl && !a->filled) {
        /* First slice out of a fresh dl block: fill the whole block once, outside
         * g_arena_mx. Without this fill, expert/matmul results computed from a
         * freshly created device-local block differ slightly (within the 1e-3
         * harness gate) and are non-deterministic run to run; filling the block
         * once here — with ANY value, 0x00 and 0xFF measured identical — makes the
         * staged path byte-identical to the mapped path across all 50 harness
         * cases on RX 580 (RADV Polaris). The fill value is irrelevant, so the
         * mechanism is the first GPU-side write/placement of the allocation, not a
         * read of filler bytes; the underlying cause is unresolved (no GPU
         * validation layers on the test host) and is reported upstream as a
         * finding. Borrowing up_cmd/up_fence here is safe because dl allocations
         * only happen under g_upload_mx, so no other suballoc can race this block's
         * `filled` flag. On failure the slice just bound above is destroyed and 0
         * returned; the block stays linked with filled==0 so the next dl
         * allocation retries the fill — the lost slice is a bounded leak, like the
         * existing fill-failure remark above the arena. */
        VkBuffer zb = VK_NULL_HANDLE;
        VkBufferCreateInfo zbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = a->cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VkCommandBufferBeginInfo zbegin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VkSubmitInfo zsi = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &G.up_cmd};
        int zok = vkCreateBuffer(G.dev, &zbi, NULL, &zb) == VK_SUCCESS;
        if (zok) {
            VkMemoryRequirements zreq;
            vkGetBufferMemoryRequirements(G.dev, zb, &zreq);
            zok = zreq.size <= a->cap && (zreq.memoryTypeBits & (1u << G.memtype_dl));
        }
        zok = zok &&
              vkBindBufferMemory(G.dev, zb, a->mem, 0) == VK_SUCCESS &&
              vkResetCommandBuffer(G.up_cmd, 0) == VK_SUCCESS &&
              vkBeginCommandBuffer(G.up_cmd, &zbegin) == VK_SUCCESS;
        if (zok) {
            vkCmdFillBuffer(G.up_cmd, zb, 0, VK_WHOLE_SIZE, 0);
            VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(G.up_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            zok = vkEndCommandBuffer(G.up_cmd) == VK_SUCCESS &&
                  vkResetFences(G.dev, 1, &G.up_fence) == VK_SUCCESS &&
                  vk_submit(G.queue, &zsi, G.up_fence) == VK_SUCCESS &&
                  vk_fence_wait(G.dev, G.up_fence) == VK_SUCCESS;
        }
        if (zb) vkDestroyBuffer(G.dev, zb, NULL);
        if (!zok) { vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
        a->filled = 1;
    }
    return r;
}

/* Grow-on-demand host staging buffer for the staged weight path (4 MB granularity so
 * the per-expert tier uploads never reallocate). Callers hold g_upload_mx. */
static int stage_reserve(size_t bytes) {
    if (G.stage.cap >= bytes) return 1;
    if (G.stage.buf) { vkDestroyBuffer(G.dev, G.stage.buf, NULL); vkFreeMemory(G.dev, G.stage.mem, NULL); }
    G.stage.buf = VK_NULL_HANDLE; G.stage.mem = VK_NULL_HANDLE; G.stage.cap = 0; G.stage.ptr = NULL;
    size_t cap = (bytes + ((size_t)4 << 20) - 1) & ~(((size_t)4 << 20) - 1);
    if (!alloc_buf_mt(cap, &G.stage.buf, &G.stage.mem, &G.stage.ptr, G.memtype_stage,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        G.stage.buf = VK_NULL_HANDLE; G.stage.mem = VK_NULL_HANDLE;
        return 0;
    }
    G.stage.cap = cap;
    return 1;
}

/* Staged upload: pack the padded rows and the scale block into the staging buffer, copy
 * both into the tensor's device-local slices on the upload command buffer, wait. One
 * upload at a time (g_upload_mx); the submit is serialised with decode by vk_submit. */
static pthread_mutex_t g_upload_mx = PTHREAD_MUTEX_INITIALIZER;
static int stage_copy(ColiVkTensor *t, const void *weights, const float *scales,
                      size_t cpu_rb, size_t stride, size_t sb) {
    if (!stage_reserve(t->wbytes + sb)) return 0;
    uint8_t *p = G.stage.ptr;
    memset(p, 0, t->wbytes);
    for (int o = 0; o < t->O; o++)
        memcpy(p + (size_t)o * stride, (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    memcpy(p + t->wbytes, scales, sb);
    VKCHECK(vkResetCommandBuffer(G.up_cmd, 0), "up resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHECK(vkBeginCommandBuffer(G.up_cmd, &begin), "up beginCmd");
    VkBufferCopy cw = {.srcOffset = 0, .dstOffset = 0, .size = t->wbytes};
    VkBufferCopy cs = {.srcOffset = t->wbytes, .dstOffset = 0, .size = sb};
    vkCmdCopyBuffer(G.up_cmd, G.stage.buf, t->wbuf, 1, &cw);
    vkCmdCopyBuffer(G.up_cmd, G.stage.buf, t->sbuf, 1, &cs);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.up_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    VKCHECK(vkEndCommandBuffer(G.up_cmd), "up endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.up_cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.up_fence), "up resetFence");
    VKCHECK(vk_submit(G.queue, &si, G.up_fence), "up queueSubmit");
    if (vk_fence_wait(G.dev, G.up_fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] staged upload fence wait failed — disabling GPU offload\n");
        G.ready = 0; return 0;
    }
    return 1;
}

static int upload_tensor(ColiVkTensor **out, const void *weights, const float *scales,
                         int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 &&              /* fmt=4/7: word-aligned groups only */
        !((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;
    size_t sfl = scale_floats(fmt, I, O, gs);            // fmt=5: O*ceil(I/64) group scales
    t->wbytes = stride * (size_t)O;
    size_t sb = sfl * sizeof(float);
    if (G.staged) {
        pthread_mutex_lock(&g_upload_mx);
        int ok = arena_suballoc(t->wbytes, &t->wbuf, NULL, 1) &&
                 arena_suballoc(sb, &t->sbuf, NULL, 1) &&
                 stage_copy(t, weights, scales, cpu_rb, stride, sb);
        pthread_mutex_unlock(&g_upload_mx);
        if (!ok) {
            if (t->wbuf) vkDestroyBuffer(G.dev, t->wbuf, NULL);
            if (t->sbuf) vkDestroyBuffer(G.dev, t->sbuf, NULL);
            free(t); return 0;
        }
    } else {
        void *wptr;
        if (!arena_suballoc(t->wbytes, &t->wbuf, &wptr, 0)) { free(t); return 0; }
        memset(wptr, 0, t->wbytes);
        for (int o = 0; o < O; o++)                        // copy row-by-row into padded layout
            memcpy((uint8_t *)wptr + (size_t)o * stride,
                   (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
        void *sptr;
        if (!arena_suballoc(sb, &t->sbuf, &sptr, 0)) {
            vkDestroyBuffer(G.dev, t->wbuf, NULL); free(t); return 0;
        }
        memcpy(sptr, scales, sb);
    }
    // Counters are touched concurrently: frees run from expert_load under
    // `#pragma omp parallel`, so RMW them atomically (torn counts otherwise).
    __atomic_add_fetch(&G.used_bytes, t->wbytes + sfl * sizeof(float), __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    *out = t;
    return 1;
}

/* Upload a resident tensor without computing (for the expert tier: gate/up/down are
 * uploaded once, then driven by coli_vk_expert_group). Returns 0 on failure. */
int coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp) {
    if (!G.ready) return 0;
    return upload_tensor(tensor, weights, scales, fmt, I, O, grp);
}

/* Sync-path fence wait. A blocked vkWaitForFences pays a scheduler wake on
 * signal (~50-150 us) — and the engine fences ~2 sync submits per layer per
 * token, so the wakes alone cost seconds per run. Spin on vkGetFenceStatus for
 * a short budget first (the common decode dispatch completes in 0.5-2 ms),
 * then fall back to the blocking wait. The spinning thread is stalled on the
 * GPU result anyway. COLI_VK_SPIN_US=0 restores the pure blocking wait. */
static VkResult vk_fence_wait(VkDevice dev, VkFence f) {
    if (g_vk_spin_us > 0) {
        double t0 = vk_now();
        do {
            VkResult r = vkGetFenceStatus(dev, f);
            if (r != VK_NOT_READY) return r;   /* VK_SUCCESS or a real error */
        } while ((vk_now() - t0) * 1000.0 < (double)g_vk_spin_us);
    }
    return vkWaitForFences(dev, 1, &f, VK_TRUE, 10000000000ULL);
}

/* Global submit/wait totals across EVERY synchronous GPU path (VK_PROF=1) — the
 * per-path counters miss traffic that flows through the fused pair/absorb/group
 * entries, so the tier-size-linear per-submit tax is localized here instead. */
static double g_vsub_ms, g_vwait_ms; static long g_vsub_n;
static void vkprof_tick(void) {
    if ((++g_vsub_n & 2047) == 0)
        fprintf(stderr, "[VK_PROF sub] n=%ld | submit %.0f | wait %.0f ms\n", g_vsub_n, g_vsub_ms, g_vwait_ms);
}

int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O, int gs) {
    if (!G.ready || S < 1 || !upload_tensor(tensor, weights, scales, fmt, I, O, gs)) return 0;
    ColiVkTensor *t = *tensor;
    /* VK_PROF=1: phase split of the dense per-call cost, printed every 8192 calls —
     * separates our code (memcpy/desc/record) from the driver (submit) and the GPU
     * (fence wait) to localize the tier-size-linear tax. */
    static double p_x, p_desc, p_rec, p_sub, p_wait, p_y; static long p_n;
    double t0 = G.eg_prof ? vk_now() : 0, tA;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    VkBuffer old_x = G.x.buf, old_y = G.y.buf;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, yb, G.memtype_cached)) return 0;  /* y read back */
    VkBuffer xin = arena_buf(x, xb);                 /* host arena: bound as is, no copy */
    if (!xin) { memcpy(G.x.ptr, x, xb); xin = G.x.buf; }
    if (G.eg_prof) { tA = vk_now(); p_x += tA - t0; t0 = tA; }

    /* Rebind descriptors only when the tensor or a scratch buffer changed (a realloc
     * makes the old VkBuffer handle stale); otherwise the previous binding is still valid. */
    int rebind = G.bound_tensor != t || G.x.buf != old_x || G.y.buf != old_y
              || G.bound_xbuf != xin || G.bound_ybuf != G.y.buf;
    if (rebind) {
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = xin, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[4];
        for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset,
            .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
        vkUpdateDescriptorSets(G.dev, 4, w, 0, NULL);
        G.bound_tensor = t; G.bound_xbuf = xin; G.bound_ybuf = G.y.buf;
    }
    if (G.eg_prof) { tA = vk_now(); p_desc += tA - t0; t0 = tA; }

    /* Re-record the command buffer only when the binding or the dispatch shape changed.
     * Recorded WITHOUT one-time-submit so the same buffer can be resubmitted verbatim —
     * for repeated calls to the same expert this drops setup to a bare submit+wait. */
    if (rebind || !G.cmd_ready || G.bound_S != S || G.bound_I != I || G.bound_O != O) {
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        ts_begin(G.cmd, TS_SLOT_CMD);
        int co = coop_use(fmt, S, I, O, t->gs, 0), mr = !co && mr_use(fmt, S, I, O);
        uint32_t cgx = 0, cgy = 0;
        VkPipeline cpl = co ? coop_pick(0, S, O, &cgx, &cgy) : VK_NULL_HANDLE;
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, co ? cpl : mr ? G.pipe_mr : G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
        struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        /* Grid-stride shader: one subgroup per output row (~8 rows/workgroup at wave32).
         * Launch ~O/8 workgroups for occupancy; the shader loops to cover any O / wave width.
         * The multi-row shader covers MR token rows per workgroup in y. */
        if (co) vkCmdDispatch(G.cmd, cgx, cgy, 1);
        else vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)(mr ? (S + G.mr - 1) / G.mr : S), 1);
        host_read_barrier(G.cmd);
        ts_end(G.cmd, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
        G.cmd_ready = 1; G.bound_S = S; G.bound_I = I; G.bound_O = O;
    }
    if (G.eg_prof) { tA = vk_now(); p_rec += tA - t0; t0 = tA; }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double th0 = vk_now();
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { tA = vk_now(); p_sub += tA - t0; g_vsub_ms += tA - t0; t0 = tA; }
    // Bounded wait: a GPU hang/TDR must never wedge the process. 10s is orders of
    // magnitude over a single-GEMV dispatch; on timeout/device-loss disable VK for
    // the rest of the run and fall back to CPU (the caller degrades on our 0 return).
    VkResult wr = vk_fence_wait(G.dev, G.fence);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d — disabling GPU offload, staying on CPU\n", wr);
        G.ready = 0;
        return 0;
    }
    ts_read(TS_SLOT_CMD, TS_MATMUL, vk_now() - th0);
    if (G.eg_prof) { tA = vk_now(); p_wait += tA - t0; g_vwait_ms += tA - t0; t0 = tA; vkprof_tick(); }
    memcpy(y, G.y.ptr, yb);
    if (G.eg_prof) {
        p_y += vk_now() - t0;
        if ((++p_n & 8191) == 0)
            fprintf(stderr, "[VK_PROF dense] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f | wait %.0f | memcpy_y %.0f ms\n",
                    p_n, p_x, p_desc, p_rec, p_sub, p_wait, p_y);
    }
    return 1;
}

/* Fused first half of the expert MLP: hidden = silu(gate(x)) * up(x), computed in ONE
 * dispatch that reads x once for both projections. gate/up are resident (uploaded on
 * first call). D = input (hidden) dim, I = moe_inter. Returns 0 -> caller falls back. */
int coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up, float *hidden, const float *x,
                    const void *gw, const float *gs, const void *uw, const float *us,
                    int fmt, int S, int D, int I, int grp) {
    if (!G.ready || !G.shader_gu || S < 1 || D > 6144) return 0;   /* shader stages x in xsh[6144] */
    if (!upload_tensor(gate, gw, gs, fmt, D, I, grp) || !upload_tensor(up, uw, us, fmt, D, I, grp)) return 0;
    ColiVkTensor *tg = *gate, *tu = *up;
    size_t xb = (size_t)S * D * sizeof(float), hb = (size_t)S * I * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.h, hb, G.memtype_cached)) return 0;  /* hidden read back */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo bi[6] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tg->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tg->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = tu->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tu->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.h.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_gu,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    ts_begin(G.cmd, TS_SLOT_CMD);
    int co = coop_use(fmt, S, D, I, tg->gs, 0), mr = !co && mr_use(fmt, S, D, I);
    uint32_t cgx = 0, cgy = 0;
    VkPipeline cpl = co ? coop_pick(1, S, I, &cgx, &cgy) : VK_NULL_HANDLE;
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, co ? cpl : mr ? G.pipe_gu_mr : G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PCGU pc = pcgu(fmt, S, D, I, tg->rowWords, tg->gs);   // PC.I = input D, PC.O = moe_inter I
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (co) vkCmdDispatch(G.cmd, cgx, cgy, 1);
    else vkCmdDispatch(G.cmd, (uint32_t)((I + 7) / 8), (uint32_t)(mr ? (S + G.mr - 1) / G.mr : S), 1);
    host_read_barrier(G.cmd);
    ts_end(G.cmd, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    double th0 = vk_now();
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    ts_read(TS_SLOT_CMD, TS_GATEUP, vk_now() - th0);
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(hidden, G.h.ptr, hb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi);
/* DeltaNet gated delta rule over a prefill block (dn_recur.comp). qn/kn: [S][vk][kdim]
 * l2-normalised (q scaled), v row t at vsrc[t*vstride + voff + h*vdim], beta/gexp:
 * [S][vh] (gexp = exp(g)), state: [vh][kdim][vdim] read and written back in place,
 * outv: [S][vh][vdim]. Returns 0 when unavailable -> the caller's CPU recurrence. */
/* Host-visible staging for coli_vk_dn_recur: which 0..4 = qn, kn, v, beta, gexp.
 * A caller that fills these in place saves the upload copy (the call skips the
 * memcpy for any input that already is the staged pointer). NULL on failure. */
float *coli_vk_dn_stage(int which, size_t bytes) {
    Scratch *sc[5] = {&G.dr_q, &G.dr_k, &G.dr_v, &G.dr_b, &G.dr_g};
    if (!G.ready || which < 0 || which > 4 || !scratch_reserve(sc[which], bytes)) return NULL;
    return (float *)sc[which]->ptr;
}
int coli_vk_dn_recur(float *outv, float *state, const float *qn, const float *kn,
                     const float *vsrc, int vstride, int voff, const float *beta, const float *gexp,
                     int S, int vh, int vk, int kdim, int vdim) {
    if (!G.ready || !G.pipe_dr || S < 1 || vk < 1 || vh % vk || kdim % 16 || kdim > 256 || vdim < 1) return 0;
    size_t qb = (size_t)S * vk * kdim * 4, vb = ((size_t)(S - 1) * vstride + voff + (size_t)vh * vdim) * 4;
    size_t bb = (size_t)S * vh * 4, sb = (size_t)vh * kdim * vdim * 4, ob = (size_t)S * vh * vdim * 4;
    if (!scratch_reserve(&G.dr_q, qb) || !scratch_reserve(&G.dr_k, qb) || !scratch_reserve(&G.dr_v, vb) ||
        !scratch_reserve(&G.dr_b, bb) || !scratch_reserve(&G.dr_g, bb) ||
        !scratch_reserve_mt(&G.dr_s, sb, G.memtype_cached) || !scratch_reserve_mt(&G.dr_o, ob, G.memtype_cached)) return 0;
    double dp0 = vk_now();
    if (qn != G.dr_q.ptr) memcpy(G.dr_q.ptr, qn, qb);
    if (kn != G.dr_k.ptr) memcpy(G.dr_k.ptr, kn, qb);
    if (vsrc != G.dr_v.ptr) memcpy(G.dr_v.ptr, vsrc, vb);
    if (beta != G.dr_b.ptr) memcpy(G.dr_b.ptr, beta, bb);
    if (gexp != G.dr_g.ptr) memcpy(G.dr_g.ptr, gexp, bb);
    memcpy(G.dr_s.ptr, state, sb);
    double dp1 = vk_now();
    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.dr_q.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.dr_k.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.dr_v.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.dr_b.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.dr_g.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.dr_s.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.dr_o.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_dr, 7, bi);
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    ts_begin(G.cmd, TS_SLOT_CMD);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_dr);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dr, 0, 1, &G.dset_dr, 0, NULL);
    struct PCDR pc = {S, vh, vk, kdim, vdim, vstride, voff};
    vkCmdPushConstants(G.cmd, G.plyt_dr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)vh, (uint32_t)((vdim + 3) / 4), 1);
    host_read_barrier(G.cmd);
    ts_end(G.cmd, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double th0 = vk_now();
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    ts_read(TS_SLOT_CMD, TS_DNREC, vk_now() - th0);
    double dp2 = vk_now();
    memcpy(outv, G.dr_o.ptr, ob); memcpy(state, G.dr_s.ptr, sb);
    if (getenv("VK_PROF")) fprintf(stderr, "[VK_PROF] dn_recur upload %.2f | gpu %.2f | readback %.2f ms\n", dp1 - dp0, dp2 - dp1, vk_now() - dp2);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* Record one matmul y = x W^T of an uploaded tensor into cb, the same shader choice
 * as coli_vk_matmul (tiles / multi-row / one-row), on descriptor set `set`. */
static void rec_mm_off(VkCommandBuffer cb, VkDescriptorSet set, ColiVkTensor *t, VkBuffer xb, VkBuffer yb,
                       VkDeviceSize yoff, int S) {
    VkDescriptorBufferInfo bi[4] = {
        {.buffer = xb, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
        {.buffer = yb, .offset = yoff, .range = (VkDeviceSize)S * t->O * sizeof(float)}};
    wr_desc(set, 4, bi);
    int co = coop_use(t->fmt, S, t->I, t->O, t->gs, 0), mr = !co && mr_use(t->fmt, S, t->I, t->O);
    uint32_t gx = 0, gy = 0;
    VkPipeline pl = co ? coop_pick(0, S, t->O, &gx, &gy) : mr ? G.pipe_mr : G.pipe;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &set, 0, NULL);
    struct PC pc = {t->fmt, S, t->I, t->O, t->rowWords, t->gs};
    vkCmdPushConstants(cb, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (co) vkCmdDispatch(cb, gx, gy, 1);
    else vkCmdDispatch(cb, (uint32_t)((t->O + 7) / 8), (uint32_t)(mr ? (S + G.mr - 1) / G.mr : S), 1);
}
static void rec_mm(VkCommandBuffer cb, VkDescriptorSet set, ColiVkTensor *t, VkBuffer xb, VkBuffer yb, int S) {
    rec_mm_off(cb, set, t, xb, yb, 0, S);
}
static void cc_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
/* A whole DeltaNet layer of a prefill block in ONE submit, nothing read back in
 * between: qkv|z projection (proj, int8 [proj_dim x H]) -> causal conv + SiLU ->
 * l2norm/beta/exp(g) -> gated delta rule -> gated RMSNorm -> out_proj (outp,
 * [H x value_dim]). b|a ([S][2*vh], small f32 projections) come from the caller;
 * par = A_log | dt_bias. ring (conv_dim x (convk-1)) and state (vh x kdim x vdim)
 * are read and written back; y = [S][H]. 0 -> caller's path. */
int coli_vk_dn_block(ColiVkTensor *proj, ColiVkTensor *outp, const float *x, const float *ba,
                     const float *convw, const float *par, const float *normw, float *ring, float *state,
                     int S, int H, int conv_dim, int convk, int vh, int vk, int kdim, int vdim,
                     float eps, float qscale, float *y) {
    if (!G.ready || !G.pipe_dc || !G.pipe_dr || !proj || !outp || S < 1 || convk < 2 || convk > 17 ||
        vh % vk || kdim % 16 || kdim > 256) return 0;
    int value_dim = vh * vdim, pd = proj->O, ktot = vk * kdim;
    if (proj->I != H || pd != conv_dim + value_dim || outp->I != value_dim || outp->O != H) return 0;
    size_t f = sizeof(float);
    size_t nx = (size_t)S * H * f, nqz = (size_t)S * pd * f, nba = (size_t)S * 2 * vh * f, npar = (size_t)2 * vh * f;
    size_t ncw = (size_t)conv_dim * convk * f, nring = (size_t)conv_dim * (convk - 1) * f, nconv = (size_t)S * conv_dim * f;
    size_t nqk = (size_t)S * ktot * f, nbg = (size_t)2 * S * vh * f, nst = (size_t)vh * kdim * vdim * f;
    size_t nov = (size_t)S * value_dim * f, nnw = (size_t)vdim * f;
    if (!scratch_reserve(&G.db_x, nx) || !scratch_reserve(&G.db_qz, nqz) || !scratch_reserve(&G.db_ba, nba) ||
        !scratch_reserve(&G.db_par, npar) || !scratch_reserve(&G.db_cw, ncw) || !scratch_reserve(&G.db_ring, nring) ||
        !scratch_reserve(&G.db_conv, nconv) || !scratch_reserve(&G.dr_q, nqk) || !scratch_reserve(&G.dr_k, nqk) ||
        !scratch_reserve(&G.db_bg, nbg) || !scratch_reserve_mt(&G.dr_s, nst, G.memtype_cached) ||
        !scratch_reserve(&G.dr_o, nov) || !scratch_reserve(&G.db_nw, nnw) || !scratch_reserve(&G.db_or, nov) ||
        !scratch_reserve_mt(&G.db_y, nx, G.memtype_cached)) return 0;
    /* the input stays a copy: bound from the (host-cached) arena, the projection
     * re-reading x once per output block ran 121 -> 209 ms (30 layers, 1011 tokens) */
    VkBuffer xa = getenv("COLI_VK_DN_ARENA_X") ? arena_buf(x, nx) : VK_NULL_HANDLE, ya = arena_buf(y, nx);
    if (!xa) memcpy(G.db_x.ptr, x, nx);
    memcpy(G.db_ba.ptr, ba, nba); memcpy(G.db_par.ptr, par, npar);
    memcpy(G.db_cw.ptr, convw, ncw); memcpy(G.db_ring.ptr, ring, nring); memcpy(G.dr_s.ptr, state, nst);
    memcpy(G.db_nw.ptr, normw, nnw);
    VkCommandBuffer cb = G.cmd;
    VKCHECK(vkResetCommandBuffer(cb, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(cb, &begin), "beginCmd");
    ts_begin(cb, TS_SLOT_CMD); ts_stage0(cb);
    /* 1. qkv|z = x Wp^T */
    rec_mm(cb, G.db_mm[0], proj, xa ? xa : G.db_x.buf, G.db_qz.buf, S);
    cc_barrier(cb); ts_stage(cb, 0);
    /* 2. conv + SiLU */
    { VkDescriptorBufferInfo bi[4] = {{G.db_qz.buf, 0, VK_WHOLE_SIZE}, {G.db_cw.buf, 0, VK_WHOLE_SIZE},
                                      {G.db_ring.buf, 0, VK_WHOLE_SIZE}, {G.db_conv.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_dc, 4, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_dc);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dc, 0, 1, &G.dset_dc, 0, NULL);
      struct PCDC pc = {S, conv_dim, convk, pd};
      vkCmdPushConstants(cb, G.plyt_dc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)((conv_dim + 255) / 256), (uint32_t)S, 1); }
    cc_barrier(cb); ts_stage(cb, 1);
    /* 3. l2norm q/k, beta, exp(g) */
    { VkDescriptorBufferInfo bi[6] = {{G.db_conv.buf, 0, VK_WHOLE_SIZE}, {G.db_ba.buf, 0, VK_WHOLE_SIZE},
                                      {G.db_par.buf, 0, VK_WHOLE_SIZE}, {G.dr_q.buf, 0, VK_WHOLE_SIZE},
                                      {G.dr_k.buf, 0, VK_WHOLE_SIZE}, {G.db_bg.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_dp, 6, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_dp);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dp, 0, 1, &G.dset_dp, 0, NULL);
      struct PCDP pc = {S, conv_dim, vk, vh, kdim, qscale};
      vkCmdPushConstants(cb, G.plyt_dp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)((S + 7) / 8), (uint32_t)vk, 1); }
    cc_barrier(cb); ts_stage(cb, 2);
    /* 4. recurrence (v straight out of the conv output) */
    { VkDeviceSize hb = (VkDeviceSize)S * vh * f;
      VkDescriptorBufferInfo bi[7] = {{G.dr_q.buf, 0, VK_WHOLE_SIZE}, {G.dr_k.buf, 0, VK_WHOLE_SIZE},
                                      {G.db_conv.buf, 0, VK_WHOLE_SIZE}, {G.db_bg.buf, 0, hb}, {G.db_bg.buf, hb, hb},
                                      {G.dr_s.buf, 0, VK_WHOLE_SIZE}, {G.dr_o.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_dr, 7, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_dr);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dr, 0, 1, &G.dset_dr, 0, NULL);
      struct PCDR pc = {S, vh, vk, kdim, vdim, conv_dim, 2 * ktot};
      vkCmdPushConstants(cb, G.plyt_dr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)vh, (uint32_t)((vdim + 3) / 4), 1); }
    cc_barrier(cb); ts_stage(cb, 3);
    /* 5. gated RMSNorm */
    { VkDescriptorBufferInfo bi[4] = {{G.dr_o.buf, 0, VK_WHOLE_SIZE}, {G.db_qz.buf, 0, VK_WHOLE_SIZE},
                                      {G.db_nw.buf, 0, VK_WHOLE_SIZE}, {G.db_or.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_dg, 4, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_dg);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dg, 0, 1, &G.dset_dg, 0, NULL);
      struct PCDG pc = {S, vh, vdim, pd, conv_dim, eps};
      vkCmdPushConstants(cb, G.plyt_dg, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)((S + 7) / 8), (uint32_t)vh, 1); }
    cc_barrier(cb); ts_stage(cb, 4);
    /* 6. out_proj */
    rec_mm(cb, G.db_mm[1], outp, G.db_or.buf, ya ? ya : G.db_y.buf, S); ts_stage(cb, 5);
    host_read_barrier(cb);
    ts_end(cb, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(cb), "endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double th0 = vk_now();
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    ts_read(TS_SLOT_CMD, TS_DNBLOCK, vk_now() - th0); ts_stage_read(TSB_DN, 6);
    if (!ya) memcpy(y, G.db_y.ptr, nx);
    memcpy(state, G.dr_s.ptr, nst);
    /* new ring: the last convk-1 unconvolved qkv inputs (older ones from the old ring) */
    /* (the last km rows of db_qz are copied out in one piece first: scattered
     * reads from that mapped, uncached scratch cost ~250 ns each) */
    { int km = convk - 1, r0 = S - km > 0 ? S - km : 0, nr = S - r0;
      float *old = malloc(nring), *qz = malloc((size_t)nr * pd * f);
      if (!old || !qz) { free(old); free(qz); return 0; }
      memcpy(old, ring, nring);
      memcpy(qz, (const float *)G.db_qz.ptr + (size_t)r0 * pd, (size_t)nr * pd * f);
      for (int cc = 0; cc < conv_dim; cc++)
          for (int kk = 0; kk < km; kk++) {
              int src = S - km + kk;
              ring[cc * km + kk] = src >= 0 ? qz[(size_t)(src - r0) * pd + cc] : old[cc * km + src + km];
          }
      free(old); free(qz); }
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* Matrix-unit attention core for this shape? attn_flash.comp folds all query
 * heads of a kv head into its 16 rows (so H/KV must divide 16), needs a head dim
 * that is a multiple of 64 up to 256 and wave64 (4 subgroups, 16-lane clusters).
 * Returns the pipeline (built on first use for this shape) or NULL. */
static VkPipeline af_pipeline(int H, int KV, int hd) {
    int gq = H / KV;
    if (!G.shader_af || G.sgsize != 64 || hd % 64 || hd > 256 || gq < 1 || gq > 16 || 16 % gq) return VK_NULL_HANDLE;
    if (G.pipe_af && G.af_hd == hd && G.af_gq == gq) return G.pipe_af;
    if (G.pipe_af) { vkDeviceWaitIdle(G.dev); vkDestroyPipeline(G.dev, G.pipe_af, NULL); G.pipe_af = VK_NULL_HANDLE; }
    int sc[2] = {hd, gq};
    VkSpecializationMapEntry me[2] = {{.constantID = 0, .offset = 0, .size = sizeof(int)},
                                      {.constantID = 1, .offset = sizeof(int), .size = sizeof(int)}};
    VkSpecializationInfo spi = {.mapEntryCount = 2, .pMapEntries = me, .dataSize = sizeof sc, .pData = sc};
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader_af, .pName = "main",
                  .pSpecializationInfo = &spi},
        .layout = G.plyt_ap};
    if (vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &G.pipe_af) != VK_SUCCESS) {
        G.pipe_af = VK_NULL_HANDLE; vkDestroyShaderModule(G.dev, G.shader_af, NULL); G.shader_af = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    G.af_hd = hd; G.af_gq = gq;
    return G.pipe_af;
}

/* A whole gated-attention layer of a prefill block in ONE submit: q/k/v projections
 * -> split/norm/RoPE/KV-cache rows (attn_prep) -> causal core (attn_prefill) ->
 * sigmoid gate (attn_gate) -> o_proj. K/V rows 0..pos_base-1 come from the host
 * caches (row t of kv head g at (g*ldt + t)*hd); the new rows are written back
 * there. qn/kn: q_norm / k_norm weights [hd] or NULL. out = [S][D]. 0 -> caller. */
int coli_vk_attn_block(ColiVkTensor *tq, ColiVkTensor *tk, ColiVkTensor *tv, ColiVkTensor *to,
                       const float *x, const float *qn, const float *kn, float *Kc, float *Vc, int ldt,
                       int S, int D, int H, int KV, int hd, int qdim, int rotary, int pos_base,
                       float eps, float theta, float scale, float *out) {
    if (!G.ready || !G.pipe_ab || !G.pipe_ap || !tq || !tk || !tv || !to || S < 1 || KV < 1 || H % KV ||
        hd < 1 || hd > 512 || rotary > hd) return 0;
    int gate_dim = qdim > hd ? qdim - hd : 0, nt = pos_base + S;
    if ((gate_dim && gate_dim != hd) || tq->I != D || tq->O != H * qdim || tk->I != D || tk->O != KV * hd ||
        tv->I != D || tv->O != KV * hd || to->I != H * hd || to->O != D || nt > ldt) return 0;
    size_t f = sizeof(float);
    size_t nx = (size_t)S * D * f, nq = (size_t)S * H * qdim * f, nk = (size_t)S * KV * hd * f;
    size_t nqb = (size_t)S * H * hd * f, nkc = (size_t)KV * nt * hd * f;
    if (!scratch_reserve(&G.ab_x, nx) || !scratch_reserve(&G.ab_qkv, nq + 2 * nk) || !scratch_reserve(&G.ab_nw, 2 * (size_t)hd * f) ||
        !scratch_reserve(&G.ab_qb, nqb) || !scratch_reserve(&G.ab_gb, nqb) ||
        !scratch_reserve_mt(&G.ab_k, nkc, G.memtype_cached) || !scratch_reserve_mt(&G.ab_v, nkc, G.memtype_cached) ||
        !scratch_reserve(&G.ab_cx, nqb) || !scratch_reserve(&G.ab_ag, nqb) || !scratch_reserve_mt(&G.ab_y, nx, G.memtype_cached)) return 0;
    VkBuffer xa = arena_buf(x, nx), ya = arena_buf(out, nx);
    if (!xa) memcpy(G.ab_x.ptr, x, nx);
    float *nw = (float *)G.ab_nw.ptr;
    for (int d = 0; d < hd; d++) { nw[d] = qn ? qn[d] : 0.f; nw[hd + d] = kn ? kn[d] : 0.f; }
    for (int g = 0; g < KV && pos_base > 0; g++) {   /* cached prefix keys/values */
        memcpy((char *)G.ab_k.ptr + (size_t)g * nt * hd * f, Kc + (size_t)g * ldt * hd, (size_t)pos_base * hd * f);
        memcpy((char *)G.ab_v.ptr + (size_t)g * nt * hd * f, Vc + (size_t)g * ldt * hd, (size_t)pos_base * hd * f);
    }
    VkCommandBuffer cb = G.cmd;
    VKCHECK(vkResetCommandBuffer(cb, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(cb, &begin), "beginCmd");
    ts_begin(cb, TS_SLOT_CMD); ts_stage0(cb);
    /* 1. q | k | v projections into one buffer */
    /* one after the other: run concurrently (no barrier) the three measured
     * 74 ms over 10 layers, serialised 56 ms (the small k/v tiles compete with q's) */
    rec_mm_off(cb, G.ab_mm[0], tq, xa ? xa : G.ab_x.buf, G.ab_qkv.buf, 0, S);
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 0);
    rec_mm_off(cb, G.ab_mm[1], tk, xa ? xa : G.ab_x.buf, G.ab_qkv.buf, nq, S);
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 1);
    rec_mm_off(cb, G.ab_mm[2], tv, xa ? xa : G.ab_x.buf, G.ab_qkv.buf, nq + nk, S);
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 2);
    /* 2. split, norm, RoPE, cache rows */
    { VkDescriptorBufferInfo bi[6] = {{G.ab_qkv.buf, 0, VK_WHOLE_SIZE}, {G.ab_nw.buf, 0, VK_WHOLE_SIZE},
                                      {G.ab_qb.buf, 0, VK_WHOLE_SIZE}, {G.ab_gb.buf, 0, VK_WHOLE_SIZE},
                                      {G.ab_k.buf, 0, VK_WHOLE_SIZE}, {G.ab_v.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_ab, 6, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_ab);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_ab, 0, 1, &G.dset_ab, 0, NULL);
      struct PCAB pc = {S, H, KV, hd, qdim, gate_dim, rotary, pos_base, nt, qn != NULL, kn != NULL,
                        0, (int)(nq / f), (int)((nq + nk) / f), eps, theta};
      vkCmdPushConstants(cb, G.plyt_ab, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)S, (uint32_t)(H + KV), 1); }
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 3);
    /* 3. causal core */
    { VkDescriptorBufferInfo bi[4] = {{G.ab_qb.buf, 0, VK_WHOLE_SIZE}, {G.ab_k.buf, 0, VK_WHOLE_SIZE},
                                      {G.ab_v.buf, 0, VK_WHOLE_SIZE}, {G.ab_cx.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_ap, 4, bi);
      VkPipeline af = af_pipeline(H, KV, hd);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, af ? af : G.pipe_ap);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_ap, 0, 1, &G.dset_ap, 0, NULL);
      struct PCAP pc = {S, H, KV, hd, pos_base, nt, scale};
      vkCmdPushConstants(cb, G.plyt_ap, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      if (af) { int tpw = 16 / (H / KV); vkCmdDispatch(cb, (uint32_t)((S + tpw - 1) / tpw), (uint32_t)KV, 1); }
      else vkCmdDispatch(cb, (uint32_t)S, (uint32_t)H, 1); }
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 4);
    /* 4. gate */
    { VkDescriptorBufferInfo bi[3] = {{G.ab_cx.buf, 0, VK_WHOLE_SIZE}, {G.ab_gb.buf, 0, VK_WHOLE_SIZE}, {G.ab_ag.buf, 0, VK_WHOLE_SIZE}};
      wr_desc(G.dset_ag, 3, bi);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_ag);
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_ag, 0, 1, &G.dset_ag, 0, NULL);
      int n = S * H * hd;
      struct PCAG pc = {n, gate_dim > 0};
      vkCmdPushConstants(cb, G.plyt_ag, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cb, (uint32_t)((n + 255) / 256), 1, 1); }
    cc_barrier(cb); ts_stageb(cb, TSB_ATTN, 5);
    /* 5. o_proj */
    rec_mm(cb, G.ab_mm[3], to, G.ab_ag.buf, ya ? ya : G.ab_y.buf, S); ts_stageb(cb, TSB_ATTN, 6);
    host_read_barrier(cb);
    ts_end(cb, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(cb), "endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double th0 = vk_now();
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    ts_read(TS_SLOT_CMD, TS_ATTN, vk_now() - th0); ts_stage_read(TSB_ATTN, 7);
    if (!ya) memcpy(out, G.ab_y.ptr, nx);
    for (int g = 0; g < KV; g++) {   /* new rows back into the host caches (decode reads them) */
        memcpy(Kc + ((size_t)g * ldt + pos_base) * hd, (char *)G.ab_k.ptr + ((size_t)g * nt + pos_base) * hd * f, (size_t)S * hd * f);
        memcpy(Vc + ((size_t)g * ldt + pos_base) * hd, (char *)G.ab_v.ptr + ((size_t)g * nt + pos_base) * hd * f, (size_t)S * hd * f);
    }
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* Causal attention core of a prefill block (attn_prefill.comp): q [S][H][hd] after
 * norm/RoPE, K/V the engine's per-kv-head caches, row t of kv head g at
 * (g*ldt + t)*hd, keys 0 .. pos_base+S-1 valid; ctx [S][H][hd] out. Only the valid
 * key rows travel (packed per kv head). Returns 0 when unavailable -> CPU path. */
int coli_vk_attn_prefill(float *ctx, const float *q, const float *K, const float *V, int ldt,
                         int S, int H, int KV, int hd, int pos_base, float scale) {
    if (!G.ready || !G.pipe_ap || G.sgsize != 64 || S < 1 || KV < 1 || H % KV || hd < 1 || hd > 512) return 0;
    int nt = pos_base + S;
    size_t qb = (size_t)S * H * hd * sizeof(float), kb = (size_t)KV * nt * hd * sizeof(float);
    if (!scratch_reserve(&G.ap_q, qb) || !scratch_reserve(&G.ap_k, kb) || !scratch_reserve(&G.ap_v, kb) ||
        !scratch_reserve_mt(&G.ap_o, qb, G.memtype_cached)) return 0;
    memcpy(G.ap_q.ptr, q, qb);
    for (int g = 0; g < KV; g++) {
        memcpy((char *)G.ap_k.ptr + (size_t)g * nt * hd * sizeof(float), K + (size_t)g * ldt * hd, (size_t)nt * hd * sizeof(float));
        memcpy((char *)G.ap_v.ptr + (size_t)g * nt * hd * sizeof(float), V + (size_t)g * ldt * hd, (size_t)nt * hd * sizeof(float));
    }
    VkDescriptorBufferInfo bi[4] = {
        {.buffer = G.ap_q.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.ap_k.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.ap_v.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.ap_o.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_ap, 4, bi);
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    ts_begin(G.cmd, TS_SLOT_CMD);
    VkPipeline af = af_pipeline(H, KV, hd);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, af ? af : G.pipe_ap);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_ap, 0, 1, &G.dset_ap, 0, NULL);
    struct PCAP pc = {S, H, KV, hd, pos_base, nt, scale};
    vkCmdPushConstants(G.cmd, G.plyt_ap, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (af) { int tpw = 16 / (H / KV); vkCmdDispatch(G.cmd, (uint32_t)((S + tpw - 1) / tpw), (uint32_t)KV, 1); }
    else vkCmdDispatch(G.cmd, (uint32_t)S, (uint32_t)H, 1);
    host_read_barrier(G.cmd);
    ts_end(G.cmd, TS_SLOT_CMD);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double th0 = vk_now();
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    ts_read(TS_SLOT_CMD, TS_ATTN, vk_now() - th0);
    memcpy(ctx, G.ap_o.ptr, qb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

static void wr_desc_dev(VkDevice dev, VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    VkWriteDescriptorSet w[8];   /* up to 8 bindings (dn_recur uses 7) */
    if (n > 8) n = 8;
    for (int i = 0; i < n; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = (uint32_t)i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(dev, (uint32_t)n, w, 0, NULL);
}
static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    wr_desc_dev(G.dev, set, n, bi);
}

/* Full batched expert MLP for `count` experts, hidden staying on-device:
 * for each c, hidden_c = silu(gate_c(x_c))*up_c(x_c) (fused), then y_c = down_c(hidden_c).
 * x/y are packed [sum(rows)*D]; experts are resident VkTensors (gate/up: D->I, down: I->D).
 * Mirrors coli_cuda_expert_group. Split into prepare+submit / take so the caller can
 * overlap the GPU batch with its own CPU share (issue -> CPU rows -> take); the group
 * runs on its OWN command buffer + fence, so in-flight work never collides with the
 * main pipeline (dense matmuls, absorb attention). Returns 0 -> caller falls back. */
static int eg_prepare_submit(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                             ColiVkTensor *const *downs, const int *rows, int count,
                             const float *x, const ExpertPrefill *ep) {
    double t_in = vk_now();
    if (!G.ready || !G.shader_gu || count < 1 || count > (ep ? EG_MAX_EXPERTS : 64)) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int D = g0->I, I = g0->O, total = 0, off[EG_MAX_EXPERTS];
    if (D > 6144) return 0;   /* gate_up shader stages x in xsh[6144] */
    if (!downs[0]) return 0;
    /* down may be a different quant than gate/up (per-projection containers, e.g.
     * --up-bits 3); see the per-expert check below */
    for (int c = 0; c < count; c++) {
        if (!gates[c] || !ups[c] || !downs[c] || rows[c] < 1 || rows[c] > INT_MAX-total ||
            gates[c]->dev || ups[c]->dev || downs[c]->dev) return 0;
        off[c] = total; total += rows[c];
        /* formats per expert (the prefill batch mixes the int4 routed experts with the
         * int8 shared expert); gate and up of ONE expert must match, the fused
         * gate_up shader decodes both with one fmt */
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != gates[c]->fmt || ups[c]->gs != gates[c]->gs ||
            downs[c]->I != I || downs[c]->O != D) return 0;
    }
    if (ep) {
        if (!G.ep_pipe[0] || !G.ep_pipe[1] || ep->S < 2 || ep->K < 1 ||
            ep->S > INT_MAX/ep->K || total > ep->S*ep->K || !ep->order || !ep->weights) return 0;
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(G.phys, &props);
        VkPhysicalDeviceLimits *lim = &props.limits;
        /* Shader indexing is signed 32-bit; descriptor slices must be aligned.
         * Refuse oversized prompts before allocation, allowing CPU fallback. */
        if (D < 1 || I < 1 || total > INT_MAX/D || total > INT_MAX/I || ep->S > INT_MAX/D ||
            (size_t)total*D*4 > lim->maxStorageBufferRange ||
            (size_t)total*I*4 > lim->maxStorageBufferRange ||
            (size_t)ep->S*D*4 > lim->maxStorageBufferRange ||
            (size_t)ep->S*ep->K*4 > lim->maxStorageBufferRange ||
            (D*4) % lim->minStorageBufferOffsetAlignment ||
            (I*4) % lim->minStorageBufferOffsetAlignment ||
            (uint32_t)total > lim->maxComputeWorkGroupCount[1] ||
            (uint32_t)ep->S > lim->maxComputeWorkGroupCount[1]) return 0;
        size_t pb = (size_t)ep->S*ep->K*4;
        if (!scratch_reserve(&G.ep_x, (size_t)ep->S*D*4) ||
            !scratch_reserve(&G.ep_order, (size_t)total*4) ||
            !scratch_reserve(&G.ep_inverse, pb) || !scratch_reserve(&G.ep_weights, pb) ||
            !scratch_reserve_mt(&G.ep_y, (size_t)ep->S*D*4, G.memtype_cached)) return 0;
        /* built in ordinary memory and copied once: the check reads inv[p] back,
         * and reads from the mapped (write-combined) scratch cost ~250 ns each --
         * 2.3 ms per layer at 9099 pairs */
        int *inv = malloc(pb);
        if (!inv) return 0;
        for (int p = 0; p < ep->S*ep->K; p++) inv[p] = -1;
        for (int r = 0; r < total; r++) {
            int p = ep->order[r];
            if (p < 0 || p >= ep->S*ep->K || inv[p] != -1) { free(inv); return 0; }
            inv[p] = r;
        }
        memcpy(G.ep_inverse.ptr, inv, pb);
        free(inv);
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    if (!scratch_reserve(&G.eg_x, xb) || !scratch_reserve(&G.eg_h, hb) ||
        !scratch_reserve_mt(&G.eg_y, yb, G.memtype_cached)) return 0;   /* eg_y is read back -> cached */
    G.eg_prof = getenv("VK_PROF") != NULL;
    if (G.eg_prof) { G.eg_t0 = vk_now(); G.eg_tin = t_in; }
    if (ep) {
        G.ep_xa = arena_buf(x, (size_t)ep->S*D*4); G.ep_ya = arena_buf(ep->y, (size_t)ep->S*D*4);
        if (!G.ep_xa) memcpy(G.ep_x.ptr, x, (size_t)ep->S*D*4);
        memcpy(G.ep_order.ptr, ep->order, (size_t)total*4);
        memcpy(G.ep_weights.ptr, ep->weights, (size_t)ep->S*ep->K*4);
    } else { G.ep_xa = G.ep_ya = VK_NULL_HANDLE; memcpy(G.eg_x.ptr, x, xb); }
    if (G.eg_prof) G.eg_t1 = vk_now();

    if (G.eg_nsets < count) {
        if (G.eg_pool) vkDestroyDescriptorPool(G.dev, G.eg_pool, NULL);
        G.eg_pool = VK_NULL_HANDLE; G.eg_nsets = 0;
        int n = count < 64 ? 64 : count;
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = n*10};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = n*2, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.eg_pool), "eg descPool");
        VkDescriptorSetLayout lg[EG_MAX_EXPERTS], ld[EG_MAX_EXPERTS];
        for (int c = 0; c < n; c++) { lg[c] = G.dsl_gu; ld[c] = G.dsl; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = n, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = n, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ag, G.eg_gu), "eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ad, G.eg_dn), "eg dn sets");
        G.eg_nsets = n;
    }
    if (ep) {
        VkDescriptorBufferInfo gi[3] = {{G.ep_xa ? G.ep_xa : G.ep_x.buf, 0, (size_t)ep->S*D*4},
            {G.ep_order.buf, 0, (size_t)total*4}, {G.eg_x.buf, 0, xb}};
        VkDescriptorBufferInfo ri[4] = {{G.eg_y.buf, 0, yb},
            {G.ep_inverse.buf, 0, (size_t)ep->S*ep->K*4},
            {G.ep_weights.buf, 0, (size_t)ep->S*ep->K*4}, {G.ep_ya ? G.ep_ya : G.ep_y.buf, 0, (size_t)ep->S*D*4}};
        wr_desc(G.ep_set[0], 3, gi); wr_desc(G.ep_set[1], 4, ri);
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        VkDescriptorBufferInfo gi[6] = {
            {G.eg_x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
            {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
            {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
        wr_desc(G.eg_gu[c], 6, gi);
        VkDescriptorBufferInfo di[4] = {
            {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc(G.eg_dn[c], 4, di);
    }
    if (G.eg_prof) G.eg_t2 = vk_now();

    VKCHECK(vkResetCommandBuffer(G.eg_cmd, 0), "eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.eg_cmd, &begin), "eg beginCmd");
    ts_begin(G.eg_cmd, TS_SLOT_EG); ts_stage0b(G.eg_cmd, TSB_EG);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    /* Grouped: every expert of the prefill batch in ONE dispatch per phase (expert
     * table with device addresses + item list), when all of them take the tiles. */
    int grp = ep && G.coop && G.pipe_grp[0] && G.pipe_grp[1] && I % 128 == 0 && D % 128 == 0 && D % 64 == 0 && I % 64 == 0;
    for (int c = 0; grp && c < count; c++) {
        int gf = gates[c]->fmt, df = downs[c]->fmt;
        if (!(gf == 1 || gf == 2 || gf == 4) || !(df == 1 || df == 2 || df == 4) ||
            (gf == 4 && (gates[c]->gs < 8 || gates[c]->gs % 8)) || (df == 4 && (downs[c]->gs < 8 || downs[c]->gs % 8))) grp = 0;
    }
    /* grouped gate_up reads the token rows through the order list itself
     * (COLI_VK_GRP_GATHER=1: separate gather pass as before) */
    int ggat = grp && !(getenv("COLI_VK_GRP_GATHER") && getenv("COLI_VK_GRP_GATHER")[0] == '1');
    if (ep && !ggat) {
        struct PCEP pc = {ep->S, D, ep->K, total};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.ep_pipe[0]);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.ep_layout[0], 0, 1, &G.ep_set[0], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.ep_layout[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.eg_cmd, (D+255)/256, total, 1);
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    ts_stageb(G.eg_cmd, TSB_EG, 0);
    if (grp && getenv("VK_DBG_FMT")) { static int once; if (!once++) fprintf(stderr, "[dbg] experts: count %d total %d gate fmt %d gs %d down fmt %d gs %d | last fmt %d\n", count, total, gates[0]->fmt, gates[0]->gs, downs[0]->fmt, downs[0]->gs, gates[count-1]->fmt); }
    if (grp) {
        /* tiles per expert: full ones (16*TT rows) while more than 32 rows remain,
         * the rest (<= 32 rows) on the TT = 2 twin when it exists -- an expert gets
         * ~32 rows on average, a 64-row tile alone ran half empty */
        /* measured (1011 tokens): down 124 -> 109 ms with the twin, gate_up 248 -> 298
         * (its tile dequantises two matrices, a 32-row tile does not pay for that)
         * -> COLI_VK_GRP_MIX: 1 = down only (default), 2 = both, 0 = off */
        static int mixm = -1;
        if (mixm < 0) { const char *e = getenv("COLI_VK_GRP_MIX"); mixm = e ? atoi(e) : 1; }
        int tm = 16 * G.grp_tt, nbig[2], nsm[2];
        for (int v = 0; v < 2; v++) {          /* v 0: down (out D), v 1: gate_up (out I) */
            int mix = G.pipe_grp_s[v] != VK_NULL_HANDLE && (mixm == 2 || (mixm == 1 && v == 0)), nb = 0, ns = 0, O = v ? I : D;
            for (int c = 0; c < count; c++) {
                int r = rows[c];
                while (r > (mix ? 32 : 0)) { nb++; r -= tm; }
                if (r > 0) ns++;
            }
            nbig[v] = nb * (O / 128); nsm[v] = ns * (O / 128); G.grp_mix[v] = mix;
            grp = grp && scratch_reserve(&G.grp_it[v], (size_t)(nbig[v] + nsm[v]) * 16) && scratch_reserve(&G.grp_et[v], (size_t)count * 64);
        }
        for (int v = 0; v < 2 && grp; v++) {   /* v 1: gate_up (out I), v 0: down (out D) */
            uint32_t *et = G.grp_et[v].ptr, *it = G.grp_it[v].ptr; int n = 0, m = nbig[v], O = v ? I : D, mix = G.grp_mix[v];
            for (int c = 0; c < count; c++) {
                ColiVkTensor *a = v ? gates[c] : downs[c], *b = v ? ups[c] : NULL;
                uint64_t wa = vk_addr(a->wbuf), sa = vk_addr(a->sbuf), wb = b ? vk_addr(b->wbuf) : 0, sb = b ? vk_addr(b->sbuf) : 0;
                uint32_t *e = et + (size_t)c * 16;
                e[0] = (uint32_t)wa; e[1] = (uint32_t)(wa >> 32); e[2] = (uint32_t)sa; e[3] = (uint32_t)(sa >> 32);
                e[4] = (uint32_t)wb; e[5] = (uint32_t)(wb >> 32); e[6] = (uint32_t)sb; e[7] = (uint32_t)(sb >> 32);
                e[8] = (uint32_t)rows[c]; e[9] = (uint32_t)off[c]; e[10] = (uint32_t)a->fmt; e[11] = (uint32_t)a->rowWords; e[12] = (uint32_t)a->gs;
                /* A-major: token tiles of one (expert, output block) adjacent; full
                 * tiles in [0, nbig), the small last tiles after them */
                for (int ob = 0; ob < O / 128; ob++) {
                    int t0 = 0;
                    for (; rows[c] - t0 > (mix ? 32 : 0); t0 += tm) { uint32_t *q = it + (size_t)n * 4; q[0] = c; q[1] = t0; q[2] = ob; q[3] = 0; n++; }
                    if (rows[c] - t0 > 0) { uint32_t *q = it + (size_t)m * 4; q[0] = c; q[1] = t0; q[2] = ob; q[3] = 0; m++; }
                }
            }
        }
        G.grp_n[0] = nbig[0]; G.grp_n[1] = nbig[1]; G.grp_ns[0] = nsm[0]; G.grp_ns[1] = nsm[1];
    }
    if (grp) {
        VkBuffer xin[2] = {G.eg_h.buf, ggat ? (G.ep_xa ? G.ep_xa : G.ep_x.buf) : G.eg_x.buf}, yout[2] = {G.eg_y.buf, G.eg_h.buf};
        for (int step = 0; step < 2; step++) {
            int v = step == 0 ? 1 : 0;       /* gate_up first, then down */
            VkDescriptorBufferInfo bi[5] = {{xin[v], 0, VK_WHOLE_SIZE}, {G.grp_it[v].buf, 0, VK_WHOLE_SIZE},
                                            {G.grp_et[v].buf, 0, VK_WHOLE_SIZE}, {yout[v], 0, VK_WHOLE_SIZE},
                                            {G.ep_order.buf, 0, VK_WHOLE_SIZE}};
            wr_desc(G.dset_grp[v], v ? 5 : 4, bi);
            struct PCGRP pc = {v ? D : I, v ? I : D, g_swiglu_limit, v && ggat ? ep->K : 0, 0};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_grp[v], 0, 1, &G.dset_grp[v], 0, NULL);
            if (G.grp_n[v]) {
                vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_grp[v]);
                vkCmdPushConstants(G.eg_cmd, G.plyt_grp[v], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(G.eg_cmd, (uint32_t)G.grp_n[v], 1, 1);
            }
            if (G.grp_ns[v]) {               /* disjoint rows: no barrier in between */
                pc.ibase = G.grp_n[v];
                vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_grp_s[v]);
                vkCmdPushConstants(G.eg_cmd, G.plyt_grp[v], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(G.eg_cmd, (uint32_t)G.grp_ns[v], 1, 1);
            }
            if (step == 0) vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            ts_stageb(G.eg_cmd, TSB_EG, 1 + step);
        }
    } else {
    /* phase 1: fused gate+up+silu -> hidden (per expert, bound to its x/hidden slices) */
    /* Experts with more than one routed row take the multi-row shader, the rest the
     * one-row shader; order within a phase is free (the dispatches are independent). */
    VkPipeline bound = VK_NULL_HANDLE;
    for (int c = 0; c < count; c++) {
        int gf = gates[c]->fmt;
        int co = coop_use(gf, rows[c], D, I, gates[c]->gs, 1), mr = !co && mr_use(gf, rows[c], D, I);
        uint32_t cgx = 0, cgy = 0;
        VkPipeline want = co ? coop_pick(1, rows[c], I, &cgx, &cgy) : mr ? G.pipe_gu_mr : G.pipe_gu;
        if (want != bound) { vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, want); bound = want; }
        struct PCGU pc = pcgu(gf, rows[c], D, I, gates[c]->rowWords, gates[c]->gs);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.eg_gu[c], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (co) vkCmdDispatch(G.eg_cmd, cgx, cgy, 1);
        else vkCmdDispatch(G.eg_cmd, (uint32_t)((I + 7) / 8), (uint32_t)(mr ? (rows[c] + G.mr - 1) / G.mr : rows[c]), 1);
    }
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* phase 2: down projection hidden -> y */
    bound = VK_NULL_HANDLE;
    for (int c = 0; c < count; c++) {
        int df = downs[c]->fmt;
        int co = coop_use(df, rows[c], I, D, downs[c]->gs, 1), mr = !co && mr_use(df, rows[c], I, D);
        uint32_t cgx = 0, cgy = 0;
        VkPipeline want = co ? coop_pick(0, rows[c], D, &cgx, &cgy) : mr ? G.pipe_mr : G.pipe;
        if (want != bound) { vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, want); bound = want; }
        struct PC pc = {df, rows[c], I, D, downs[c]->rowWords, downs[c]->gs};
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.eg_dn[c], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (co) vkCmdDispatch(G.eg_cmd, cgx, cgy, 1);
        else vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 7) / 8), (uint32_t)(mr ? (rows[c] + G.mr - 1) / G.mr : rows[c]), 1);
    }
    }   /* per-expert dispatches */
    if (ep) {
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        struct PCEP pc = {ep->S, D, ep->K, total};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.ep_pipe[1]);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.ep_layout[1], 0, 1, &G.ep_set[1], 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.ep_layout[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G.eg_cmd, (D+255)/256, ep->S, 1);
    }
    ts_stageb(G.eg_cmd, TSB_EG, 3);
    host_read_barrier(G.eg_cmd);
    ts_end(G.eg_cmd, TS_SLOT_EG);
    VKCHECK(vkEndCommandBuffer(G.eg_cmd), "eg endCmd");
    if (G.eg_prof) G.eg_t3 = vk_now();

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.eg_fence), "eg resetFence");
    { double vp0 = G.eg_prof ? vk_now() : 0;
      G.eg_th0 = vk_now();
      VKCHECK(vk_submit(G.queue, &si, G.eg_fence), "eg queueSubmit");
      if (G.eg_prof) g_vsub_ms += vk_now() - vp0; }
    G.eg_pending_yb = ep ? (size_t)ep->S*D*4 : yb; G.eg_reduced = ep != NULL; G.eg_inflight = 1;
    return 1;
}

/* Synchronous token I/O; all experts, gather and reduction share one submit. */
int coli_vk_expert_prefill(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           const int *order, const float *weights, int S, int K,
                           const float *x, float *y) {
    if (G.eg_inflight || !x || !y) return 0;
    ExpertPrefill ep = {S, K, order, weights, y};
    if (!eg_prepare_submit(gates, ups, downs, rows, count, x, &ep)) return 0;
    return coli_vk_expert_group_take(y);
}

/* Issue a group asynchronously: submit and return WITHOUT waiting, so the caller
 * computes its CPU share concurrently. Exactly one group may be in flight. */
int coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                               ColiVkTensor *const *downs, const int *rows, int count,
                               const float *x) {
    if (G.eg_inflight) return 0;
    return eg_prepare_submit(gates, ups, downs, rows, count, x, NULL);
}

/* Join the in-flight group and read back the packed outputs. */
int coli_vk_expert_group_take(float *y) {
    if (!G.eg_inflight) return 0;
    G.eg_inflight = 0;
    if (vk_fence_wait(G.dev, G.eg_fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] expert-group fence wait failed — disabling GPU offload\n");
        G.ready = 0; return 0;
    }
    ts_read(TS_SLOT_EG, TS_EG, vk_now() - G.eg_th0);
    if (G.eg_reduced) ts_stage_read(TSB_EG, 4);
    double t4 = G.eg_prof ? vk_now() : 0;
    if (!(G.eg_reduced && G.ep_ya)) memcpy(y, G.eg_reduced ? G.ep_y.ptr : G.eg_y.ptr, G.eg_pending_yb);
    if (G.eg_prof) {
        double t5 = vk_now();
        fprintf(stderr, "[VK_PROF] pre %.3f | memcpy_x %.3f | desc %.3f | record %.3f | issue->take %.3f | memcpy_y %.3f ms\n",
                G.eg_t0-G.eg_tin, G.eg_t1-G.eg_t0, G.eg_t2-G.eg_t1, G.eg_t3-G.eg_t2, t4-G.eg_t3, t5-t4);
    }
    return 1;
}

/* Synchronous form (shared expert, harness): issue + take in one call. */
int coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                         ColiVkTensor *const *downs, const int *rows, int count,
                         float *y, const float *x) {
    if (G.eg_inflight) return 0;
    if (!eg_prepare_submit(gates, ups, downs, rows, count, x, NULL)) return 0;
    return coli_vk_expert_group_take(y);
}

/* ==================== SECOND DEVICE: expert tier only (COLI_VK_DEV2) ============
 * A self-contained context for a second Vulkan GPU (e.g. an RX 580 beside the
 * RX 9070) that hosts ONLY resident tier experts and runs ONLY the async
 * expert-group path (fused gate_up -> down). Attention, dense, q-prep and the
 * KV mirror stay on device 0. Deliberately separate from G so the device-0 hot
 * path is untouched and both groups can be in flight simultaneously. The same
 * physical device as dev0 is allowed when forced by index (a second logical
 * device — the pre-hardware test mode); `auto` requires a distinct real GPU. */
static struct {
    int ready;
    VkPhysicalDevice phys; VkDevice dev; VkQueue queue; uint32_t qfam;
    uint32_t memtype, memtype_cached;
    VkShaderModule sh_qmm, sh_gu;
    VkDescriptorSetLayout dsl, dsl_gu; VkPipelineLayout plyt, plyt_gu;
    VkPipeline pipe, pipe_gu;
    VkCommandPool cpool; VkCommandBuffer cmd; VkFence fence;
    VkDescriptorPool pool; VkDescriptorSet gu[64], dn[64]; int nsets;
    Scratch x, h, y;
    int inflight; size_t pending_yb;
    VkWArena *arena;
    size_t used_bytes, tensor_count;
    int has_budget, has_portability;
} G2;

static int alloc_hostvis_d2(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G2.dev, &bi, NULL, buf), "d2 vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G2.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
    VKCHECK(vkAllocateMemory(G2.dev, &ai, NULL, mem), "d2 vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G2.dev, *buf, *mem, 0), "d2 vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G2.dev, *mem, 0, bytes, 0, ptr), "d2 vkMapMemory");
    return 1;
}
static int scratch_reserve_d2(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G2.dev, s->buf, NULL); vkFreeMemory(G2.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    if (!alloc_hostvis_d2(bytes, &s->buf, &s->mem, &s->ptr, memtype)) return 0;
    s->cap = bytes;
    return 1;
}
static int arena_suballoc_d2(size_t bytes, VkBuffer *buf, void **ptr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G2.dev, &bi, NULL, buf), "d2 vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G2.dev, *buf, &req);
    if (!(req.memoryTypeBits & (1u << G2.memtype))) { vkDestroyBuffer(G2.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
    size_t align = req.alignment ? req.alignment : 256, off = 0;
    VkWArena *a = G2.arena;
    for (; a; a = a->next) {
        off = (a->off + align - 1) & ~(align - 1);
        if (off + req.size <= a->cap) break;
    }
    if (!a) {
        size_t cap = req.size > VK_WARENA_BLOCK ? (req.size + 4095) & ~(size_t)4095 : VK_WARENA_BLOCK;
        a = calloc(1, sizeof(*a));
        if (!a) { vkDestroyBuffer(G2.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = cap, .memoryTypeIndex = G2.memtype};
        if (vkAllocateMemory(G2.dev, &ai, NULL, &a->mem) != VK_SUCCESS ||
            vkMapMemory(G2.dev, a->mem, 0, cap, 0, (void **)&a->base) != VK_SUCCESS) {
            if (a->mem) vkFreeMemory(G2.dev, a->mem, NULL);
            free(a); vkDestroyBuffer(G2.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
        }
        a->cap = cap; a->next = G2.arena; G2.arena = a;
        off = 0;
    }
    VKCHECK(vkBindBufferMemory(G2.dev, *buf, a->mem, off), "d2 vkBindBufferMemory");
    if (ptr) *ptr = a->base + off;
    a->off = off + req.size;
    return 1;
}
static int upload_tensor_d2(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 &&
        !(fmt == 4 && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    t->dev = 1;
    size_t stride = (size_t)t->rowWords * 4;
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;
    size_t sfl = scale_floats(fmt, I, O, gs);
    t->wbytes = stride * (size_t)O;
    void *wptr;
    if (!arena_suballoc_d2(t->wbytes, &t->wbuf, &wptr)) { free(t); return 0; }
    memset(wptr, 0, t->wbytes);
    for (int o = 0; o < O; o++)
        memcpy((uint8_t *)wptr + (size_t)o * stride,
               (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    void *sptr;
    if (!arena_suballoc_d2(sfl * sizeof(float), &t->sbuf, &sptr)) {
        vkDestroyBuffer(G2.dev, t->wbuf, NULL); free(t); return 0;
    }
    memcpy(sptr, scales, sfl * sizeof(float));
    __atomic_add_fetch(&G2.used_bytes, t->wbytes + sfl * sizeof(float), __ATOMIC_RELAXED);
    __atomic_add_fetch(&G2.tensor_count, 1, __ATOMIC_RELAXED);
    *out = t;
    return 1;
}

/* Bring up the second device. devidx: -1 = auto (best-ranked real GPU that is NOT
 * device 0; fails if none), >=0 = that enumeration index (same-physical-device
 * allowed with a warning — the pre-hardware test mode). Requires coli_vk_init. */
int coli_vk_init_dev2(const char *spv_path, int devidx) {
    if (G2.ready) return 1;
    if (!G.ready) return 0;
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    if (!nd) return 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    if (devidx >= 0) {
        if ((uint32_t)devidx >= nd) { fprintf(stderr, "[VK] dev2: index %d out of range (%u devices)\n", devidx, nd); return 0; }
        G2.phys = devs[devidx];
        if (G2.phys == G.phys)
            fprintf(stderr, "[VK] dev2: SAME physical device as dev0 — second logical device (test mode)\n");
    } else {
        int bestrank = -1;
        for (uint32_t i = 0; i < nd; i++) {
            if (devs[i] == G.phys) continue;
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
            int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                       p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 : -1;
            if (rank > bestrank) { bestrank = rank; G2.phys = devs[i]; }
        }
        if (bestrank < 0) { fprintf(stderr, "[VK] dev2=auto: no second real GPU found\n"); return 0; }
    }
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G2.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G2.phys, &nq, qf);
    G2.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G2.qfam = i; break; }
    if (G2.qfam == UINT32_MAX) { fprintf(stderr, "[VK] dev2: no compute queue\n"); return 0; }
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G2.qfam, .queueCount = 1, .pQueuePriorities = &qprio};
    const char *dext[2]; uint32_t ndext = 0;
#ifdef VK_EXT_memory_budget
    {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(G2.phys, NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) {
            vkEnumerateDeviceExtensionProperties(G2.phys, NULL, &ne, ep);
            for (uint32_t i = 0; i < ne; i++) {
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) G2.has_budget = 1;
                if (!strcmp(ep[i].extensionName, "VK_KHR_portability_subset")) G2.has_portability = 1;
            }
            free(ep);
        }
        if (G2.has_budget) dext[ndext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
        if (G2.has_portability) dext[ndext++] = "VK_KHR_portability_subset";   /* as on the first device */
    }
#endif
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = ndext, .ppEnabledExtensionNames = ndext ? dext : NULL};
    VKCHECK(vkCreateDevice(G2.phys, &di, NULL, &G2.dev), "d2 vkCreateDevice");
    vkGetDeviceQueue(G2.dev, G2.qfam, 0, &G2.queue);
    int mt = pick_memtype(G2.phys);
    if (mt < 0) { fprintf(stderr, "[VK] dev2: no host-visible memory\n"); return 0; }
    G2.memtype = (uint32_t)mt;
    G2.memtype_cached = (uint32_t)pick_memtype_cached(G2.phys);
    G2.sh_qmm = load_spv(G2.dev, spv_path);
    if (!G2.sh_qmm) return 0;
    char gu_path[512]; derive_sibling(spv_path, "_gate_up.spv", gu_path, sizeof(gu_path));
    G2.sh_gu = load_spv(G2.dev, gu_path);
    if (!G2.sh_gu) { fprintf(stderr, "[VK] dev2: gate_up shader required for the tier\n"); return 0; }
    VkDescriptorPool dp; VkDescriptorSet ds;   /* build_pipeline's singleton set: unused here */
    if (!build_pipeline(G2.dev, 4, sizeof(struct PC), G2.sh_qmm, &G2.dsl, &G2.plyt, &G2.pipe, &dp, &ds)) return 0;
    if (!build_pipeline(G2.dev, 6, sizeof(struct PCGU), G2.sh_gu, &G2.dsl_gu, &G2.plyt_gu, &G2.pipe_gu, &dp, &ds)) return 0;
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G2.qfam};
    VKCHECK(vkCreateCommandPool(G2.dev, &cpci, NULL, &G2.cpool), "d2 cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G2.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G2.dev, &cbi, &G2.cmd), "d2 cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G2.dev, &fi, NULL, &G2.fence), "d2 fence");
    G2.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G2.phys, &p);
    fprintf(stderr, "[VK] dev2 ready: %s (expert tier only), compute qfam %u, memtype %u\n",
            p.deviceName, G2.qfam, G2.memtype);
    return 1;
}

int coli_vk_dev2_available(void) { return G2.ready; }
int coli_vk_tensor_dev(const ColiVkTensor *t) { return t ? t->dev : 0; }

int coli_vk_mem_budget2(double *used_gb, double *budget_gb) {
#ifdef VK_EXT_memory_budget
    if (!G2.has_budget || !G2.phys) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
    vkGetPhysicalDeviceMemoryProperties2(G2.phys, &mp2);
    double u = 0, b = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++)
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += (double)bud.heapUsage[i]; b += (double)bud.heapBudget[i];
        }
    if (used_gb) *used_gb = u / 1e9;
    if (budget_gb) *budget_gb = b / 1e9;
    return b > 0;
#else
    (void)used_gb; (void)budget_gb; return 0;
#endif
}

int coli_vk_tensor_ensure2(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp) {
    if (!G2.ready) return 0;
    return upload_tensor_d2(tensor, weights, scales, fmt, I, O, grp);
}

/* dev2 mirror of eg_prepare_submit: identical structure on G2's pipelines/scratches. */
static int eg2_prepare_submit(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                              ColiVkTensor *const *downs, const int *rows, int count,
                              const float *x) {
    if (!G2.ready || count < 1 || count > 64) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int D = g0->I, I = g0->O, fmt = g0->fmt, total = 0, off[64];
    if (D > 6144) return 0;
    int dfmt = downs[0]->fmt;
    for (int c = 0; c < count; c++) {
        off[c] = total; total += rows[c];
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != fmt ||
            downs[c]->I != I || downs[c]->O != D || downs[c]->fmt != dfmt) return 0;
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    /* VK_PROF=1: phase split of the dev2 issue cost (same scheme as the dense path) —
     * localizes the per-block tax between our copy, descriptors, recording and the
     * driver's submit on the chipset-x4 Polaris path. */
    static double q_x, q_desc, q_rec, q_sub; static long q_n;
    double t0 = G.eg_prof ? vk_now() : 0, tA;
    if (!scratch_reserve_d2(&G2.x, xb, G2.memtype) || !scratch_reserve_d2(&G2.h, hb, G2.memtype) ||
        !scratch_reserve_d2(&G2.y, yb, G2.memtype_cached)) return 0;
    memcpy(G2.x.ptr, x, xb);
    if (G.eg_prof) { tA = vk_now(); q_x += tA - t0; t0 = tA; }
    if (!G2.pool) {
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64*6 + 64*4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 128, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G2.dev, &dpi, NULL, &G2.pool), "d2 eg descPool");
        VkDescriptorSetLayout lg[64], ld[64];
        for (int c = 0; c < 64; c++) { lg[c] = G2.dsl_gu; ld[c] = G2.dsl; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G2.pool, .descriptorSetCount = 64, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G2.pool, .descriptorSetCount = 64, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G2.dev, &ag, G2.gu), "d2 eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G2.dev, &ad, G2.dn), "d2 eg dn sets");
        G2.nsets = 64;
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        VkDescriptorBufferInfo gi[6] = {
            {G2.x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
            {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
            {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G2.h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
        wr_desc_dev(G2.dev, G2.gu[c], 6, gi);
        VkDescriptorBufferInfo di[4] = {
            {G2.h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G2.y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc_dev(G2.dev, G2.dn[c], 4, di);
    }
    if (G.eg_prof) { tA = vk_now(); q_desc += tA - t0; t0 = tA; }
    VKCHECK(vkResetCommandBuffer(G2.cmd, 0), "d2 eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G2.cmd, &begin), "d2 eg beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.pipe_gu);
    for (int c = 0; c < count; c++) {
        struct PCGU pc = pcgu(fmt, rows[c], D, I, gates[c]->rowWords, gates[c]->gs);
        vkCmdBindDescriptorSets(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.plyt_gu, 0, 1, &G2.gu[c], 0, NULL);
        vkCmdPushConstants(G2.cmd, G2.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G2.cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
    }
    vkCmdPipelineBarrier(G2.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.pipe);
    for (int c = 0; c < count; c++) {
        struct PC pc = {dfmt, rows[c], I, D, downs[c]->rowWords, downs[c]->gs};
        vkCmdBindDescriptorSets(G2.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G2.plyt, 0, 1, &G2.dn[c], 0, NULL);
        vkCmdPushConstants(G2.cmd, G2.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(G2.cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
    }
    host_read_barrier(G2.cmd);
    VKCHECK(vkEndCommandBuffer(G2.cmd), "d2 eg endCmd");
    if (G.eg_prof) { tA = vk_now(); q_rec += tA - t0; t0 = tA; }
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G2.cmd};
    VKCHECK(vkResetFences(G2.dev, 1, &G2.fence), "d2 eg resetFence");
    VKCHECK(vkQueueSubmit(G2.queue, 1, &si, G2.fence), "d2 eg queueSubmit");
    if (G.eg_prof) { tA = vk_now(); q_sub += tA - t0;
        if ((++q_n & 2047) == 0)
            fprintf(stderr, "[VK_PROF d2iss] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f ms\n",
                    q_n, q_x, q_desc, q_rec, q_sub);
    }
    G2.pending_yb = yb; G2.inflight = 1;
    return 1;
}

int coli_vk_expert_group_issue2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x) {
    if (G2.inflight) return 0;
    return eg2_prepare_submit(gates, ups, downs, rows, count, x);
}
int coli_vk_expert_group_take2(float *y) {
    if (!G2.inflight) return 0;
    G2.inflight = 0;
    if (vk_fence_wait(G2.dev, G2.fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] dev2 expert-group fence wait failed — disabling dev2 offload\n");
        G2.ready = 0; return 0;
    }
    memcpy(y, G2.y.ptr, G2.pending_yb);
    return 1;
}
int coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x) {
    if (G2.inflight) return 0;
    if (!eg2_prepare_submit(gates, ups, downs, rows, count, x)) return 0;
    return coli_vk_expert_group_take2(y);
}

/* ---- MLA absorb attention core -------------------------------------------------
 * The KV latent (L, [rows,K]) and rope (R, [rows,Rd]) caches live in persistent
 * per-layer device buffers, appended row-by-row as tokens decode (the host stays
 * canonical; glm.c tracks a valid-watermark and re-appends after invalidation).
 * Rows are indexed by ABSOLUTE position, so kv_start windows just skip rows. */

int coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || max_rows < 1 || K < 1 || Rd < 1) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (v->bl) return v->rows >= max_rows && v->K == K && v->R == Rd;  /* resize goes through coli_vk_kv_reset */
    float p0 = G.prio; G.prio = 1.0f;            /* KV mirror rides every attention submit */
    int ok1 = alloc_hostvis((size_t)max_rows * K * 4, &v->bl, &v->ml, &v->pl);
    int ok = ok1 && alloc_hostvis((size_t)max_rows * Rd * 4, &v->br, &v->mr, &v->pr);
    G.prio = p0;
    if (!ok) {
        if (ok1) { vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL); }
        memset(v, 0, sizeof(*v)); return 0;
    }
    v->rows = max_rows; v->K = K; v->R = Rd;
    return 1;
}

/* Mirror one host cache row into the device copy (write-combined memory: the CPU
 * only ever WRITES these buffers, the GPU reads them). */
int coli_vk_kv_row(int layer, int pos, const float *L, const float *R) {
    if (layer < 0 || layer >= VK_KV_LAYERS) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (!v->pl || pos < 0 || pos >= v->rows) return 0;
    memcpy((float *)v->pl + (size_t)pos * v->K, L, (size_t)v->K * 4);
    memcpy((float *)v->pr + (size_t)pos * v->R, R, (size_t)v->R * 4);
    return 1;
}

/* Drop all per-layer KV device caches (cache resize in kv_alloc). */
void coli_vk_kv_reset(void) {
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        VkKvLayer *v = &G.kv[i];
        if (!v->bl) continue;
        if (G.ready) {   /* dead device: leak GPU handles like coli_vk_tensor_free */
            vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL);
            vkDestroyBuffer(G.dev, v->br, NULL); vkFreeMemory(G.dev, v->mr, NULL);
        }
        memset(v, 0, sizeof(*v));
    }
}

/* Decode MLA absorption core for S causal query rows of one sequence, one submit:
 * ctx[s,h,:] = softmax((Wnope_h^T q_nope).L_t + q_rope.R_t) weighted latent context
 * projected through the value rows of kv_b. kv_b ([H*(Q+V), K]) uploads on first
 * call and stays resident; L/R rows [st0, T) must already be mirrored via
 * coli_vk_kv_row. Returns 0 -> caller falls back to CPU. */
int coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                             float *ctx, const float *q, int layer, int S, int H,
                             int Q, int R, int V, int K, int st0, int T, float scale) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0) return 0;  /* shared-array limits */
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    ColiVkTensor *t = *kvb;
    int cap = T - st0;
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    size_t sb = (size_t)S * H * cap * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve_mt(&G.y, cb, G.memtype_cached) ||
        !scratch_reserve(&G.att_sc, sb)) return 0;    /* y (ctx) is read back -> cached */
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_sc.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wd[7];
    for (int i = 0; i < 7; i++) wd[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_att,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 7, wd, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_att);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_att, 0, 1, &G.dset_att, 0, NULL);
    struct PCAttn pc = {fmt, S, H, Q, R, V, K, st0, T, t->rowWords, cap, scale, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt_att, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);     /* one workgroup per (head, row) */
    host_read_barrier(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(ctx, G.y.ptr, cb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Two resident matmuls sharing the SAME input x in ONE submit (q_a + kv_a in the
 * attention prologue): one x staging, two dispatches, one fence — replaces two
 * full submit+wait roundtrips. Outputs y1 [S,O1] and y2 [S,O2] read back from
 * cached memory. Returns 0 -> caller falls back to the single-matmul path. */
int coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                        ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                        int fmt, const float *x, int S, int I, int grp) {
    if (!G.ready || S < 1) return 0;
    if (!upload_tensor(t1p, w1, s1, fmt, I, O1, grp) || !upload_tensor(t2p, w2, s2, fmt, I, O2, grp)) return 0;
    ColiVkTensor *t1 = *t1p, *t2 = *t2p;
    size_t xb = (size_t)S * I * 4, yb1 = (size_t)S * O1 * 4, yb2 = (size_t)S * O2 * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, yb1, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, yb2, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);

    if (!G.pair_pool) {   /* one-time: a second 4-binding set (G.dset serves the first) */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t1->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t1->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t2->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t2->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, b1);
    wr_desc(G.dset_pair, 4, b2);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, O1, t1->rowWords, t1->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((O1 + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, O2, t2->rowWords, t2->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((O2 + 7) / 8), (uint32_t)S, 1);
    host_read_barrier(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(y1, G.y.ptr, yb1);
    memcpy(y2, G.y2.ptr, yb2);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}


/* q-prep chain: [q_a + kv_a pair] -> rmsnorm(q_latent) -> [q_b], recorded in ONE
 * command buffer with compute barriers — one submit+fence where the engine paid
 * three (the middle CPU norm forced two roundtrips). Only q [S,Oqb] and the kv
 * latent [S,Okva] return to the host (RoPE + canonical KV append stay CPU-side).
 * The per-layer norm weights upload once into a tiny resident buffer (KV-mirror
 * pattern). All three tensors must share fmt (dense io is int8 in practice).
 * Returns 0 -> caller runs the 3-submit path (also when rmsnorm.spv is absent). */
int coli_vk_attn_qprep(int layer,
                       ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                       ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                       ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                       int fmt, int grp, const float *lnw, float eps,
                       const float *x, int S, int I, float *q_out, float *kv_out, float *lat_out) {
    if (!G.ready || !G.shader_nrm || S < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (!upload_tensor(qa, wqa, sqa, fmt, I, Oqa, grp) || !upload_tensor(kva, wkva, skva, fmt, I, Okva, grp) ||
        !upload_tensor(qb, wqb, sqb, fmt, Oqa, Oqb, grp)) return 0;
    ColiVkTensor *tqa = *qa, *tkv = *kva, *tqb = *qb;
    if (!G.lnbuf[layer]) {                       /* resident norm weights, uploaded once */
        void *lp; float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis((size_t)Oqa * 4, &G.lnbuf[layer], &G.lnmem[layer], &lp);
        G.prio = p0;
        if (!ok) { G.lnbuf[layer] = VK_NULL_HANDLE; return 0; }
        memcpy(lp, lnw, (size_t)Oqa * 4); G.lnlen[layer] = Oqa;
    }
    if (G.lnlen[layer] != Oqa) return 0;
    size_t xb = (size_t)S * I * 4, qb_b = (size_t)S * Oqb * 4, kvb_b = (size_t)S * Okva * 4;
    size_t lat = (size_t)S * Oqa * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, qb_b, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, kvb_b, G.memtype_cached) ||
        !scratch_reserve(&G.qp1, lat) ||
        !scratch_reserve_mt(&G.qp2, lat, G.memtype_cached)) return 0;   /* normed latent reads back (DSA indexer) */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqa->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqa->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tkv->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tkv->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo bn[3] = {
        {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.lnbuf[layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b3[4] = {
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqb->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqb->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    if (!G.pair_pool) {   /* the chain reuses the pair's 2nd matmul set */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    wr_desc(G.dset, 4, b1); wr_desc(G.dset_pair, 4, b2); wr_desc(G.dset_qp3, 4, b3);
    wr_desc(G.dset_nrm, 3, bn);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, Oqa, tqa->rowWords, tqa->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqa + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, Okva, tkv->rowWords, tkv->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((Okva + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrm);
    struct PCN pcn = {S, Oqa, eps};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcn), &pcn);
    vkCmdDispatch(G.cmd, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc3 = {fmt, S, Oqa, Oqb, tqb->rowWords, tqb->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_qp3, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqb + 7) / 8), (uint32_t)S, 1);
    host_read_barrier(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(q_out, G.y.ptr, qb_b);
    memcpy(kv_out, G.y2.ptr, kvb_b);
    if (lat_out) memcpy(lat_out, G.qp2.ptr, lat);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Fused absorb attention + o-projection in ONE submit: the absorb kernel writes ctx
 * [S,H*V] to a device-only scratch, a barrier, then the resident o_proj ([Dout, H*V])
 * runs on it via the plain matmul pipeline — only out [S,Dout] returns to the host.
 * Kills the per-layer ctx readback + re-upload + second submit of the unfused path.
 * Returns 0 -> caller falls back (plain absorb or CPU). */
int coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                                     ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                                     float *out, const float *q, int layer, int S, int H,
                                     int Q, int R, int V, int K, int st0, int T, float scale,
                                     int Dout) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0 || Dout < 1) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    if (!upload_tensor(ot, ow, osc, ofmt, H * V, Dout, ogrp)) return 0;
    ColiVkTensor *t = *kvb, *to = *ot;
    int cap = T - st0;
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    size_t sb = (size_t)S * H * cap * 4, ob = (size_t)S * Dout * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve(&G.att_sc, sb) || !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_sc.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wd[7];
    for (int i = 0; i < 7; i++) wd[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_att,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 7, wd, 0, NULL);
    VkDescriptorBufferInfo oi[4] = {
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}, {.buffer = to->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = to->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, oi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_att);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_att, 0, 1, &G.dset_att, 0, NULL);
    struct PCAttn pc = {fmt, S, H, Q, R, V, K, st0, T, t->rowWords, cap, scale, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt_att, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC opc = {ofmt, S, H * V, Dout, to->rowWords, to->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    host_read_barrier(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vk_submit(G.queue, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait(G.dev, G.fence) != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    if (t->dev == 1) {   /* dev2 tensor: destroy on ITS device, count in ITS counters */
        if (G2.ready) {
            if (t->wbuf) { vkDestroyBuffer(G2.dev, t->wbuf, NULL); vkFreeMemory(G2.dev, t->wmem, NULL); }
            if (t->sbuf) { vkDestroyBuffer(G2.dev, t->sbuf, NULL); vkFreeMemory(G2.dev, t->smem, NULL); }
        }
        __atomic_sub_fetch(&G2.tensor_count, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&G2.used_bytes, t->wbytes + scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float), __ATOMIC_RELAXED);
        free(t);
        return;
    }
    if (G.bound_tensor == t) { G.bound_tensor = NULL; G.cmd_ready = 0; }  /* drop stale cache */
    // If the device was lost/disabled (the fence-timeout path sets G.ready=0), a submission
    // may still reference these buffers — do NOT vkDestroy into a dead device (GPU-side UAF).
    // Leak the GPU handles (we're degrading to CPU for the rest of the run) and reclaim the
    // host struct + counters only.
    if (G.ready) {
        if (t->wbuf) { vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); }
        if (t->sbuf) { vkDestroyBuffer(G.dev, t->sbuf, NULL); vkFreeMemory(G.dev, t->smem, NULL); }
    }
    // Mirror upload_tensor exactly (weights + scales), atomically — otherwise
    // used_bytes leaks the scales buffer on every free and drifts upward.
    __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&G.used_bytes, t->wbytes + scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float), __ATOMIC_RELAXED);
    free(t);
}

size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? t->wbytes : 0; }

/* Is the device we picked an integrated GPU? The backend already computes this
 * at device selection (see the deviceType ranking above) but only uses it to
 * rank candidates. The RAM planner needs it too: on an integrated GPU our
 * HOST_VISIBLE|DEVICE_LOCAL allocations are the same physical memory the host
 * expert cache draws from, so whoever sizes that cache has to know. Mirrors
 * coli_cuda_device_integrated() (#653) for the Vulkan path. */
int coli_vk_device_integrated(void) {
    if (!G.phys) return 0;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(G.phys, &p);
    return p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
}

void coli_vk_shutdown(void) {
    ts_report();
    if (G.tsq) { vkDestroyQueryPool(G.dev, G.tsq, NULL); G.tsq = VK_NULL_HANDLE; G.ts_on = 0; }
    if (!G.ready) return;
    vkDeviceWaitIdle(G.dev);
    if (G.x.buf) { vkDestroyBuffer(G.dev, G.x.buf, NULL); vkFreeMemory(G.dev, G.x.mem, NULL); }
    if (G.y.buf) { vkDestroyBuffer(G.dev, G.y.buf, NULL); vkFreeMemory(G.dev, G.y.mem, NULL); }
    if (G.h.buf) { vkDestroyBuffer(G.dev, G.h.buf, NULL); vkFreeMemory(G.dev, G.h.mem, NULL); }
    if (G.eg_x.buf) { vkDestroyBuffer(G.dev, G.eg_x.buf, NULL); vkFreeMemory(G.dev, G.eg_x.mem, NULL); }
    for (int i = 0; i < ARENA_N; i++)
        if (g_arena[i].buf) { vkDestroyBuffer(G.dev, g_arena[i].buf, NULL); vkFreeMemory(G.dev, g_arena[i].mem, NULL); g_arena[i] = (Scratch){0}; }
    if (G.eg_h.buf) { vkDestroyBuffer(G.dev, G.eg_h.buf, NULL); vkFreeMemory(G.dev, G.eg_h.mem, NULL); }
    if (G.eg_y.buf) { vkDestroyBuffer(G.dev, G.eg_y.buf, NULL); vkFreeMemory(G.dev, G.eg_y.mem, NULL); }
    if (G.att_sc.buf) { vkDestroyBuffer(G.dev, G.att_sc.buf, NULL); vkFreeMemory(G.dev, G.att_sc.mem, NULL); }
    if (G.att_ctx.buf) { vkDestroyBuffer(G.dev, G.att_ctx.buf, NULL); vkFreeMemory(G.dev, G.att_ctx.mem, NULL); }
    if (G.y2.buf) { vkDestroyBuffer(G.dev, G.y2.buf, NULL); vkFreeMemory(G.dev, G.y2.mem, NULL); }
    if (G.stage.buf) { vkDestroyBuffer(G.dev, G.stage.buf, NULL); vkFreeMemory(G.dev, G.stage.mem, NULL); }
    if (G.up_fence) vkDestroyFence(G.dev, G.up_fence, NULL);
    if (G.up_cpool) vkDestroyCommandPool(G.dev, G.up_cpool, NULL);
    if (G.qp1.buf) { vkDestroyBuffer(G.dev, G.qp1.buf, NULL); vkFreeMemory(G.dev, G.qp1.mem, NULL); }
    if (G.qp2.buf) { vkDestroyBuffer(G.dev, G.qp2.buf, NULL); vkFreeMemory(G.dev, G.qp2.mem, NULL); }
    for (int l = 0; l < VK_KV_LAYERS; l++)   /* per-layer resident norm weights (attn_qprep) */
        if (G.lnbuf[l]) { vkDestroyBuffer(G.dev, G.lnbuf[l], NULL); vkFreeMemory(G.dev, G.lnmem[l], NULL); }
    if (G.pair_pool) vkDestroyDescriptorPool(G.dev, G.pair_pool, NULL);
    coli_vk_kv_reset();
    Scratch *ep_scratch[] = {&G.ep_x, &G.ep_order, &G.ep_inverse, &G.ep_weights, &G.ep_y};
    for (int i = 0; i < 5; i++) if (ep_scratch[i]->buf) {
        vkDestroyBuffer(G.dev, ep_scratch[i]->buf, NULL); vkFreeMemory(G.dev, ep_scratch[i]->mem, NULL);
    }
    for (int i = 0; i < 2; i++) {
        if (G.ep_pool[i]) vkDestroyDescriptorPool(G.dev, G.ep_pool[i], NULL);
        if (G.ep_pipe[i]) vkDestroyPipeline(G.dev, G.ep_pipe[i], NULL);
        if (G.ep_layout[i]) vkDestroyPipelineLayout(G.dev, G.ep_layout[i], NULL);
        if (G.ep_dsl[i]) vkDestroyDescriptorSetLayout(G.dev, G.ep_dsl[i], NULL);
        if (G.ep_shader[i]) vkDestroyShaderModule(G.dev, G.ep_shader[i], NULL);
    }
    if (G.eg_pool) vkDestroyDescriptorPool(G.dev, G.eg_pool, NULL);
    vkDestroyFence(G.dev, G.fence, NULL);
    vkDestroyFence(G.dev, G.eg_fence, NULL);
    vkDestroyCommandPool(G.dev, G.cpool, NULL);
    vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    vkDestroyPipeline(G.dev, G.pipe, NULL);
    vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    vkDestroyShaderModule(G.dev, G.shader, NULL);
    for (int v = 0; v < 2; v++) {
        if (G.pipe_co2[v]) vkDestroyPipeline(G.dev, G.pipe_co2[v], NULL);
        if (G.pipe_gu_co2[v]) vkDestroyPipeline(G.dev, G.pipe_gu_co2[v], NULL);
    }
    if (G.shader_co2) vkDestroyShaderModule(G.dev, G.shader_co2, NULL);
    if (G.shader_gu_co2) vkDestroyShaderModule(G.dev, G.shader_gu_co2, NULL);
    if (G.db_pool) vkDestroyDescriptorPool(G.dev, G.db_pool, NULL);
    if (G.ab_pool) vkDestroyDescriptorPool(G.dev, G.ab_pool, NULL);
    { VkPipeline pl[2] = {G.pipe_ab, G.pipe_ag}; VkPipelineLayout ly[2] = {G.plyt_ab, G.plyt_ag};
      VkDescriptorSetLayout dl[2] = {G.dsl_ab, G.dsl_ag}; VkDescriptorPool dp[2] = {G.dpool_ab, G.dpool_ag};
      VkShaderModule sm[2] = {G.shader_ab, G.shader_ag};
      for (int i = 0; i < 2; i++) {
          if (pl[i]) vkDestroyPipeline(G.dev, pl[i], NULL);
          if (ly[i]) vkDestroyPipelineLayout(G.dev, ly[i], NULL);
          if (dl[i]) vkDestroyDescriptorSetLayout(G.dev, dl[i], NULL);
          if (dp[i]) vkDestroyDescriptorPool(G.dev, dp[i], NULL);
          if (sm[i]) vkDestroyShaderModule(G.dev, sm[i], NULL);
      } }
    { VkPipeline pl[3] = {G.pipe_dc, G.pipe_dp, G.pipe_dg}; VkPipelineLayout ly[3] = {G.plyt_dc, G.plyt_dp, G.plyt_dg};
      VkDescriptorSetLayout dl[3] = {G.dsl_dc, G.dsl_dp, G.dsl_dg}; VkDescriptorPool dp[3] = {G.dpool_dc, G.dpool_dp, G.dpool_dg};
      VkShaderModule sm[3] = {G.shader_dc, G.shader_dp, G.shader_dg};
      for (int i = 0; i < 3; i++) {
          if (pl[i]) vkDestroyPipeline(G.dev, pl[i], NULL);
          if (ly[i]) vkDestroyPipelineLayout(G.dev, ly[i], NULL);
          if (dl[i]) vkDestroyDescriptorSetLayout(G.dev, dl[i], NULL);
          if (dp[i]) vkDestroyDescriptorPool(G.dev, dp[i], NULL);
          if (sm[i]) vkDestroyShaderModule(G.dev, sm[i], NULL);
      } }
    if (G.pipe_dr) {
        vkDestroyDescriptorPool(G.dev, G.dpool_dr, NULL);
        vkDestroyPipeline(G.dev, G.pipe_dr, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_dr, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_dr, NULL);
        vkDestroyShaderModule(G.dev, G.shader_dr, NULL);
    }
    if (G.pipe_af) vkDestroyPipeline(G.dev, G.pipe_af, NULL);
    if (G.shader_af) vkDestroyShaderModule(G.dev, G.shader_af, NULL);
    if (G.pipe_ap) {
        vkDestroyDescriptorPool(G.dev, G.dpool_ap, NULL);
        vkDestroyPipeline(G.dev, G.pipe_ap, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_ap, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_ap, NULL);
        vkDestroyShaderModule(G.dev, G.shader_ap, NULL);
    }
    if (G.pipe_co) vkDestroyPipeline(G.dev, G.pipe_co, NULL);
    if (G.pipe_gu_co) vkDestroyPipeline(G.dev, G.pipe_gu_co, NULL);
    if (G.shader_co) vkDestroyShaderModule(G.dev, G.shader_co, NULL);
    if (G.shader_gu_co) vkDestroyShaderModule(G.dev, G.shader_gu_co, NULL);
    if (G.pipe_mr) vkDestroyPipeline(G.dev, G.pipe_mr, NULL);
    if (G.pipe_gu_mr) vkDestroyPipeline(G.dev, G.pipe_gu_mr, NULL);
    if (G.shader_mr) vkDestroyShaderModule(G.dev, G.shader_mr, NULL);
    if (G.shader_gu_mr) vkDestroyShaderModule(G.dev, G.shader_gu_mr, NULL);
    if (G.shader_gu) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gu, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gu, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gu, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gu, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gu, NULL);
    }
    if (G.shader_att) {
        vkDestroyDescriptorPool(G.dev, G.dpool_att, NULL);
        vkDestroyPipeline(G.dev, G.pipe_att, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_att, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_att, NULL);
        vkDestroyShaderModule(G.dev, G.shader_att, NULL);
    }
    if (G.pipe_nrm) {   /* the rmsnorm chain: its pool used to be a dropped local */
        vkDestroyPipeline(G.dev, G.pipe_nrm, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_nrm, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_nrm, NULL);
        vkDestroyShaderModule(G.dev, G.shader_nrm, NULL);
        vkDestroyDescriptorPool(G.dev, G.dpool_nrm, NULL);
        vkDestroyDescriptorPool(G.dev, G.qprep_pool, NULL);
    }
    for (VkWArena *a = g_warena; a;) {   /* weight arenas: unmapped/freed with the device */
        VkWArena *nx = a->next;
        vkUnmapMemory(G.dev, a->mem); vkFreeMemory(G.dev, a->mem, NULL);
        free(a); a = nx;
    }
    for (VkWArena *a = g_warena_dl; a;) {   /* the staged device-local chain is never mapped */
        VkWArena *nx = a->next;
        vkFreeMemory(G.dev, a->mem, NULL);
        free(a); a = nx;
    }
    g_warena_dl = NULL;
    g_warena = NULL;
    vkDestroyDevice(G.dev, NULL);
    vkDestroyInstance(G.inst, NULL);
    memset(&G, 0, sizeof(G));
}

#ifdef VK_TEST
// ---- standalone GPU-vs-CPU validation + microbench --------------------------
#include <math.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; }

static int g_ref_gs = 64;   /* fmt=4 group size the harness cases use */
static size_t ref_rowbytes(int fmt, int I) {
    return fmt == 1 ? (size_t)I : fmt == 5 ? (size_t)((I + 63) / 64) * 24 : (size_t)(I + 1) / 2;
}
static size_t ref_scales(int fmt, int I, int O) {   // scale COUNT (per-group for fmt 4/5)
    if (fmt == 5) return (size_t)O * (size_t)((I + 63) / 64);
    if (fmt == 4) return (size_t)O * (size_t)((I + g_ref_gs - 1) / g_ref_gs);
    return (size_t)O;
}
static float deq(const uint8_t *row, int fmt, int i) {
    if (fmt == 1) { int b = ((const int8_t *)row)[i]; return (float)b; }
    if (fmt == 5) {   // int3-g64: 16B low plane (2 bits) + 8B high plane (1 bit), v+4
        const uint8_t *lo = row + (size_t)(i >> 6) * 24, *hi = lo + 16; int j = i & 63;
        unsigned u = ((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2);
        return (float)((int)u - 4); }
    uint8_t v = row[i >> 1]; int nib = (i & 1) ? (v >> 4) : (v & 15); return (float)(nib - 8);
}

static void cpu_ref(float *y, const float *x, const uint8_t *w, const float *sc,
                    int fmt, int S, int I, int O) {
    size_t rb = ref_rowbytes(fmt, I);
    int gw2 = fmt == 4 ? g_ref_gs : 64, ng = (I + gw2 - 1) / gw2;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double sum = 0; const uint8_t *row = w + (size_t)o * rb;
        if (fmt == 5 || fmt == 4) {   // per-group scales fold inside the sum
            for (int g = 0; g < ng; g++) {
                double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
                for (int i = g * gw2; i < end; i++) a += x[s * I + i] * deq(row, fmt, i);
                sum += a * sc[(size_t)o * ng + g];
            }
            y[s * O + o] = (float)sum;
        } else {
            for (int i = 0; i < I; i++) sum += x[s * I + i] * deq(row, fmt, i);
            y[s * O + o] = (float)(sum * sc[o]);
        }
    }
}

/* dequant dot of one weight row against x with that row's scales applied —
 * per-row for fmt 1/2, per 64-group for fmt=5. scb = tensor scale array, o = row. */
static double ref_dot(const float *x, const uint8_t *row, const float *scb, int o, int fmt, int I) {
    if (fmt == 5 || fmt == 4) {
        int gw2 = fmt == 4 ? g_ref_gs : 64;
        int ng = (I + gw2 - 1) / gw2; double sum = 0;
        for (int g = 0; g < ng; g++) {
            double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
            for (int i = g * gw2; i < end; i++) a += x[i] * deq(row, fmt, i);
            sum += a * scb[(size_t)o * ng + g];
        }
        return sum;
    }
    double sum = 0;
    for (int i = 0; i < I; i++) sum += x[i] * deq(row, fmt, i);
    return sum * scb[o];
}

static int run_case(int fmt, int S, int I, int O, int iters) {
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O);
    float *x = malloc((size_t)S * I * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *sc = malloc(nsc * sizeof(float));
    float *yg = malloc((size_t)S * O * sizeof(float));
    float *yc = malloc((size_t)S * O * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); return 1; }
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double maxerr = 0, maxrel = 0;
    for (int i = 0; i < S * O; i++) {
        double e = fabs(yg[i] - yc[i]); if (e > maxerr) maxerr = e;
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; }
    }
    // microbench (GPU)
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs);
    double gpu_ms = (now() - t0) * 1000 / iters;
    // microbench (CPU ref, 1 iter — it's slow)
    double c0 = now(); cpu_ref(yc, x, w, sc, fmt, S, I, O); double cpu_ms = (now() - c0) * 1000;
    printf("fmt=%d S=%d I=%d O=%d | maxerr=%.4g maxrel=%.4g | gpu=%.3f ms  cpu_ref=%.3f ms\n",
           fmt, S, I, O, maxerr, maxrel, gpu_ms, cpu_ms);
    coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(yg); free(yc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput: record N dispatches in ONE command buffer, one submit + one
 * fence wait — the amortized per-matmul cost with the submit roundtrip spread across
 * the batch (how the real expert tier would drive it). Reuses the descriptor binding
 * left by a prior coli_vk_matmul call for this tensor/shape. */
static double bench_batched(ColiVkTensor *t, const float *x, int fmt, int S, int I, int O, int N) {
    size_t xb = (size_t)S * I * sizeof(float);
    memcpy(G.x.ptr, x, xb);
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* Every dispatch in the loop rewrites the same output, so the barrier must also
     * order the next write after the previous one (WAW), not only the reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int warm = 0; warm < 2; warm++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    G.cmd_ready = 0;   /* we clobbered the cached command buffer */
    return (now() - t0) * 1000.0 / iters / N;   /* ms per matmul, roundtrip amortized */
}

/* Fused gate+up correctness vs CPU ref: hidden = silu(gate*x)*(up*x). */
static int run_gate_up(int fmt, int S, int D, int I) {
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    float *x = malloc((size_t)S*D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
    float *gs = malloc(nsc*4), *us = malloc(nsc*4);
    float *hg = malloc((size_t)S*I*4), *hc = malloc((size_t)S*I*4);
    for (int i = 0; i < S*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
    for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f+(rand()%100)/10000.0f; us[o] = 0.01f+(rand()%100)/10000.0f; }
    ColiVkTensor *tg = NULL, *tu = NULL;
    if (!coli_vk_gate_up(&tg, &tu, hg, x, gw, gs, uw, us, fmt, S, D, I, g_ref_gs)) { printf("gate_up failed\n"); return 1; }
    for (int s = 0; s < S; s++) for (int o = 0; o < I; o++) {
        float gt = (float)ref_dot(x+(size_t)s*D, gw+(size_t)o*rb, gs, o, fmt, D);
        float ut = (float)ref_dot(x+(size_t)s*D, uw+(size_t)o*rb, us, o, fmt, D);
        hc[s*I+o] = (gt/(1.0f+expf(-gt)))*ut;
    }
    double maxrel = 0;
    for (int i = 0; i < S*I; i++) { double e = fabs(hg[i]-hc[i]); if (fabs(hc[i])>1e-2) { double r = e/fabs(hc[i]); if (r>maxrel) maxrel = r; } }
    printf("gate_up(fused) fmt=%d S=%d D=%d I=%d | maxrel=%.4g\n", fmt, S, D, I, maxrel);
    coli_vk_tensor_free(tg); coli_vk_tensor_free(tu);
    free(x); free(gw); free(uw); free(gs); free(us); free(hg); free(hc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput of the fused gate_up (N dispatches / one submit). */
static double bench_gu_batched(ColiVkTensor *tg, const float *x, int fmt, int S, int D, int I, int N) {
    if (!G.pipe_gu) return -1;   /* gate_up shader not loaded: run_gate_up already reported it */
    memcpy(G.x.ptr, x, (size_t)S*D*sizeof(float));
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PCGU pc = pcgu(fmt, S, D, I, tg->rowWords, tg->gs);
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* Every dispatch in the loop rewrites the same output, so the barrier must also
     * order the next write after the previous one (WAW), not only the reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    return (now()-t0)*1000.0/iters/N;
}

/* FAIR fused-gate_up throughput: cycle K DISTINCT experts (own descriptor set each) so
 * weights come from VRAM, not L2 — matching ROCm's expert_group reading distinct experts.
 * Returns ms per gate_up (one expert). */
static double bench_experts_fair(int fmt, int D, int I, int K, int Npass) {
    if (!G.pipe_gu) return -1;   /* gate_up shader not loaded: run_gate_up already reported it */
    if (K > 32) K = 32;
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    ColiVkTensor *tg[32] = {0}, *tu[32] = {0};
    float *h = malloc((size_t)I*4), *x = malloc((size_t)D*4);
    for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I); float *gs = malloc(nsc*4), *us = malloc(nsc*4);
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        coli_vk_gate_up(&tg[c], &tu[c], h, x, gw, gs, uw, us, fmt, 1, D, I, g_ref_gs);   /* uploads distinct experts */
        free(gw); free(uw); free(gs); free(us);
    }
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)(6*K)};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = (uint32_t)K, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool; vkCreateDescriptorPool(G.dev, &dpi, NULL, &pool);
    VkDescriptorSetLayout lays[32]; VkDescriptorSet sets[32]; for (int c = 0; c < K; c++) lays[c] = G.dsl_gu;
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = (uint32_t)K, .pSetLayouts = lays};
    vkAllocateDescriptorSets(G.dev, &dsa, sets);
    memcpy(G.x.ptr, x, (size_t)D*4);
    for (int c = 0; c < K; c++) {
        VkDescriptorBufferInfo bi[6] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.h.buf,.range=VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]; for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[c],.dstBinding=(uint32_t)i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);
    }
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    struct PCGU pc = pcgu(fmt, 1, D, I, tg[0]->rowWords, tg[0]->gs);
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    /* K experts write the same hidden slice in turn: order each write after the
     * previous one (WAW), not only after its reads. */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int pass = 0; pass < Npass; pass++) for (int c = 0; c < K; c++) {
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &sets[c], 0, NULL);
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), 1, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double ms = (now()-t0)*1000.0/iters/((double)Npass*K);
    vkDestroyDescriptorPool(G.dev, pool, NULL);
    for (int c = 0; c < K; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); }
    free(h); free(x); G.cmd_ready = 0; G.bound_tensor = NULL;
    return ms;
}

/* Full expert-group correctness (vs CPU ref) + fair throughput: K distinct experts,
 * one submit, hidden on-device. The real comparison to ROCm's coli_cuda_expert_group. */
static int run_expert_group(int fmt, int D, int I, int K) {
    if (K > 64) K = 64;
    size_t gu_rb = ref_rowbytes(fmt, D), gu_sc = ref_scales(fmt, D, I);
    size_t d_rb  = ref_rowbytes(fmt, I), d_sc  = ref_scales(fmt, I, D);
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    float *x = malloc((size_t)K*D*4), *yg = malloc((size_t)K*D*4), *yc = malloc((size_t)K*D*4);
    float *tmp = malloc((size_t)(D > I ? D : I) * 4);
    for (int i = 0; i < K*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc(gu_sc*4); hus[c] = malloc(gu_sc*4); hds[c] = malloc(d_sc*4);
        for (size_t o = 0; o < gu_sc; o++) { hgs[c][o] = 0.01f+(rand()%100)/10000.0f; hus[c][o] = 0.01f+(rand()%100)/10000.0f; }
        for (size_t o = 0; o < d_sc; o++) hds[c][o] = 0.01f+(rand()%100)/10000.0f;
        coli_vk_matmul(&tg[c], tmp, x, hgw[c], hgs[c], fmt, 1, D, I, g_ref_gs);   /* upload gate  (D->I) */
        coli_vk_matmul(&tu[c], tmp, x, huw[c], hus[c], fmt, 1, D, I, g_ref_gs);   /* upload up    (D->I) */
        coli_vk_matmul(&td[c], tmp, x, hdw[c], hds[c], fmt, 1, I, D, g_ref_gs);   /* upload down  (I->D) */
    }
    int rows[64]; for (int c = 0; c < K; c++) rows[c] = 1;
    if (!coli_vk_expert_group(tg, tu, td, rows, K, yg, x)) { printf("expert_group failed\n"); return 1; }
    float *hid = malloc((size_t)I*4);
    for (int c = 0; c < K; c++) {
        float *xc = x + (size_t)c*D;
        for (int o = 0; o < I; o++) {
            float gt = (float)ref_dot(xc, hgw[c]+(size_t)o*gu_rb, hgs[c], o, fmt, D);
            float ut = (float)ref_dot(xc, huw[c]+(size_t)o*gu_rb, hus[c], o, fmt, D);
            hid[o] = (gt/(1.0f+expf(-gt)))*ut;
        }
        for (int d = 0; d < D; d++)
            yc[c*D+d] = (float)ref_dot(hid, hdw[c]+(size_t)d*d_rb, hds[c], d, fmt, I);
    }
    double maxrel = 0;
    for (int i = 0; i < K*D; i++) { double e = fabs(yg[i]-yc[i]); if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
    coli_vk_expert_group(tg, tu, td, rows, K, yg, x);   /* warm + leaves G.eg_cmd recorded */
    /* async issue/take must reproduce the sync result exactly (same buffers/records) */
    {
        float *ya = malloc((size_t)K*D*4);
        if (!coli_vk_expert_group_issue(tg, tu, td, rows, K, x) || !coli_vk_expert_group_take(ya)) {
            printf("expert_group issue/take failed\n"); maxrel = 1;
        } else {
            double amr = 0;
            for (int i = 0; i < K*D; i++) { double e = fabs(ya[i]-yg[i]); if (fabs(yg[i])>1e-2) { double r = e/fabs(yg[i]); if (r>amr) amr = r; } }
            if (amr > 1e-6) { printf("issue/take deviates from sync: %.4g\n", amr); maxrel = 1; }
        }
        free(ya);
    }
    int iters = 20; double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, yg, x);
    double ms = (now()-t0)*1000.0/iters/K;
    /* GPU-only: re-submit the already-recorded command buffer (skips per-call host setup:
     * descriptor updates + recording), isolating raw GPU throughput. */
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double g0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double gpums = (now()-g0)*1000.0/iters/K;
    printf("FULL VK expert_group fmt=%d %2d experts | maxrel=%.4g | per-call %.4f  GPU-only %.4f ms/expert (ROCm 0.179)\n",
           fmt, K, maxrel, ms, gpums);
    free(hid); free(x); free(yg); free(yc); free(tmp);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    /* The gate_up -> down chain accumulates fp32 rounding across two long reductions
     * (D=6144 then I) vs the double-precision CPU ref; 1-2e-3 shows up at any K
     * depending on the random draw, and is fine for greedy argmax. 3e-3 keeps the
     * gate honest without flagging the known fp behavior as a failure. */
    return maxrel > 3e-3 ? 1 : 0;
}

/* MLA absorb attention vs a CPU ref that mirrors glm.c's absorb loop exactly:
 * qabs = sum_d q[d]*deq(row rbase+d)*ws, scores over cache rows [st0, T-S+s],
 * softmax, weighted latent, value-row projection. */
static int run_absorb(int fmt, int S, int H, int Q, int R, int V, int K, int st0, int T, int layer) {
    size_t rb = ref_rowbytes(fmt, K);
    int O = H * (Q + V), ngK = (K + 63) / 64;
    size_t nws = ref_scales(fmt, K, O);
    uint8_t *w = malloc(rb * O); float *ws = malloc(nws * 4);
    float *q = malloc((size_t)S * H * (Q + R) * 4);
    float *L = malloc((size_t)T * K * 4), *Rr = malloc((size_t)T * R * 4);
    float *cg = malloc((size_t)S * H * V * 4), *cc = malloc((size_t)S * H * V * 4);
    for (size_t i = 0; i < rb * (size_t)O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nws; o++) ws[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < S * H * (Q + R); i++) q[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * K; i++) L[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * R; i++) Rr[i] = (rand() % 200 - 100) / 100.0f;
    float scale = 0.13f;
    if (!coli_vk_kv_ensure(layer, T, K, R)) { printf("kv_ensure failed\n"); return 1; }
    for (int t = 0; t < T; t++)
        if (!coli_vk_kv_row(layer, t, L + (size_t)t * K, Rr + (size_t)t * R)) { printf("kv_row failed\n"); return 1; }
    ColiVkTensor *kvb = NULL;
    if (!coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale)) {
        printf("absorb failed\n"); return 1; }
    float *qabs = malloc((size_t)K * 4), *clat = malloc((size_t)K * 4), *sc = malloc((size_t)(T - st0) * 4);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *qp = q + ((size_t)s * H + h) * (Q + R), *qr = qp + Q;
        int rbase = h * (Q + V), nt = (T - S + s + 1) - st0;
        for (int i = 0; i < K; i++) qabs[i] = 0;
        for (int d = 0; d < Q; d++) { const uint8_t *row = w + (size_t)(rbase + d) * rb;
            for (int i = 0; i < K; i++) {
                float sw = fmt == 5 ? ws[(size_t)(rbase + d) * ngK + (i >> 6)]
                         : fmt == 4 ? ws[(size_t)(rbase + d) * ((K + g_ref_gs - 1) / g_ref_gs) + i / g_ref_gs]
                         : ws[rbase + d];
                qabs[i] += qp[d] * deq(row, fmt, i) * sw; } }
        for (int j = 0; j < nt; j++) { int t = st0 + j;
            double a = 0;
            for (int i = 0; i < K; i++) a += qabs[i] * L[(size_t)t * K + i];
            for (int d = 0; d < R; d++) a += qr[d] * Rr[(size_t)t * R + d];
            sc[j] = (float)(a * scale); }
        float mx = sc[0]; for (int j = 1; j < nt; j++) if (sc[j] > mx) mx = sc[j];
        double sum = 0; for (int j = 0; j < nt; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        for (int i = 0; i < K; i++) clat[i] = 0;
        for (int j = 0; j < nt; j++) { float a = (float)(sc[j] / sum);
            for (int i = 0; i < K; i++) clat[i] += a * L[(size_t)(st0 + j) * K + i]; }
        for (int v = 0; v < V; v++)
            cc[((size_t)s * H + h) * V + v] =
                (float)ref_dot(clat, w + (size_t)(rbase + Q + v) * rb, ws, rbase + Q + v, fmt, K);
    }
    double maxrel = 0, maxerr = 0;
    for (int i = 0; i < S * H * V; i++) { double e = fabs(cg[i] - cc[i]); if (e > maxerr) maxerr = e;
        if (fabs(cc[i]) > 1e-2) { double r = e / fabs(cc[i]); if (r > maxrel) maxrel = r; } }
    double t0 = now(); int iters = 20;   /* per-call cost, the engine pattern (one submit/layer) */
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale);
    double ms = (now() - t0) * 1000 / iters;
    printf("absorb fmt=%d S=%d H=%d Q=%d R=%d V=%d K=%d st0=%d T=%d | maxerr=%.4g maxrel=%.4g | %.3f ms/call\n",
           fmt, S, H, Q, R, V, K, st0, T, maxerr, maxrel, ms);
    /* fused absorb+project vs (CPU ctx ref) @ (CPU o ref) */
    int Dout = 512; size_t orb = ref_rowbytes(fmt, H * V), onsc = ref_scales(fmt, H * V, Dout);
    uint8_t *owt = malloc(orb * Dout); float *osc = malloc(onsc * 4);
    float *og = malloc((size_t)S * Dout * 4), *oc = malloc((size_t)S * Dout * 4);
    for (size_t i = 0; i < orb * (size_t)Dout; i++) owt[i] = rand() & 0xff;
    for (size_t o = 0; o < onsc; o++) osc[o] = 0.01f + (rand() % 100) / 10000.0f;
    ColiVkTensor *ot = NULL;
    if (!coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout)) { printf("absorb_project failed\n"); return 1; }
    cpu_ref(oc, cc, owt, osc, fmt, S, H * V, Dout);
    double pmaxrel = 0;
    for (int i = 0; i < S * Dout; i++) { double e = fabs(og[i] - oc[i]);
        if (fabs(oc[i]) > 1e-2) { double r = e / fabs(oc[i]); if (r > pmaxrel) pmaxrel = r; } }
    t0 = now();
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout);
    printf("absorb+project fused                  | maxrel=%.4g | %.3f ms/call (unfused absorb was %.3f)\n",
           pmaxrel, (now() - t0) * 1000 / iters, ms);
    coli_vk_tensor_free(ot); free(owt); free(osc); free(og); free(oc);
    coli_vk_tensor_free(kvb);
    free(w); free(ws); free(q); free(L); free(Rr); free(cg); free(cc); free(qabs); free(clat); free(sc);
    return (maxrel > 2e-3 || pmaxrel > 5e-3) ? 1 : 0;
}


/* q-prep chain vs CPU ref: matmul(qa) -> rmsnorm -> matmul(qb), kv_a alongside. */
static int run_qprep(int fmt, int S, int I, int Oqa, int Okva, int Oqb) {
    size_t rba = ref_rowbytes(fmt, I), rbb = ref_rowbytes(fmt, Oqa);
    size_t nsa = ref_scales(fmt, I, Oqa), nsk = ref_scales(fmt, I, Okva), nsb = ref_scales(fmt, Oqa, Oqb);
    uint8_t *wa = malloc(rba * Oqa), *wk = malloc(rba * Okva), *wb = malloc(rbb * Oqb);
    float *sa = malloc(nsa * 4), *sk = malloc(nsk * 4), *sb = malloc(nsb * 4);
    float *ln = malloc((size_t)Oqa * 4), *x = malloc((size_t)S * I * 4);
    float *qg = malloc((size_t)S * Oqb * 4), *kvg = malloc((size_t)S * Okva * 4);
    float *lat = malloc((size_t)S * Oqa * 4), *qc = malloc((size_t)S * Oqb * 4), *kvc = malloc((size_t)S * Okva * 4);
    for (size_t i = 0; i < rba * (size_t)Oqa; i++) wa[i] = rand() & 0xff;
    for (size_t i = 0; i < rba * (size_t)Okva; i++) wk[i] = rand() & 0xff;
    for (size_t i = 0; i < rbb * (size_t)Oqb; i++) wb[i] = rand() & 0xff;
    for (size_t o = 0; o < nsa; o++) sa[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsk; o++) sk[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsb; o++) sb[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < Oqa; i++) ln[i] = 0.5f + (rand() % 100) / 100.0f;
    for (int i = 0; i < S * I; i++) x[i] = (rand() % 2000 - 1000) / 500.0f;
    ColiVkTensor *ta = NULL, *tk = NULL, *tb = NULL;
    static int qp_layer = 140;         /* distinct high slot per case: the per-layer ln
                                        * cache is upload-once by design (engine weights
                                        * are immutable); reuse here would mix cases */
    int layer = qp_layer++;
    if (!coli_vk_attn_qprep(layer, &ta, wa, sa, Oqa, &tk, wk, sk, Okva, &tb, wb, sb, Oqb,
                            fmt, g_ref_gs, ln, 1e-6f, x, S, I, qg, kvg, NULL)) {
        printf("qprep unavailable (rmsnorm.spv missing?)\n"); return 1; }
    cpu_ref(lat, x, wa, sa, fmt, S, I, Oqa);
    cpu_ref(kvc, x, wk, sk, fmt, S, I, Okva);
    for (int s = 0; s < S; s++) {                       /* rmsnorm rows like colibri.c */
        double ms = 0; float *r = lat + (size_t)s * Oqa;
        for (int i = 0; i < Oqa; i++) ms += (double)r[i] * r[i];
        float rr = 1.0f / sqrtf((float)(ms / Oqa) + 1e-6f);
        for (int i = 0; i < Oqa; i++) r[i] = r[i] * rr * ln[i];
    }
    cpu_ref(qc, lat, wb, sb, fmt, S, Oqa, Oqb);
    float mq = 0, mk = 0;
    for (int i = 0; i < S * Oqb; i++) { float d = fabsf(qg[i] - qc[i]) / (fabsf(qc[i]) + 1e-3f); if (d > mq) mq = d; }
    for (int i = 0; i < S * Okva; i++) { float d = fabsf(kvg[i] - kvc[i]) / (fabsf(kvc[i]) + 1e-3f); if (d > mk) mk = d; }
    printf("qprep fmt=%d S=%d I=%d (%d->%d, kv %d) | maxrel q %.4g kv %.4g\n", fmt, S, I, Oqa, Oqb, Okva, mq, mk);
    coli_vk_tensor_free(ta); coli_vk_tensor_free(tk); coli_vk_tensor_free(tb);
    free(wa); free(wk); free(wb); free(sa); free(sk); free(sb); free(ln); free(x);
    free(qg); free(kvg); free(lat); free(qc); free(kvc);
    /* q crosses TWO quantized reductions + the norm; random +-8-nibble rows are
     * cancellation-heavy, so fp32-vs-f64 divergence amplifies ~10x vs one GEMV
     * (same reasoning as the expert_group 3e-3 threshold). Engine-level logit
     * comparison on real weights is the tight check. */
    return mq > 1e-2f || mk > 1e-3f;
}

/* Multi-row shaders (qmatmul_mr / qmatmul_gate_up_mr) against the CPU reference AND
 * against the one-row shaders they replace for S > 1: same per-row summation order,
 * so the two must agree bit for bit. Also times both on the same inputs. */
/* Elements whose bit patterns differ (+0/-0 and NaN payloads count), plus the
 * non-finite elements of `a`; a NaN would otherwise slip through a > comparison. */
static size_t mr_diff(const float *a, const float *b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], sizeof(float)) != 0 || !isfinite(a[i])) d++;
    return d;
}
static int run_mr_case(int fmt, int S, int I, int O, int iters) {
    if (!G.mr) { printf("multi-row shaders not loaded\n"); return 1; }
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O), ny = (size_t)S * O;
    float *x = malloc((size_t)S * I * sizeof(float)), *sc = malloc(nsc * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *ym = malloc(ny * 4), *y1 = malloc(ny * 4), *yc = malloc(ny * 4);
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
    ColiVkTensor *t = NULL;
    int mr = G.mr, bad = 0;
    if (!coli_vk_matmul(&t, ym, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); return 1; }
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, ym, x, w, sc, fmt, S, I, O, g_ref_gs);
    double ms_mr = (now() - t0) * 1000 / iters;
    G.mr = 0; G.cmd_ready = 0;
    coli_vk_matmul(&t, y1, x, w, sc, fmt, S, I, O, g_ref_gs);
    t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, y1, x, w, sc, fmt, S, I, O, g_ref_gs);
    double ms_1 = (now() - t0) * 1000 / iters;
    G.mr = mr; G.cmd_ready = 0;
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double maxrel = 0;
    for (size_t i = 0; i < ny; i++) { double e = fabs(ym[i] - yc[i]);
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; } }
    size_t d1 = mr_diff(ym, y1, ny);
    printf("MR%d fmt=%d S=%2d I=%5d O=%5d | maxrel=%.4g | vs one-row %s | %.3f ms (one-row %.3f, %.2fx)\n",
           mr, fmt, S, I, O, maxrel, d1 == 0 ? "bit-identical" : "DIFFERS", ms_mr, ms_1, ms_1 / ms_mr);
    bad = maxrel > 1e-3 || d1 != 0;
    coli_vk_tensor_free(t);
    free(x); free(sc); free(w); free(ym); free(y1); free(yc);
    return bad;
}
/* Fused gate+up, multi-row vs one-row, optionally with the SwiGLU clamp on. */
static int run_mr_gate_up(int fmt, int S, int D, int I, float limit) {
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I), nh = (size_t)S * I;
    float *x = malloc((size_t)S*D*4), *gs = malloc(nsc*4), *us = malloc(nsc*4);
    uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
    float *hm = malloc(nh*4), *h1 = malloc(nh*4), *hc = malloc(nh*4);
    for (int i = 0; i < S*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
    for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f+(rand()%100)/10000.0f; us[o] = 0.01f+(rand()%100)/10000.0f; }
    coli_vk_set_swiglu_limit(limit);
    ColiVkTensor *tg = NULL, *tu = NULL;
    int ok = coli_vk_gate_up(&tg, &tu, hm, x, gw, gs, uw, us, fmt, S, D, I, g_ref_gs);
    int mr = G.mr; G.mr = 0;
    ok = ok && coli_vk_gate_up(&tg, &tu, h1, x, gw, gs, uw, us, fmt, S, D, I, g_ref_gs);
    G.mr = mr;
    coli_vk_set_swiglu_limit(0.f);
    if (!ok) { printf("gate_up failed\n"); return 1; }
    for (int s = 0; s < S; s++) for (int o = 0; o < I; o++) {
        float gt = (float)ref_dot(x+(size_t)s*D, gw+(size_t)o*rb, gs, o, fmt, D);
        float ut = (float)ref_dot(x+(size_t)s*D, uw+(size_t)o*rb, us, o, fmt, D);
        if (limit > 0.f) { if (gt > limit) gt = limit; if (ut > limit) ut = limit; if (ut < -limit) ut = -limit; }
        hc[s*I+o] = (gt/(1.0f+expf(-gt)))*ut;
    }
    double maxrel = 0;
    for (size_t i = 0; i < nh; i++) { double e = fabs(hm[i]-hc[i]); if (fabs(hc[i])>1e-2) { double r = e/fabs(hc[i]); if (r>maxrel) maxrel = r; } }
    size_t d1 = mr_diff(hm, h1, nh);
    printf("MR%d gate_up fmt=%d S=%2d D=%d I=%d limit=%.1f | maxrel=%.4g | vs one-row %s\n",
           mr, fmt, S, D, I, limit, maxrel, d1 == 0 ? "bit-identical" : "DIFFERS");
    coli_vk_tensor_free(tg); coli_vk_tensor_free(tu);
    free(x); free(gs); free(us); free(gw); free(uw); free(hm); free(h1); free(hc);
    return maxrel > 1e-3 || d1 != 0;
}
/* One tensor driven through changing S (1 -> 5 -> 2 -> 1 -> 9): coli_vk_matmul caches
 * the recorded command buffer per shape, and the S=1 / multi-row switch must re-record. */
static int run_mr_cache(int fmt, int I, int O) {
    static const int seq[5] = {1, 5, 2, 1, 9};
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O);
    float *x = malloc((size_t)9*I*4), *sc = malloc(nsc*4), *yg = malloc((size_t)9*O*4), *yc = malloc((size_t)9*O*4);
    uint8_t *w = malloc(rb*O);
    for (int i = 0; i < 9*I; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*O; i++) w[i] = rand()&0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f+(rand()%100)/10000.0f;
    ColiVkTensor *t = NULL; int bad = 0;
    for (int k = 0; k < 5; k++) {
        int S = seq[k];
        if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); bad = 1; break; }
        cpu_ref(yc, x, w, sc, fmt, S, I, O);
        double maxrel = 0; size_t nonfin = 0;
        for (int i = 0; i < S*O; i++) { if (!isfinite(yg[i])) nonfin++;
            double e = fabs(yg[i]-yc[i]); if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
        if (maxrel > 1e-3 || nonfin) { printf("MR cache fmt=%d S=%d: maxrel=%.4g nonfinite=%zu\n", fmt, S, maxrel, nonfin); bad = 1; }
    }
    printf("MR%d cache transitions fmt=%d I=%d O=%d S=1,5,2,1,9 | %s\n", G.mr, fmt, I, O, bad ? "FAIL" : "ok");
    coli_vk_tensor_free(t); free(x); free(sc); free(yg); free(yc); free(w);
    return bad;
}
/* Tiled cooperative-matrix shaders: inputs are rounded to f16, so they are held
 * to the CPU reference by error norms instead of bit identity -- relative L2
 * error of the whole output, and the largest single error against the largest
 * magnitude -- plus finiteness. Also times them against the GEMV path. */
static void coop_err(const float *y, const float *ref, size_t n, double *rl2, double *rmax, size_t *nonfin) {
    double se = 0, sr = 0, me = 0, mr = 0; *nonfin = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(y[i])) { (*nonfin)++; continue; }
        double e = (double)y[i] - ref[i];
        se += e * e; sr += (double)ref[i] * ref[i];
        if (fabs(e) > me) me = fabs(e);
        if (fabs(ref[i]) > mr) mr = fabs(ref[i]);
    }
    *rl2 = sr > 0 ? sqrt(se / sr) : 0; *rmax = mr > 0 ? me / mr : 0;
}
static int run_coop_case(int fmt, int S, int I, int O, int iters) {
    if (!G.coop) { printf("coopmat shaders not loaded\n"); return 1; }
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O), ny = (size_t)S * O;
    float *x = malloc((size_t)S * I * 4), *sc = malloc(nsc * 4), *yc0 = malloc(ny * 4);
    uint8_t *w = malloc(rb * O);
    float *yco = malloc(ny * 4), *y1 = malloc(ny * 4), *yc = malloc(ny * 4);
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
    ColiVkTensor *t = NULL;
    int cmin = G.coop_min; G.coop_min = 1; g_coop_any = 1;
    if (!coli_vk_matmul(&t, yco, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); return 1; }
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yco, x, w, sc, fmt, S, I, O, g_ref_gs);
    double ms_co = (now() - t0) * 1000 / iters;
    int co = G.coop; G.coop = 0; G.cmd_ready = 0;
    coli_vk_matmul(&t, y1, x, w, sc, fmt, S, I, O, g_ref_gs);
    t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, y1, x, w, sc, fmt, S, I, O, g_ref_gs);
    double ms_1 = (now() - t0) * 1000 / iters;
    G.coop = co; G.coop_min = cmin; g_coop_any = 0; G.cmd_ready = 0;
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double rl2, rmax, rl2g, rmaxg; size_t nf, nfg;
    coop_err(yco, yc, ny, &rl2, &rmax, &nf);
    coop_err(y1, yc, ny, &rl2g, &rmaxg, &nfg);
    printf("COOP fmt=%d S=%3d I=%5d O=%5d | relL2 %.2e max %.2e (GEMV %.2e / %.2e)%s | %.3f ms (GEMV %.3f, %.2fx)\n",
           fmt, S, I, O, rl2, rmax, rl2g, rmaxg, nf ? " NONFINITE" : "", ms_co, ms_1, ms_1 / ms_co);
    coli_vk_tensor_free(t);
    free(x); free(sc); free(yc0); free(w); free(yco); free(y1); free(yc);
    return rl2 > 2e-3 || rmax > 1e-2 || nf;
}
/* Expert group through the tiled shaders (rows per expert cycle 1..maxrows; the
 * experts below coop_min rows take the GEMV path inside the same group). */
static int run_coop_expert_group(int fmt, int D, int I, int K, int maxrows, int iters, float limit) {
    if (!G.coop) { printf("coopmat shaders not loaded\n"); return 1; }
    if (K > 64) K = 64;
    size_t gu_rb = ref_rowbytes(fmt, D), gu_sc = ref_scales(fmt, D, I);
    size_t d_rb  = ref_rowbytes(fmt, I), d_sc  = ref_scales(fmt, I, D);
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    int rows[64], total = 0;
    for (int c = 0; c < K; c++) { rows[c] = 1 + (c * 7) % maxrows; total += rows[c]; }
    float *x = malloc((size_t)total*D*4), *yco = malloc((size_t)total*D*4);
    float *y1 = malloc((size_t)total*D*4), *yc = malloc((size_t)total*D*4), *hid = malloc((size_t)I*4);
    for (int i = 0; i < total*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc(gu_sc*4); hus[c] = malloc(gu_sc*4); hds[c] = malloc(d_sc*4);
        for (size_t o = 0; o < gu_sc; o++) { hgs[c][o] = 0.01f+(rand()%100)/10000.0f; hus[c][o] = 0.01f+(rand()%100)/10000.0f; }
        for (size_t o = 0; o < d_sc; o++) hds[c][o] = 0.01f+(rand()%100)/10000.0f;
        coli_vk_tensor_ensure(&tg[c], hgw[c], hgs[c], fmt, D, I, g_ref_gs);
        coli_vk_tensor_ensure(&tu[c], huw[c], hus[c], fmt, D, I, g_ref_gs);
        coli_vk_tensor_ensure(&td[c], hdw[c], hds[c], fmt, I, D, g_ref_gs);
    }
    coli_vk_set_swiglu_limit(limit);
    if (!coli_vk_expert_group(tg, tu, td, rows, K, yco, x)) { printf("expert_group failed\n"); return 1; }
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, yco, x);
    double ms_co = (now() - t0) * 1000 / iters;
    int co = G.coop; G.coop = 0;
    coli_vk_expert_group(tg, tu, td, rows, K, y1, x);
    t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, y1, x);
    double ms_1 = (now() - t0) * 1000 / iters;
    G.coop = co;
    coli_vk_set_swiglu_limit(0.f);
    for (int c = 0, off = 0; c < K; off += rows[c], c++)
        for (int r = 0; r < rows[c]; r++) {
            const float *xr = x + (size_t)(off + r)*D;
            for (int o = 0; o < I; o++) {
                float gt = (float)ref_dot(xr, hgw[c]+(size_t)o*gu_rb, hgs[c], o, fmt, D);
                float ut = (float)ref_dot(xr, huw[c]+(size_t)o*gu_rb, hus[c], o, fmt, D);
                if (limit > 0.f) { if (gt > limit) gt = limit; if (ut > limit) ut = limit; if (ut < -limit) ut = -limit; }
                hid[o] = (gt/(1.0f+expf(-gt)))*ut;
            }
            for (int d = 0; d < D; d++)
                yc[(size_t)(off + r)*D + d] = (float)ref_dot(hid, hdw[c]+(size_t)d*d_rb, hds[c], d, fmt, I);
        }
    double rl2, rmax, rl2g, rmaxg; size_t nf, nfg;
    coop_err(yco, yc, (size_t)total*D, &rl2, &rmax, &nf);
    coop_err(y1, yc, (size_t)total*D, &rl2g, &rmaxg, &nfg);
    printf("COOP expert_group fmt=%d D=%d I=%d %2d experts, %4d rows (1..%d, coop from %d) limit=%.1f | relL2 %.2e max %.2e (GEMV %.2e / %.2e)%s | %.3f ms (GEMV %.3f, %.2fx)\n",
           fmt, D, I, K, total, maxrows, G.coop_min, limit, rl2, rmax, rl2g, rmaxg, nf ? " NONFINITE" : "", ms_co, ms_1, ms_1 / ms_co);
    free(x); free(yco); free(y1); free(yc); free(hid);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    return rl2 > 3e-3 || rmax > 2e-2 || nf;
}

/* End-to-end token input -> gather -> experts -> ordered weighted reduction.
 * Shuffle pair IDs, leave holes (CPU misses), and reserve one all-miss token.
 * Compare both CPU math and the old <=64-expert host pack/readback path. */
static int run_expert_prefill(int D, int I, int E, int maxrows, float limit) {
    int fmt = 4, K = 8, total = 0, bad = 0;
    int *rows = malloc(E*sizeof(int));
    ColiVkTensor **tg = calloc(E, sizeof(*tg)), **tu = calloc(E, sizeof(*tu)), **td = calloc(E, sizeof(*td));
    for (int c = 0; c < E; c++) { rows[c] = 1+(c*7)%maxrows; total += rows[c]; }
    int S = total/K + total/(K*3) + 4, pairs = S*K;
    int *order = malloc(pairs*sizeof(int)), *inv = malloc(pairs*sizeof(int)), eligible = 0;
    for (int p = 0; p < pairs; p++) {
        inv[p] = -1;
        if (p/K < S-1 && p%5) order[eligible++] = p;
    }
    for (int r = eligible-1; r > 0; r--) { int j = rand()%(r+1), t = order[r]; order[r] = order[j]; order[j] = t; }
    for (int r = 0; r < total; r++) inv[order[r]] = r;
    size_t ny = (size_t)S*D;
    float *x = malloc(ny*4), *w = malloc(pairs*4), *packed = malloc((size_t)total*D*4);
    float *yp = malloc((size_t)total*D*4), *yc = malloc((size_t)total*D*4);
    float *y = malloc(ny*4), *again = malloc(ny*4), *ref = calloc(ny,4), *old = calloc(ny,4);
    float *hid = malloc(I*4);
    for (size_t j = 0; j < ny; j++) x[j] = (rand()%200-100)/100.f;
    for (int p = 0; p < pairs; p++) w[p] = p%11 ? (rand()%100)/400.f : 0.f;
    for (int r = 0; r < total; r++) memcpy(packed+(size_t)r*D, x+(size_t)(order[r]/K)*D, D*4);
    size_t rb = ref_rowbytes(fmt,D), drb = ref_rowbytes(fmt,I);
    size_t ns = ref_scales(fmt,D,I), nds = ref_scales(fmt,I,D);
    uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I), *dw = malloc(drb*D);
    float *gs = malloc(ns*4), *us = malloc(ns*4), *ds = malloc(nds*4);
    for (int c = 0, off = 0; c < E; off += rows[c++]) {
        for (size_t j = 0; j < rb*I; j++) { gw[j] = rand()&255; uw[j] = rand()&255; }
        for (size_t j = 0; j < drb*D; j++) dw[j] = rand()&255;
        for (size_t j = 0; j < ns; j++) { gs[j] = .01f+(rand()%100)/10000.f; us[j] = .01f+(rand()%100)/10000.f; }
        for (size_t j = 0; j < nds; j++) ds[j] = .01f+(rand()%100)/10000.f;
        if (!coli_vk_tensor_ensure(&tg[c],gw,gs,fmt,D,I,g_ref_gs) ||
            !coli_vk_tensor_ensure(&tu[c],uw,us,fmt,D,I,g_ref_gs) ||
            !coli_vk_tensor_ensure(&td[c],dw,ds,fmt,I,D,g_ref_gs)) { bad = 1; goto done; }
        for (int r = 0; r < rows[c]; r++) {
            const float *xr = packed+(size_t)(off+r)*D;
            for (int o = 0; o < I; o++) {
                float gt = (float)ref_dot(xr,gw+(size_t)o*rb,gs,o,fmt,D);
                float ut = (float)ref_dot(xr,uw+(size_t)o*rb,us,o,fmt,D);
                if (limit > 0) { if (gt > limit) gt = limit; if (ut > limit) ut = limit; if (ut < -limit) ut = -limit; }
                hid[o] = (gt/(1.f+expf(-gt)))*ut;
            }
            for (int d = 0; d < D; d++) yc[(size_t)(off+r)*D+d] = (float)ref_dot(hid,dw+(size_t)d*drb,ds,d,fmt,I);
        }
    }
    coli_vk_set_swiglu_limit(limit);
    for (int c = 0, off = 0; c < E;) {
        int n = E-c < 64 ? E-c : 64;
        if (!coli_vk_expert_group(tg+c,tu+c,td+c,rows+c,n,yp+(size_t)off*D,packed+(size_t)off*D)) { bad = 1; goto done; }
        for (int j = 0; j < n; j++) off += rows[c++];
    }
    for (int p = 0; p < pairs; p++) if (inv[p] >= 0)
        for (int d = 0; d < D; d++) {
            ref[(size_t)(p/K)*D+d] += w[p]*yc[(size_t)inv[p]*D+d];
            old[(size_t)(p/K)*D+d] += w[p]*yp[(size_t)inv[p]*D+d];
        }
    if (!coli_vk_expert_prefill(tg,tu,td,rows,E,order,w,S,K,x,y) ||
        !coli_vk_expert_prefill(tg,tu,td,rows,E,order,w,S,K,x,again)) { bad = 1; goto done; }
    double l2, mx, lo, mo; size_t nf, no;
    coop_err(y,ref,ny,&l2,&mx,&nf); coop_err(y,old,ny,&lo,&mo,&no);
    int repeat = memcmp(y,again,ny*4) == 0;
    /* coop_err intentionally normalizes by reference energy: separately check
     * exact zeros for tokens with no residents, where that norm is undefined. */
    for (int d = 0; d < D; d++) if (y[(size_t)(S-1)*D+d] != 0.f) bad = 1;
    printf("PREFILL D=%d I=%d E=%d rows=%d S=%d clamp=%.1f coop=%d | CPU relL2 %.2e max %.2e | old relL2 %.2e max %.2e | repeat %s\n",
           D,I,E,total,S,limit,G.coop,l2,mx,lo,mo,repeat ? "exact" : "DIFF");
    /* the grouped path runs every expert through the f16 tiles, the old per-expert
     * path takes the f32 GEMV below 8 rows: equal only up to f16 rounding then */
    double tol_old = (G.coop && G.pipe_grp[0]) ? 3e-3 : 2e-6;
    bad |= nf || no || l2 > 3e-3 || mx > 2e-2 || lo > tol_old || mo > (tol_old > 1e-5 ? 2e-2 : 2e-6) || !repeat;
    /* Reject duplicate/out-of-range metadata and decode without submitting. */
    int saved = order[0]; order[0] = pairs;
    bad |= coli_vk_expert_prefill(tg,tu,td,rows,E,order,w,S,K,x,y);
    if (total > 1) { order[0] = order[1]; bad |= coli_vk_expert_prefill(tg,tu,td,rows,E,order,w,S,K,x,y); }
    order[0] = saved;
    bad |= coli_vk_expert_prefill(tg,tu,td,rows,E,order,w,1,K,x,y);
 done:
    coli_vk_set_swiglu_limit(0);
    for (int c = 0; c < E; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]); }
    free(rows); free(tg); free(tu); free(td); free(order); free(inv); free(x); free(w); free(packed);
    free(yp); free(yc); free(y); free(again); free(ref); free(old); free(hid);
    free(gw); free(uw); free(dw); free(gs); free(us); free(ds);
    return bad;
}

/* VK_EG_BENCH=1 (with COLI_VK_TS=1): one Qwen3.6 prefill MoE layer through the
 * grouped path -- S tokens, top-K of E experts, random routing, fmt 4 g64 -- timed
 * per stage by the GPU timestamps, no reference. For kernel experiments. */
static int run_expert_prefill_bench(int S, int K, int E, int D, int I, int iters) {
    int fmt = 4, pairs = S * K, bad = 0;
    int *rows = calloc(E, sizeof(int)), *pe = malloc(pairs * sizeof(int)), *order = malloc(pairs * sizeof(int));
    float *w = malloc(pairs * 4), *x = malloc((size_t)S * D * 4), *y = malloc((size_t)S * D * 4);
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int e, dup;
            do { e = rand() % E; dup = 0; for (int j = 0; j < k; j++) dup |= pe[s * K + j] == e; } while (dup);
            pe[s * K + k] = e; rows[e]++; w[s * K + k] = 0.1f;
        }
    int *first = calloc(E, sizeof(int)), *fill = calloc(E, sizeof(int)), nex = 0, *ex = malloc(E * sizeof(int)), *rr = malloc(E * sizeof(int));
    for (int e = 0, o = 0; e < E; e++) { first[e] = o; o += rows[e]; }
    for (int p = 0; p < pairs; p++) { int e = pe[p]; order[first[e] + fill[e]++] = p; }
    for (int e = 0; e < E; e++) if (rows[e]) { ex[nex] = e; rr[nex++] = rows[e]; }
    for (size_t j = 0; j < (size_t)S * D; j++) x[j] = (rand() % 200 - 100) / 100.f;
    size_t rb = ref_rowbytes(fmt, D), drb = ref_rowbytes(fmt, I), ns = ref_scales(fmt, D, I), nds = ref_scales(fmt, I, D);
    uint8_t *gw = malloc(rb * I), *dw = malloc(drb * D); float *gs = malloc(ns * 4), *ds = malloc(nds * 4);
    for (size_t j = 0; j < rb * I; j++) gw[j] = rand() & 255;
    for (size_t j = 0; j < drb * D; j++) dw[j] = rand() & 255;
    for (size_t j = 0; j < ns; j++) gs[j] = .01f;
    for (size_t j = 0; j < nds; j++) ds[j] = .01f;
    ColiVkTensor **tg = calloc(nex, sizeof(*tg)), **tu = calloc(nex, sizeof(*tu)), **td = calloc(nex, sizeof(*td));
    for (int c = 0; c < nex; c++)
        if (!coli_vk_tensor_ensure(&tg[c], gw, gs, fmt, D, I, 64) || !coli_vk_tensor_ensure(&tu[c], gw, gs, fmt, D, I, 64) ||
            !coli_vk_tensor_ensure(&td[c], dw, ds, fmt, I, D, 64)) { bad = 1; goto out; }
    /* order must be grouped by expert in the ex[] order: it already is (ascending e) */
    coli_vk_expert_prefill(tg, tu, td, rr, nex, order, w, S, K, x, y);   /* warm-up */
    memset(g_tsb, 0, sizeof g_tsb);
    for (int i = 0; i < iters; i++)
        if (!coli_vk_expert_prefill(tg, tu, td, rr, nex, order, w, S, K, x, y)) { bad = 1; goto out; }
    printf("EG_BENCH S=%d K=%d E=%d (%d used) D=%d I=%d | per layer: gate_up %.3f  down %.3f  reduce %.3f ms\n",
           S, K, E, nex, D, I, g_tsb[TSB_EG][1] / iters, g_tsb[TSB_EG][2] / iters, g_tsb[TSB_EG][3] / iters);
 out:
    for (int c = 0; c < nex; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]); }
    free(rows); free(pe); free(order); free(w); free(x); free(y); free(first); free(fill); free(ex); free(rr);
    free(gw); free(dw); free(gs); free(ds); free(tg); free(tu); free(td);
    return bad;
}

static int run_expert_prefill_tests(void) {
    g_ref_gs = 64;
    int bad = 0, co = G.coop, any = g_mr_any;
    bad |= run_expert_prefill(256,128,65,17,0.f);
    bad |= run_expert_prefill(256,128,256,9,0.5f);
    bad |= run_expert_prefill(2048,512,8,33,0.f);
    bad |= run_expert_prefill(256,128,1,1,0.f);  /* shrink after large buffers */
    G.coop = 0; g_mr_any = 1;
    bad |= run_expert_prefill(256,128,65,17,0.f); /* GEMV/MR without tiles */
    G.coop = co; g_mr_any = any;
    return bad;
}

/* DeltaNet recurrence against the CPU loop of qwen36.c (deltanet_phased, step 3):
 * same state layout, key head h/rep, decay then k.S, delta, update, q.S. */
static int run_dn_recur_case(int S, int vh, int vk, int kdim, int vdim) {
    if (!G.pipe_dr) { printf("dn_recur shader not loaded\n"); return 1; }
    int vstride = 2 * vk * kdim + vh * vdim + 7, voff = 2 * vk * kdim;   /* conv row: q | k | v (+pad) */
    size_t nq = (size_t)S * vk * kdim, ns = (size_t)vh * kdim * vdim, no = (size_t)S * vh * vdim;
    float *qn = malloc(nq * 4), *kn = malloc(nq * 4), *vs = malloc((size_t)S * vstride * 4);
    float *bt = malloc((size_t)S * vh * 4), *ge = malloc((size_t)S * vh * 4);
    float *s0 = malloc(ns * 4), *sg = malloc(ns * 4), *sc = malloc(ns * 4), *og = malloc(no * 4), *oc = malloc(no * 4);
    for (size_t i = 0; i < nq; i++) { qn[i] = (rand() % 200 - 100) / 1000.0f; kn[i] = (rand() % 200 - 100) / 1000.0f; }
    for (size_t i = 0; i < (size_t)S * vstride; i++) vs[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < S * vh; i++) { bt[i] = (rand() % 100) / 100.0f; ge[i] = 0.9f + (rand() % 100) / 1000.0f; }
    for (size_t i = 0; i < ns; i++) s0[i] = (rand() % 200 - 100) / 1000.0f;
    memcpy(sg, s0, ns * 4); memcpy(sc, s0, ns * 4);
    double t0 = now();
    if (!coli_vk_dn_recur(og, sg, qn, kn, vs, vstride, voff, bt, ge, S, vh, vk, kdim, vdim)) { printf("dn_recur failed\n"); return 1; }
    double ms = (now() - t0) * 1000;
    int rep = vh / vk;
    float kvl[512], dl[512];
    for (int h = 0; h < vh; h++) {
        float *Sh = sc + (size_t)h * kdim * vdim;
        for (int s = 0; s < S; s++) {
            const float *kh = kn + ((size_t)s * vk + h / rep) * kdim, *qh = qn + ((size_t)s * vk + h / rep) * kdim;
            const float *vd = vs + (size_t)s * vstride + voff + (size_t)h * vdim;
            float egh = ge[s * vh + h], b = bt[s * vh + h];
            for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;
            for (int v = 0; v < vdim; v++) kvl[v] = 0.f;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) kvl[v] += kh[k] * Sh[k * vdim + v];
            for (int v = 0; v < vdim; v++) dl[v] = (vd[v] - kvl[v]) * b;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) Sh[k * vdim + v] += kh[k] * dl[v];
            float *ov = oc + ((size_t)s * vh + h) * vdim;
            for (int v = 0; v < vdim; v++) ov[v] = 0.f;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) ov[v] += qh[k] * Sh[k * vdim + v];
        }
    }
    double rl2, rmax, sl2, smax; size_t nf, nfs;
    coop_err(og, oc, no, &rl2, &rmax, &nf);
    coop_err(sg, sc, ns, &sl2, &smax, &nfs);
    printf("DNREC S=%4d vh=%d vk=%d kdim=%d vdim=%d | out relL2 %.2e max %.2e | state relL2 %.2e%s | %.2f ms (incl. upload)\n",
           S, vh, vk, kdim, vdim, rl2, rmax, sl2, (nf || nfs) ? " NONFINITE" : "", ms);
    free(qn); free(kn); free(vs); free(bt); free(ge); free(s0); free(sg); free(sc); free(og); free(oc);
    return rl2 > 1e-4 || rmax > 1e-3 || sl2 > 1e-4 || nf || nfs;
}

/* Whole DeltaNet block against a CPU chain that mirrors qwen36.c deltanet_phased:
 * int8 projections (cpu_ref), conv with ring, l2norm (double), beta/g, delta rule,
 * gated RMSNorm, out_proj; also the carried ring and state. */
static float dnb_softplus(float z) { return z > 20.f ? z : log1pf(expf(z)); }
static int run_dn_block_case(int S, int H, int vh, int vk, int kdim, int vdim, int convk) {
    if (!G.pipe_dc) { printf("dn_block shaders not loaded\n"); return 1; }
    int ktot = vk * kdim, value_dim = vh * vdim, conv_dim = 2 * ktot + value_dim, pd = conv_dim + value_dim, km = convk - 1;
    float scale = 1.f / sqrtf((float)kdim), eps = 1e-6f;
    uint8_t *wp = malloc((size_t)pd * H), *wo = malloc((size_t)H * value_dim);
    float *sp = malloc((size_t)pd * 4), *so = malloc((size_t)H * 4);
    for (size_t i = 0; i < (size_t)pd * H; i++) wp[i] = rand() & 0xff;
    for (size_t i = 0; i < (size_t)H * value_dim; i++) wo[i] = rand() & 0xff;
    for (int i = 0; i < pd; i++) sp[i] = 0.0005f + (rand() % 100) / 200000.0f;
    for (int i = 0; i < H; i++) so[i] = 0.0005f + (rand() % 100) / 200000.0f;
    ColiVkTensor *tp = NULL, *to = NULL;
    if (!coli_vk_tensor_ensure(&tp, wp, sp, 1, H, pd, 0) || !coli_vk_tensor_ensure(&to, wo, so, 1, value_dim, H, 0)) { printf("upload failed\n"); return 1; }
    float *x = malloc((size_t)S * H * 4), *ba = malloc((size_t)S * 2 * vh * 4), *cw = malloc((size_t)conv_dim * convk * 4);
    float *par = malloc((size_t)2 * vh * 4), *nw = malloc((size_t)vdim * 4);
    float *ring0 = malloc((size_t)conv_dim * km * 4), *st0 = malloc((size_t)vh * kdim * vdim * 4);
    for (int i = 0; i < S * H; i++) x[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < S * 2 * vh; i++) ba[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < conv_dim * convk; i++) cw[i] = (rand() % 200 - 100) / 200.0f;
    for (int i = 0; i < 2 * vh; i++) par[i] = (rand() % 200 - 100) / 200.0f;
    for (int i = 0; i < vdim; i++) nw[i] = 0.5f + (rand() % 100) / 100.0f;
    for (int i = 0; i < conv_dim * km; i++) ring0[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < vh * kdim * vdim; i++) st0[i] = (rand() % 200 - 100) / 1000.0f;
    float *ringg = malloc((size_t)conv_dim * km * 4), *stg = malloc((size_t)vh * kdim * vdim * 4), *yg = malloc((size_t)S * H * 4);
    memcpy(ringg, ring0, (size_t)conv_dim * km * 4); memcpy(stg, st0, (size_t)vh * kdim * vdim * 4);
    double t0 = now();
    if (!coli_vk_dn_block(tp, to, x, ba, cw, par, nw, ringg, stg, S, H, conv_dim, convk, vh, vk, kdim, vdim, eps, scale, yg)) { printf("dn_block failed\n"); return 1; }
    double ms = (now() - t0) * 1000;
    /* CPU chain */
    float *qz = malloc((size_t)S * pd * 4), *conv = malloc((size_t)S * conv_dim * 4), *ov = malloc((size_t)S * value_dim * 4);
    float *orr = malloc((size_t)S * value_dim * 4), *yc = malloc((size_t)S * H * 4);
    float *ringc = malloc((size_t)conv_dim * km * 4), *stc = malloc((size_t)vh * kdim * vdim * 4);
    memcpy(ringc, ring0, (size_t)conv_dim * km * 4); memcpy(stc, st0, (size_t)vh * kdim * vdim * 4);
    cpu_ref(qz, x, wp, sp, 1, S, H, pd);
    for (int s = 0; s < S; s++) for (int cc = 0; cc < conv_dim; cc++) {
        float acc = 0.f;
        for (int kk = 0; kk < km; kk++) { int src = s - km + kk; acc += cw[cc * convk + kk] * (src >= 0 ? qz[(size_t)src * pd + cc] : ringc[cc * km + src + km]); }
        acc += cw[cc * convk + km] * qz[(size_t)s * pd + cc];
        conv[(size_t)s * conv_dim + cc] = acc / (1.f + expf(-acc));
    }
    for (int cc = 0; cc < conv_dim; cc++) for (int kk = 0; kk < km; kk++) { int src = S - km + kk;
        if (src >= 0) ringc[cc * km + kk] = qz[(size_t)src * pd + cc]; else ringc[cc * km + kk] = ring0[cc * km + src + km]; }
    int rep = vh / vk; float qh[512], kh[512], kvl[512], dl[512];
    for (int h = 0; h < vh; h++) {
        float *Sh = stc + (size_t)h * kdim * vdim;
        for (int s = 0; s < S; s++) {
            const float *cv = conv + (size_t)s * conv_dim; int j = h / rep;
            memcpy(qh, cv + j * kdim, kdim * 4); memcpy(kh, cv + ktot + j * kdim, kdim * 4);
            double sq = 1e-6, sk = 1e-6; for (int d = 0; d < kdim; d++) { sq += (double)qh[d] * qh[d]; sk += (double)kh[d] * kh[d]; }
            for (int d = 0; d < kdim; d++) { qh[d] = (float)(qh[d] / sqrt(sq) * scale); kh[d] = (float)(kh[d] / sqrt(sk)); }
            float bt = 1.f / (1.f + expf(-ba[s * 2 * vh + h]));
            float eg = expf(-expf(par[h]) * dnb_softplus(ba[s * 2 * vh + vh + h] + par[vh + h]));
            const float *vd = cv + 2 * ktot + h * vdim;
            for (int t = 0; t < kdim * vdim; t++) Sh[t] *= eg;
            for (int v = 0; v < vdim; v++) kvl[v] = 0.f;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) kvl[v] += kh[k] * Sh[k * vdim + v];
            for (int v = 0; v < vdim; v++) dl[v] = (vd[v] - kvl[v]) * bt;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) Sh[k * vdim + v] += kh[k] * dl[v];
            float *o = ov + (size_t)s * value_dim + h * vdim;
            for (int v = 0; v < vdim; v++) o[v] = 0.f;
            for (int k = 0; k < kdim; k++) for (int v = 0; v < vdim; v++) o[v] += qh[k] * Sh[k * vdim + v];
        }
    }
    for (int s = 0; s < S; s++) for (int h = 0; h < vh; h++) {
        const float *o = ov + (size_t)s * value_dim + h * vdim, *z = qz + (size_t)s * pd + conv_dim + h * vdim;
        double m2 = 0; for (int d = 0; d < vdim; d++) m2 += (double)o[d] * o[d];
        float r = 1.f / sqrtf((float)(m2 / vdim) + eps);
        for (int d = 0; d < vdim; d++) orr[(size_t)s * value_dim + h * vdim + d] = o[d] * r * nw[d] * (z[d] / (1.f + expf(-z[d])));
    }
    cpu_ref(yc, orr, wo, so, 1, S, value_dim, H);
    double yl2, ymax, sl2, smax, rl2, rmax; size_t n1, n2, n3;
    coop_err(yg, yc, (size_t)S * H, &yl2, &ymax, &n1);
    coop_err(stg, stc, (size_t)vh * kdim * vdim, &sl2, &smax, &n2);
    coop_err(ringg, ringc, (size_t)conv_dim * km, &rl2, &rmax, &n3);
    printf("DNBLOCK S=%4d H=%d vh=%d vk=%d kdim=%d vdim=%d convk=%d | y relL2 %.2e max %.2e | state %.2e | ring %.2e%s | %.2f ms\n",
           S, H, vh, vk, kdim, vdim, convk, yl2, ymax, sl2, rl2, (n1 || n2 || n3) ? " NONFINITE" : "", ms);
    coli_vk_tensor_free(tp); coli_vk_tensor_free(to);
    free(wp); free(wo); free(sp); free(so); free(x); free(ba); free(cw); free(par); free(nw); free(ring0); free(st0);
    free(ringg); free(stg); free(yg); free(qz); free(conv); free(ov); free(orr); free(yc); free(ringc); free(stc);
    /* f16 tiles in the projections when S reaches the tile threshold: bound like the coop tests */
    return yl2 > 5e-3 || ymax > 3e-2 || sl2 > 5e-3 || rl2 > 1e-3 || n1 || n2 || n3;
}

/* Whole attention layer against a CPU chain that mirrors qwen36.c attention():
 * int8 q/k/v (cpu_ref), split query|gate, RMSNorm (1+w), partial RoPE, KV cache
 * with a cached prefix, causal softmax attention (GQA), sigmoid gate, o_proj. */
static void ab_rms(float *x, const float *w, int n, float eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * r * (1.0f + w[i]);
}
static void ab_rope(float *x, int pos, int rot, float theta) {
    int h = rot / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rot), ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j + h]; x[j] = a * cs - b * sn; x[j + h] = b * cs + a * sn;
    }
}
static int run_attn_block_case(int S, int D, int H, int KV, int hd, int rotary, int pos_base) {
    if (!G.pipe_ab) { printf("attn_block shaders not loaded\n"); return 1; }
    int qdim = 2 * hd, ldt = pos_base + S + 5, qo = H * qdim, ko = KV * hd;
    float eps = 1e-6f, theta = 1e7f, scale = 1.f / sqrtf((float)hd);
    uint8_t *wq = malloc((size_t)qo * D), *wk = malloc((size_t)ko * D), *wv = malloc((size_t)ko * D), *wo = malloc((size_t)D * H * hd);
    float *sq = malloc(qo * 4), *sk = malloc(ko * 4), *sv = malloc(ko * 4), *so = malloc(D * 4);
    for (size_t i = 0; i < (size_t)qo * D; i++) wq[i] = rand() & 0xff;
    for (size_t i = 0; i < (size_t)ko * D; i++) { wk[i] = rand() & 0xff; wv[i] = rand() & 0xff; }
    for (size_t i = 0; i < (size_t)D * H * hd; i++) wo[i] = rand() & 0xff;
    for (int i = 0; i < qo; i++) sq[i] = 0.0004f + (rand() % 100) / 300000.0f;
    for (int i = 0; i < ko; i++) { sk[i] = 0.0004f + (rand() % 100) / 300000.0f; sv[i] = 0.0004f + (rand() % 100) / 300000.0f; }
    for (int i = 0; i < D; i++) so[i] = 0.0004f + (rand() % 100) / 300000.0f;
    ColiVkTensor *tq = NULL, *tk = NULL, *tv = NULL, *to = NULL;
    if (!coli_vk_tensor_ensure(&tq, wq, sq, 1, D, qo, 0) || !coli_vk_tensor_ensure(&tk, wk, sk, 1, D, ko, 0) ||
        !coli_vk_tensor_ensure(&tv, wv, sv, 1, D, ko, 0) || !coli_vk_tensor_ensure(&to, wo, so, 1, H * hd, D, 0)) { printf("upload failed\n"); return 1; }
    float *x = malloc((size_t)S * D * 4), *qn = malloc(hd * 4), *kn = malloc(hd * 4);
    float *Kg = calloc((size_t)KV * ldt * hd, 4), *Vg = calloc((size_t)KV * ldt * hd, 4);
    for (int i = 0; i < S * D; i++) x[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < hd; i++) { qn[i] = (rand() % 100) / 200.0f - 0.25f; kn[i] = (rand() % 100) / 200.0f - 0.25f; }
    for (size_t i = 0; i < (size_t)KV * ldt * hd; i++) { Kg[i] = (rand() % 200 - 100) / 100.0f; Vg[i] = (rand() % 200 - 100) / 100.0f; }
    float *Kc = malloc((size_t)KV * ldt * hd * 4), *Vc = malloc((size_t)KV * ldt * hd * 4);
    memcpy(Kc, Kg, (size_t)KV * ldt * hd * 4); memcpy(Vc, Vg, (size_t)KV * ldt * hd * 4);
    float *yg = malloc((size_t)S * D * 4);
    double t0 = now();
    if (!coli_vk_attn_block(tq, tk, tv, to, x, qn, kn, Kg, Vg, ldt, S, D, H, KV, hd, qdim, rotary, pos_base, eps, theta, scale, yg)) { printf("attn_block failed\n"); return 1; }
    double ms = (now() - t0) * 1000;
    float *q = malloc((size_t)S * qo * 4), *k = malloc((size_t)S * ko * 4), *v = malloc((size_t)S * ko * 4);
    cpu_ref(q, x, wq, sq, 1, S, D, qo); cpu_ref(k, x, wk, sk, 1, S, D, ko); cpu_ref(v, x, wv, sv, 1, S, D, ko);
    float *qb = malloc((size_t)S * H * hd * 4), *gb = malloc((size_t)S * H * hd * 4), *ag = malloc((size_t)S * H * hd * 4), *yc = malloc((size_t)S * D * 4);
    for (int s2 = 0; s2 < S; s2++) {
        for (int h = 0; h < H; h++) {
            memcpy(qb + ((size_t)s2 * H + h) * hd, q + (size_t)s2 * qo + h * qdim, hd * 4);
            memcpy(gb + ((size_t)s2 * H + h) * hd, q + (size_t)s2 * qo + h * qdim + hd, hd * 4);
            ab_rms(qb + ((size_t)s2 * H + h) * hd, qn, hd, eps); ab_rope(qb + ((size_t)s2 * H + h) * hd, pos_base + s2, rotary, theta);
        }
        for (int g = 0; g < KV; g++) {
            float *kr = k + (size_t)s2 * ko + g * hd;
            ab_rms(kr, kn, hd, eps); ab_rope(kr, pos_base + s2, rotary, theta);
            memcpy(Kc + ((size_t)g * ldt + pos_base + s2) * hd, kr, hd * 4);
            memcpy(Vc + ((size_t)g * ldt + pos_base + s2) * hd, v + (size_t)s2 * ko + g * hd, hd * 4);
        }
    }
    double *sc = malloc((size_t)(pos_base + S) * sizeof(double));
    for (int s2 = 0; s2 < S; s2++) for (int h = 0; h < H; h++) {
        int g = h / (H / KV), qp = pos_base + s2; const float *qv = qb + ((size_t)s2 * H + h) * hd;
        double mx = -1e300;
        for (int t = 0; t <= qp; t++) {
            double a = 0; const float *kr = Kc + ((size_t)g * ldt + t) * hd;
            for (int d = 0; d < hd; d++) a += (double)qv[d] * kr[d];
            sc[t] = a * scale; if (sc[t] > mx) mx = sc[t];
        }
        double z = 0; for (int t = 0; t <= qp; t++) { sc[t] = exp(sc[t] - mx); z += sc[t]; }
        for (int d = 0; d < hd; d++) { double a = 0; for (int t = 0; t <= qp; t++) a += sc[t] / z * Vc[((size_t)g * ldt + t) * hd + d];
            size_t o = ((size_t)s2 * H + h) * hd + d; ag[o] = (float)a * (1.f / (1.f + expf(-gb[o]))); }
    }
    cpu_ref(yc, ag, wo, so, 1, S, H * hd, D);
    double yl2, ymax, kl2, kmax, vl2, vmax; size_t n1, n2, n3;
    coop_err(yg, yc, (size_t)S * D, &yl2, &ymax, &n1);
    coop_err(Kg + (size_t)pos_base * hd, Kc + (size_t)pos_base * hd, (size_t)S * hd, &kl2, &kmax, &n2);   /* kv head 0 rows */
    coop_err(Vg + (size_t)pos_base * hd, Vc + (size_t)pos_base * hd, (size_t)S * hd, &vl2, &vmax, &n3);
    printf("ATTNBLOCK S=%4d D=%d H=%d KV=%d hd=%d rot=%d pos_base=%d | y relL2 %.2e max %.2e | K %.2e | V %.2e%s | %.2f ms\n",
           S, D, H, KV, hd, rotary, pos_base, yl2, ymax, kl2, vl2, (n1 || n2 || n3) ? " NONFINITE" : "", ms);
    coli_vk_tensor_free(tq); coli_vk_tensor_free(tk); coli_vk_tensor_free(tv); coli_vk_tensor_free(to);
    free(wq); free(wk); free(wv); free(wo); free(sq); free(sk); free(sv); free(so); free(x); free(qn); free(kn);
    free(Kg); free(Vg); free(Kc); free(Vc); free(yg); free(q); free(k); free(v); free(qb); free(gb); free(ag); free(yc); free(sc);
    return yl2 > 5e-3 || ymax > 3e-2 || kl2 > 5e-3 || vl2 > 5e-3 || n1 || n2 || n3;
}

/* Prefill attention core against a CPU reference that mirrors qwen36.c's
 * attention(): scores, softmax (max-subtracted), weighted V sum, causal with a
 * prefix of pos_base cached keys, grouped-query heads. */
static int run_attn_prefill_case(int S, int H, int KV, int hd, int pos_base, int ldt) {
    if (!G.pipe_ap) { printf("attn_prefill shader not loaded\n"); return 1; }
    int nt = pos_base + S;
    float *q = malloc((size_t)S * H * hd * 4), *K = malloc((size_t)KV * ldt * hd * 4), *V = malloc((size_t)KV * ldt * hd * 4);
    float *cg = malloc((size_t)S * H * hd * 4), *cc = malloc((size_t)S * H * hd * 4);
    double *sc = malloc((size_t)nt * sizeof(double));
    for (int i = 0; i < S * H * hd; i++) q[i] = (rand() % 200 - 100) / 100.0f;
    for (size_t i = 0; i < (size_t)KV * ldt * hd; i++) { K[i] = (rand() % 200 - 100) / 100.0f; V[i] = (rand() % 200 - 100) / 100.0f; }
    float scale = 1.f / sqrtf((float)hd);
    double t0 = now();
    int ok = coli_vk_attn_prefill(cg, q, K, V, ldt, S, H, KV, hd, pos_base, scale);
    double ms = (now() - t0) * 1000;
    if (!ok) { printf("attn_prefill failed\n"); return 1; }
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int g = h / (H / KV), qpos = pos_base + s;
        const float *qv = q + ((size_t)s * H + h) * hd;
        double mx = -1e300;
        for (int t = 0; t <= qpos; t++) {
            const float *kr = K + ((size_t)g * ldt + t) * hd;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)qv[d] * kr[d];
            sc[t] = a * scale; if (sc[t] > mx) mx = sc[t];
        }
        double z = 0; for (int t = 0; t <= qpos; t++) { sc[t] = exp(sc[t] - mx); z += sc[t]; }
        float *o = cc + ((size_t)s * H + h) * hd;
        for (int d = 0; d < hd; d++) {
            double a = 0; for (int t = 0; t <= qpos; t++) a += sc[t] / z * V[((size_t)g * ldt + t) * hd + d];
            o[d] = (float)a;
        }
    }
    double rl2, rmax; size_t nf;
    coop_err(cg, cc, (size_t)S * H * hd, &rl2, &rmax, &nf);
    printf("ATTN prefill S=%4d H=%d KV=%d hd=%d pos_base=%d | relL2 %.2e max %.2e%s | %.2f ms (incl. upload)\n",
           S, H, KV, hd, pos_base, rl2, rmax, nf ? " NONFINITE" : "", ms);
    free(q); free(K); free(V); free(cg); free(cc); free(sc);
    /* f16 operands on the matrix units (attn_flash): P and Q/K/V carry 11 bits */
    int fl = af_pipeline(H, KV, hd) != VK_NULL_HANDLE;
    return fl ? (rl2 > 3e-3 || rmax > 3e-2 || nf) : (rl2 > 1e-5 || rmax > 1e-4 || nf);
}

/* Expert group with several routed rows per expert (prefill / MTP verify): rows[c]
 * cycles 1..maxrows, so one group mixes one-row and multi-row dispatches. */
static int run_mr_expert_group(int fmt, int D, int I, int K, int maxrows, int iters) {
    if (!G.mr) { printf("multi-row shaders not loaded\n"); return 1; }
    if (K > 64) K = 64;
    size_t gu_rb = ref_rowbytes(fmt, D), gu_sc = ref_scales(fmt, D, I);
    size_t d_rb  = ref_rowbytes(fmt, I), d_sc  = ref_scales(fmt, I, D);
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    int rows[64], total = 0;
    for (int c = 0; c < K; c++) { rows[c] = 1 + (c * 5) % maxrows; total += rows[c]; }
    float *x = malloc((size_t)total*D*4), *ym = malloc((size_t)total*D*4);
    float *y1 = malloc((size_t)total*D*4), *yc = malloc((size_t)total*D*4);
    float *tmp = malloc((size_t)(D > I ? D : I) * 4), *hid = malloc((size_t)I*4);
    for (int i = 0; i < total*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc(gu_sc*4); hus[c] = malloc(gu_sc*4); hds[c] = malloc(d_sc*4);
        /* int8 draws span +-128 (int4: +-8): scale them down so the gate/up/down chain
         * stays in the range where a 3e-3 fp32-vs-double bound is meaningful. */
        float sf = fmt == 1 ? 1.0f / 16 : 1.0f;
        for (size_t o = 0; o < gu_sc; o++) { hgs[c][o] = sf*(0.01f+(rand()%100)/10000.0f); hus[c][o] = sf*(0.01f+(rand()%100)/10000.0f); }
        for (size_t o = 0; o < d_sc; o++) hds[c][o] = sf*(0.01f+(rand()%100)/10000.0f);
        coli_vk_tensor_ensure(&tg[c], hgw[c], hgs[c], fmt, D, I, g_ref_gs);
        coli_vk_tensor_ensure(&tu[c], huw[c], hus[c], fmt, D, I, g_ref_gs);
        coli_vk_tensor_ensure(&td[c], hdw[c], hds[c], fmt, I, D, g_ref_gs);
    }
    int mr = G.mr, bad = 0;
    if (!coli_vk_expert_group(tg, tu, td, rows, K, ym, x)) { printf("expert_group failed\n"); return 1; }
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, ym, x);
    double ms_mr = (now() - t0) * 1000 / iters;
    G.mr = 0;
    coli_vk_expert_group(tg, tu, td, rows, K, y1, x);
    t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, y1, x);
    double ms_1 = (now() - t0) * 1000 / iters;
    G.mr = mr;
    for (int c = 0, off = 0; c < K; off += rows[c], c++)
        for (int r = 0; r < rows[c]; r++) {
            const float *xr = x + (size_t)(off + r)*D;
            for (int o = 0; o < I; o++) {
                float gt = (float)ref_dot(xr, hgw[c]+(size_t)o*gu_rb, hgs[c], o, fmt, D);
                float ut = (float)ref_dot(xr, huw[c]+(size_t)o*gu_rb, hus[c], o, fmt, D);
                hid[o] = (gt/(1.0f+expf(-gt)))*ut;
            }
            for (int d = 0; d < D; d++)
                yc[(size_t)(off + r)*D + d] = (float)ref_dot(hid, hdw[c]+(size_t)d*d_rb, hds[c], d, fmt, I);
        }
    double maxrel = 0;
    for (int i = 0; i < total*D; i++) { double e = fabs(ym[i]-yc[i]);
        if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
    size_t d1 = mr_diff(ym, y1, (size_t)total*D);
    printf("MR%d expert_group fmt=%d D=%d I=%d %2d experts, %3d rows (1..%d each) | maxrel=%.4g | vs one-row %s | %.3f ms (one-row %.3f, %.2fx)\n",
           mr, fmt, D, I, K, total, maxrows, maxrel, d1 == 0 ? "bit-identical" : "DIFFERS", ms_mr, ms_1, ms_1 / ms_mr);
    /* 5e-3: the two-reduction chain (see run_expert_group); Lavapipe measured 3.06e-3 on
     * the GLM shape with output bit-identical to the one-row shaders. */
    bad = maxrel > 5e-3 || d1 != 0;
    free(x); free(ym); free(y1); free(yc); free(tmp); free(hid);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    return bad;
}

int main(int argc, char **argv) {
    char spvbuf[1024];
    const char *spv = argc > 1 ? argv[1] : coli_vk_default_spv(spvbuf, sizeof spvbuf);
    printf("shader: %s\n", spv);
    if (!coli_vk_init(spv)) { printf("vk init failed\n"); return 1; }
    printf("weights: %s\n", coli_vk_staged() ? "staged device-local" : "mapped host-visible");
    srand(1234);
    int bad = 0;
    if (getenv("VK_COOP_BENCH")) {   /* tuning aid: the tiled shaders on two prefill shapes only */
        g_ref_gs = 64;
        bad |= run_coop_case(1, 1024, 2048, 8192, 10);   /* DeltaNet qkv, whole prompt */
        bad |= run_coop_case(4, 32, 2048, 512, 20);      /* one Qwen3.6 expert gate/up-shaped matmul */
        if (getenv("VK_COOP_BENCH")[0] == '2') {       /* attention q | k | v and o_proj shapes, whole prompt */
            bad |= run_coop_case(1, 1011, 2048, 8192, 10);
            bad |= run_coop_case(1, 1011, 2048, 512, 10);
            bad |= run_coop_case(1, 1011, 4096, 2048, 10);
            bad |= run_coop_case(1, 1011, 2048, 12288, 10);
        }
        bad |= run_coop_expert_group(4, 2048, 512, 64, 60, 10, 0.f);
        printf(bad ? "FAIL\n" : "PASS\n");
        coli_vk_shutdown();
        return bad;
    }
    if (getenv("VK_EG_BENCH")) {
        if (!G.ts_on) printf("VK_EG_BENCH needs COLI_VK_TS=1\n");
        bad = run_expert_prefill_bench(1011, 8, 256, 2048, 512, 20);
        coli_vk_shutdown(); return bad;
    }
    if (getenv("VK_ATTN_TEST")) {   /* attention cores only (attn_flash / attn_prefill + whole layer) */
        if (G.pipe_ap) {
            bad |= run_attn_prefill_case(1, 16, 2, 256, 0, 64);
            bad |= run_attn_prefill_case(37, 16, 2, 256, 0, 64);
            bad |= run_attn_prefill_case(20, 16, 2, 256, 30, 64);
            bad |= run_attn_prefill_case(33, 8, 8, 128, 5, 40);
            bad |= run_attn_prefill_case(70, 16, 4, 128, 100, 200);  /* GQA 4, prefix across blocks */
            bad |= run_attn_prefill_case(1011, 16, 2, 256, 0, 1100);
            bad |= run_attn_prefill_case(1011, 16, 2, 256, 0, 1100);  /* warm timing */
        }
        if (G.pipe_ab && G.pipe_ap) {
            bad |= run_attn_block_case(37, 256, 4, 2, 64, 32, 10);
            bad |= run_attn_block_case(300, 2048, 16, 2, 256, 64, 0);
            bad |= run_attn_block_case(64, 2048, 16, 2, 256, 64, 40);
        }
        printf(bad ? "FAIL\n" : "PASS\n");
        coli_vk_shutdown(); return bad;
    }
    if (getenv("VK_PREFILL_TEST")) {
        bad = run_expert_prefill_tests();
        printf(bad ? "FAIL\n" : "PASS\n");
        coli_vk_shutdown(); return bad;
    }
    if (getenv("VK_MR_SWEEP")) {   /* tuning aid: multi-row vs one-row across shapes only */
        G.coop = 0;
        g_mr_any = 1;
        int sh[][2] = {{2048,512},{512,2048},{2048,1536},{1536,2048},{4096,1024},{6144,1536},{6144,2048},{2048,6144},{16384,6144}};
        int ss[] = {2, 4, 8, 16, 32};
        g_ref_gs = 64;
        for (unsigned k = 0; k < sizeof sh / sizeof sh[0]; k++)
            for (unsigned j = 0; j < sizeof ss / sizeof ss[0]; j++)
                bad |= run_mr_case(4, ss[j], sh[k][0], sh[k][1], 20);
        printf(bad ? "FAIL\n" : "PASS\n");
        coli_vk_shutdown();
        return bad;
    }
    /* COLI_VK_TEST_BALLAST=N: allocate N idle 4 MB device buffers before benching.
     * Probes whether per-submit cost scales with the process's ALLOCATION COUNT
     * (RADV/amdgpu CS buffer-list accounting) independent of bytes — the suspected
     * mechanism behind decode attention degrading with expert-tier size even with
     * VRAM to spare (7.9s @2.6k BOs -> 15.6s @4.3k with 2.9 GB free). */
    {
        int nb = getenv("COLI_VK_TEST_BALLAST") ? atoi(getenv("COLI_VK_TEST_BALLAST")) : 0;
        for (int i = 0; i < nb; i++) {
            VkBuffer b; VkDeviceMemory m;
            if (!alloc_hostvis_mt(4u << 20, &b, &m, NULL, G.memtype)) { printf("ballast stop at %d\n", i); break; }
        }
        if (nb) printf("ballast: %d x 4 MB idle allocations\n", nb);
    }
    bad |= run_case(1, 1, 6144, 1536, 50);   // int8 expert gate/up shape (S=1 decode)
    bad |= run_case(2, 1, 6144, 1536, 50);   // int4 expert
    bad |= run_case(1, 1, 1536, 6144, 50);   // down proj shape
    bad |= run_case(2, 8, 6144, 1536, 20);   // prefill/MTP batch
    bad |= run_case(1, 1, 512, 512, 100);    // small
    bad |= run_case(2, 1, 16384, 6144, 20);  // o_proj shape: I > xsh capacity (unstaged path)
    /* int3-g64 (fmt=5): per-group scales through every dense shape incl. tail groups */
    bad |= run_case(5, 1, 6144, 2048, 50);   // int3 expert gate/up shape
    bad |= run_case(5, 1, 2048, 6144, 50);   // int3 down proj shape
    bad |= run_case(5, 8, 6144, 2048, 20);   // int3 batch
    bad |= run_case(5, 1, 100, 64, 20);      // partial tail group (I%64 != 0)
    bad |= run_case(5, 1, 16384, 6144, 20);  // int3 o_proj shape (unstaged path)
    /* Batched (amortized) throughput on the int4 expert shapes — the real expert-tier pattern. */
    {
        int I = 6144, O = 2048;   /* our gate/up dims */
        float *x = malloc((size_t)I * 4); uint8_t *w = malloc((size_t)(I + 1) / 2 * O); float *sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        float *y = malloc((size_t)O * 4);
        ColiVkTensor *t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);   /* bind */
        printf("BATCHED int4 S=1 6144->2048 (our gate/up): %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
        I = 2048; O = 6144;   /* our down dims */
        x = malloc((size_t)I * 4); w = malloc((size_t)(I + 1) / 2 * O); sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        y = malloc((size_t)O * 4);
        t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);
        printf("BATCHED int4 S=1 2048->6144 (our down):    %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
    }
    /* FUSED gate+up: correctness + batched throughput (vs 2x separate gate/up). */
    {
        int D = 6144, I = 2048;
        bad |= run_gate_up(2, 1, D, I);
        bad |= run_gate_up(5, 1, D, I);   // int3-g64 fused pair (per-group scales)
        size_t rb = (size_t)(D + 1) / 2;
        float *x = malloc((size_t)D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
        float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4), *h = malloc((size_t)I*4);
        for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (int o = 0; o < I; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        ColiVkTensor *tg = NULL, *tu = NULL; coli_vk_gate_up(&tg, &tu, h, x, gw, gs, uw, us, 2, 1, D, I, 0);
        printf("BATCHED fused gate_up int4 6144->2048:     %.4f ms (N=64, SAME expert = L2-cached)\n",
               bench_gu_batched(tg, x, 2, 1, D, I, 64));
        coli_vk_tensor_free(tg); coli_vk_tensor_free(tu); free(x); free(gw); free(uw); free(gs); free(us); free(h);
    }
    /* FAIR: cycle 8 distinct experts (VRAM reads, not L2) — matches ROCm expert_group. */
    printf("FAIR fused gate_up int4 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(2, 6144, 2048, 8, 8));
    printf("FAIR fused gate_up int3 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(5, 6144, 2048, 8, 8));
    /* FULL expert_group: the real primitive. Sweep K to see if per-expert cost is fixed
     * per-call overhead (drops with K) or per-dispatch (constant). */
    bad |= run_expert_group(2, 6144, 2048, 1);
    bad |= run_expert_group(2, 6144, 2048, 8);
    bad |= run_expert_group(2, 6144, 2048, 32);
    /* int3-g64 expert group: correctness + the 0.86x-bytes throughput question.
     * count=1 included — it is the SHARED-expert path shape in the engine. */
    bad |= run_qprep(1, 1, 6144, 1536, 576, 16384);   /* GLM q_a/kv_a/q_b decode shapes */
    bad |= run_qprep(1, 11, 6144, 1536, 576, 16384);  /* prefill batch */
    bad |= run_qprep(1, 2, 6144, 1536, 576, 16384);   /* S=2 (MTP verify) */
    bad |= run_qprep(2, 1, 6144, 1536, 576, 16384);   /* int4 dense variant */
    /* fmt=4 grouped int4 (#298 semantics), gs=64 across real shapes + gs=32 sanity */
    g_ref_gs = 64;
    bad |= run_case(4, 1, 6144, 2048, 50);
    bad |= run_case(4, 1, 2048, 6144, 50);
    bad |= run_case(4, 8, 6144, 1536, 20);
    bad |= run_case(4, 1, 16384, 6144, 10);
    g_ref_gs = 32;
    bad |= run_case(4, 1, 6144, 2048, 20);
    g_ref_gs = 64;
    bad |= run_expert_group(4, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 1);
    bad |= run_expert_group(5, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 32);
    /* Fused same-input matmul pair (the q_a + kv_a prologue pattern), int4 AND int3. */
    for (int pi = 0; pi < 3; pi++) {
        int pf = pi == 0 ? 2 : pi == 1 ? 5 : 4;
        int I = 6144, O1 = 2048, O2 = 576, S = 1;
        size_t rb = ref_rowbytes(pf, I);
        size_t n1 = ref_scales(pf, I, O1), n2 = ref_scales(pf, I, O2);
        uint8_t *w1 = malloc(rb * O1), *w2 = malloc(rb * O2);
        float *s1 = malloc(n1 * 4), *s2 = malloc(n2 * 4), *x = malloc((size_t)I * 4);
        float *y1 = malloc((size_t)O1 * 4), *y2 = malloc((size_t)O2 * 4);
        float *c1 = malloc((size_t)O1 * 4), *c2 = malloc((size_t)O2 * 4);
        for (size_t i = 0; i < rb * (size_t)O1; i++) w1[i] = rand() & 0xff;
        for (size_t i = 0; i < rb * (size_t)O2; i++) w2[i] = rand() & 0xff;
        for (size_t o = 0; o < n1; o++) s1[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (size_t o = 0; o < n2; o++) s2[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        ColiVkTensor *t1 = NULL, *t2 = NULL;
        if (!coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs)) {
            printf("matmul_pair fmt=%d failed\n", pf); bad = 1;
        } else {
            cpu_ref(c1, x, w1, s1, pf, S, I, O1); cpu_ref(c2, x, w2, s2, pf, S, I, O2);
            double mr = 0;
            for (int i = 0; i < O1; i++) { double e = fabs(y1[i]-c1[i]); if (fabs(c1[i])>1e-2) { double r=e/fabs(c1[i]); if (r>mr) mr=r; } }
            for (int i = 0; i < O2; i++) { double e = fabs(y2[i]-c2[i]); if (fabs(c2[i])>1e-2) { double r=e/fabs(c2[i]); if (r>mr) mr=r; } }
            double t0 = now();
            for (int k = 0; k < 30; k++) coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs);
            printf("matmul_pair fmt=%d 6144->(2048,576)   | maxrel=%.4g | %.3f ms/pair-call\n", pf, mr, (now()-t0)*1000/30);
            bad |= mr > 1e-3;
        }
        coli_vk_tensor_free(t1); coli_vk_tensor_free(t2);
        free(w1); free(w2); free(s1); free(s2); free(x); free(y1); free(y2); free(c1); free(c2);
    }
    /* MLA absorb attention core (GLM decode shape + window/causal/int8 variants). */
    if (G.pipe_att) {
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 300, 0);    // GLM-5.2 decode
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 17, 300, 1);   // kv_start window
        bad |= run_absorb(2, 2, 64, 192, 64, 256, 512, 0, 300, 2);    // S=2 causal (MTP verify)
        bad |= run_absorb(1, 1, 8, 128, 32, 64, 256, 0, 64, 3);       // int8, odd dims
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 2000, 4);   // long context
        bad |= run_absorb(5, 1, 64, 192, 64, 256, 512, 0, 300, 5);    // int3-g64 kv_b + o
        bad |= run_absorb(5, 2, 64, 192, 64, 256, 512, 17, 300, 6);   // int3 S=2 causal + window
        bad |= run_absorb(4, 1, 64, 192, 64, 256, 512, 0, 300, 7);    // grouped-int4 kv_b + o
        bad |= run_absorb(4, 2, 64, 192, 64, 256, 512, 17, 300, 8);   // fmt=4 S=2 causal + window
    }
    /* Multi-row shaders: formats 1/2/4, tails (S not a multiple of MR), the staged
     * (MR*I <= 8192) and unstaged paths, Qwen3.6 and GLM expert shapes. */
    int coop_saved = G.coop;
    G.coop = 0;          /* the multi-row block tests the multi-row shaders, not the tiles */
    if (G.mr) {
        g_ref_gs = 64;
        g_mr_any = 1;   /* exercise the multi-row shaders on every shape, not just the tuned ones */
        int fm[3] = {1, 2, 4}, ss[6] = {2, 3, 4, 5, 8, 33};
        for (int f = 0; f < 3; f++) for (int k = 0; k < 6; k++) {
            bad |= run_mr_case(fm[f], ss[k], 2048, 512, 20);    /* Qwen3.6 gate/up, staged */
            bad |= run_mr_case(fm[f], ss[k], 512, 2048, 20);    /* Qwen3.6 down, staged */
        }
        bad |= run_mr_case(2, 8, 6144, 1536, 10);               /* GLM gate/up, unstaged */
        bad |= run_mr_case(4, 11, 6144, 2048, 10);              /* unstaged, tail */
        bad |= run_mr_case(2, 3, 16384, 6144, 5);               /* o_proj shape */
        bad |= run_mr_expert_group(4, 2048, 512, 8, 9, 20);     /* Qwen3.6 experts, gs64 */
        bad |= run_mr_expert_group(4, 2048, 512, 32, 33, 10);   /* prefill-like: up to 33 rows */
        bad |= run_mr_expert_group(1, 2048, 512, 8, 9, 20);     /* int8 promotion */
        bad |= run_mr_expert_group(2, 6144, 2048, 8, 9, 10);    /* GLM shape, unstaged gate_up */
        bad |= run_mr_case(1, 5, 100, 77, 5);                   /* odd dims */
        bad |= run_mr_case(2, 6, 99, 33, 5);
        bad |= run_mr_case(4, 7, 200, 50, 5);                   /* gs64 tail group */
        bad |= run_mr_gate_up(4, 6, 2048, 512, 0.f);
        bad |= run_mr_gate_up(4, 6, 2048, 512, 0.5f);           /* SwiGLU clamp (glm53) */
        bad |= run_mr_gate_up(2, 9, 6144, 1536, 0.5f);          /* clamp, unstaged */
        bad |= run_mr_cache(4, 2048, 512);
        bad |= run_mr_cache(2, 6144, 1536);
        g_mr_any = 0;
    } else printf("multi-row shaders not loaded (COLI_VK_MR<=1 or missing .spv): skipped\n");
    G.coop = coop_saved;
    /* Tiled cooperative-matrix shaders: int4 per-row and gs64, Qwen3.6 and GLM
     * shapes, token counts around the 32-row tile (tails), the fused gate+up with
     * and without the SwiGLU clamp, and whole expert groups. */
    if (G.coop) {
        g_ref_gs = 64;
        int ss[6] = {1, 16, 31, 32, 33, 100};
        for (int k = 0; k < 6; k++) {
            bad |= run_coop_case(4, ss[k], 2048, 512, 10);   /* Qwen3.6 gate/up */
            bad |= run_coop_case(4, ss[k], 512, 2048, 10);   /* Qwen3.6 down */
            bad |= run_coop_case(2, ss[k], 6144, 1536, 5);   /* GLM gate/up, per-row int4 */
        }
        bad |= run_coop_case(4, 64, 6144, 2048, 5);
        bad |= run_coop_case(1, 100, 2048, 8192, 3);           /* dense-i8 trunk: DeltaNet qkv (Qwen3.6) */
        bad |= run_coop_case(1, 33, 4096, 2048, 5);            /* DeltaNet out_proj, tail */
        bad |= run_coop_case(1, 256, 2048, 4096, 3);
        bad |= run_coop_expert_group(4, 2048, 512, 32, 40, 10, 0.f);
        bad |= run_coop_expert_group(4, 2048, 512, 64, 100, 5, 0.f);
        bad |= run_coop_expert_group(4, 2048, 512, 16, 40, 5, 0.5f);   /* clamp */
        bad |= run_coop_expert_group(2, 6144, 2048, 8, 40, 3, 0.f);
    } else printf("coopmat shaders not loaded (no VK_KHR_cooperative_matrix f16 config, COLI_VK_COOP=0 or missing .spv): skipped\n");
    bad |= run_expert_prefill_tests();
    if (G.pipe_ab && G.pipe_ap) {
        bad |= run_attn_block_case(37, 256, 4, 2, 64, 32, 10);   /* small, cached prefix */
        bad |= run_attn_block_case(300, 2048, 16, 2, 256, 64, 0);/* Qwen3.6 shapes */
        bad |= run_attn_block_case(64, 2048, 16, 2, 256, 64, 40);/* Qwen3.6, prefix */
    } else printf("attn_block shaders not loaded: skipped\n");
    if (G.pipe_dc && G.pipe_dr) {
        bad |= run_dn_block_case(37, 256, 4, 2, 64, 64, 4);      /* small, GEMV projections */
        bad |= run_dn_block_case(300, 2048, 32, 16, 128, 128, 4);/* Qwen3.6 shapes, tiled projections */
    } else printf("dn_block shaders not loaded: skipped\n");
    if (G.pipe_dr) {
        bad |= run_dn_recur_case(1, 32, 16, 128, 128);           /* one token */
        bad |= run_dn_recur_case(37, 32, 16, 128, 128);          /* Qwen3.6 shape */
        bad |= run_dn_recur_case(20, 8, 2, 64, 30);              /* rep 4, odd vdim */
        bad |= run_dn_recur_case(1011, 32, 16, 128, 128);        /* the 1011-token prompt */
    } else printf("dn_recur shader not loaded: skipped\n");
    if (G.pipe_ap) {
        bad |= run_attn_prefill_case(1, 16, 2, 256, 0, 64);      /* one token */
        bad |= run_attn_prefill_case(37, 16, 2, 256, 0, 64);     /* Qwen3.6 shape, odd S */
        bad |= run_attn_prefill_case(20, 16, 2, 256, 30, 64);    /* cached prefix (pos_base > 0) */
        bad |= run_attn_prefill_case(33, 8, 8, 128, 5, 40);      /* no GQA, other head dim */
        bad |= run_attn_prefill_case(1011, 16, 2, 256, 0, 1100); /* the 1011-token prompt */
    } else printf("attn_prefill shader not loaded: skipped\n");
    printf(bad ? "FAIL\n" : "PASS\n");
    coli_vk_shutdown();
    return bad;
}
#endif
