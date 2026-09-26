/* The Vulkan shim of the qwen36 tier, on a fake backend, no device.
 *
 * test_qwen36_tier_vk proves the NUMERICS against a real Vulkan device and
 * skips everywhere else -- which is every CI runner but the Lavapipe job.
 * This test covers the shim's CONTROL FLOW on the fake backend in
 * tests/qwen36_fake_vulkan.h, the way the CUDA tier tests do on
 * qwen36_fake_cuda.h, so it runs in `make check` and under ASan/UBSan on
 * Linux, macOS and Windows:
 *
 *   - single device: COLI_GPUS is ignored, ndev is 1;
 *   - the budget reads VK_EXPERT_GB and the driver's memory-budget figure,
 *     and falls back to 4 GB when the driver has none;
 *   - uploads reach coli_vk_tensor_ensure in the right format (fmt 4 with
 *     the group size for int4-gs64) and at the right byte count;
 *   - fill once: with the budget full, a hotter non-resident expert never
 *     evicts a resident one (the Vulkan arena never reclaims a freed slice);
 *   - issue/take go through the tier's own ybuf and accumulate correctly, and
 *     a failed take skips those experts instead of crashing;
 *   - fp8 streaming and injected resident-trunk upload failures fall back
 *     to the CPU path. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_vulkan.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 1, NE = 4, D = 64, IH = 32, TOPK = 2, GS = 64 };

static unsigned char g4[NE][D * IH / 2], u4[NE][D * IH / 2], d4[NE][D * IH / 2];
static float sc[NE][2 * (IH * ((D + GS - 1) / GS)) + D * ((IH + GS - 1) / GS)];
static void note_block(int eid) {
    qt_note_block(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + G.sc_gu, sc[eid] + 2 * G.sc_gu);
}
static void note(int eid) {
    qt_note(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + G.sc_gu, sc[eid] + 2 * G.sc_gu);
}
static int resident_count(void) {
    int n = 0;
    for (int eid = 0; eid < NE; eid++) n += qt_is_resident(0, eid);
    return n;
}

int main(void) {
    for (int eid = 0; eid < NE; eid++) {
        memset(g4[eid], (unsigned char)(eid + 1), sizeof g4[eid]);
        memset(u4[eid], (unsigned char)(eid + 2), sizeof u4[eid]);
        memset(d4[eid], (unsigned char)(eid + 3), sizeof d4[eid]);
        for (size_t i = 0; i < sizeof sc[eid] / sizeof sc[eid][0]; i++) sc[eid][i] = 1.0f;
    }

    /* ---- 1. init: single device, budget from VK_EXPERT_GB ---------------- */
    setenv("COLI_VULKAN", "1", 1);
    setenv("COLI_GPUS", "0,1", 1);                /* CUDA syntax; Vulkan must ignore it */
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_LMHEAD_GPU", "0", 1);            /* asks for the trunk on the device (section 6) */
    /* Two experts and not three. The tier charges an expert at the allocator's
     * granularity (dev_alloc_footprint), six allocations per expert; at this
     * geometry every one of them rounds to the 8 KiB minimum. */
    size_t per_expert = 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint(8);
    check(per_expert == 6 * 8192, "test geometry should cost six minimum allocations per expert");
    setenv("VK_EXPERT_GB", "0.0001", 1);          /* 107374 bytes: two fit, three do not */

    if (!qt_init(NL, NE, D, IH, NE, TOPK, GS, 1 /* int4 */)) {
        printf("  FAIL: the Vulkan tier should start on the fake backend\n");
        return 1;
    }
    check(fake_vk_inits == 1, "coli_vk_init should be called exactly once");
    check(G.ndev == 1 && G.dev[0] == 0, "the Vulkan tier is single-device: COLI_GPUS=0,1 must not add a second one");
    check(G.wfmt == 4, "an int4 container is fmt 4 on Vulkan as on CUDA");
    check(G.exp_bytes == per_expert, "exp_bytes should follow the allocator footprint");
    check(G.budget[0] >= 2 * per_expert && G.budget[0] < 3 * per_expert,
          "VK_EXPERT_GB=0.0001 should admit exactly two experts");
    check(!strcmp(qt_backend_name(), "Vulkan"), "qt_backend_name should say Vulkan");

    /* ---- 2. uploads: format, group size, bytes, and the budget cut-off ---- */
    for (int eid = 0; eid < 3; eid++) note_block(eid);
    qt_fill_wait();
    check(resident_count() == 2, "two experts should be resident after the budget filled");
    check(qt_is_resident(0, 0) && qt_is_resident(0, 1), "the first two noted experts should be the residents");
    check(!qt_is_resident(0, 2), "the third expert should not fit the two-expert budget");
    check(fake_vk_uploads == 6, "two resident experts are six tensor uploads (gate, up, down each)");
    check(fake_vk_last_fmt == 4 && fake_vk_last_gs == GS, "int4-gs64 should reach the backend as fmt 4 with grp=64");
    check(fake_vk_last_bytes == (size_t)D * IH / 2, "packed int4 is half a byte per element");
    check(fake_vk_frees == 0, "nothing should be freed while filling");

    /* ---- 3. fill once: a hot non-resident never evicts a resident -------- */
    for (int i = 0; i < 200; i++) note(2);        /* eid 2 gets far hotter than 0 and 1 */
    fake_vk_take_D = D;
    float x[D]; for (int d = 0; d < D; d++) x[d] = 1.0f;
    float val1[1] = { 1.0f }; float out[D];
    int one[1] = { 0 };
    for (int i = 0; i < 64; i++) {                /* 64 ticks: on CUDA the LFRU pass runs every 16 */
        uint32_t m = qt_issue(0, one, 1, x);
        memset(out, 0, sizeof out);
        qt_take(m, val1, 1, out);
    }
    check(G.swaps == 0, "the Vulkan tier must never LFRU-swap (fill once)");
    check(qt_is_resident(0, 0) && qt_is_resident(0, 1) && !qt_is_resident(0, 2),
          "residency must not change after the fill, however hot a non-resident gets");
    check(fake_vk_frees == 0 && fake_vk_uploads == 6, "no tensor should be freed or re-uploaded after the fill");

    /* ---- 4. issue and take through ybuf ---------------------------------- */
    int two[2] = { 0, 1 };
    uint32_t mask = qt_issue(0, two, 2, x);
    check(mask == 3u, "both resident experts should be claimed by the GPU");
    check(fake_vk_last_issue_count == 2, "the two residents should issue as one two-row group");
    float val2[2] = { 0.5f, 2.0f };
    memset(out, 0, sizeof out);
    qt_take(mask, val2, 2, out);
    /* the fake fills row j with j+1: out = 0.5*1 + 2.0*2 = 4.5 in every lane */
    int acc_ok = 1;
    for (int d = 0; d < D; d++) if (out[d] != 4.5f) acc_ok = 0;
    check(acc_ok, "qt_take should weight each backend row by its routing value and sum them");

    int mixed[3] = { 0, 1, 2 };
    uint32_t mask3 = qt_issue(0, mixed, 3, x);
    check(mask3 == 3u, "a non-resident expert in the batch must be left to the CPU");
    memset(out, 0, sizeof out);
    qt_take(mask3, val2, 3, out);

    /* ---- 5. a failed take skips those experts, nothing else -------------- */
    fake_vk_take_ok = 0;
    mask = qt_issue(0, two, 2, x);
    memset(out, 0, sizeof out);
    qt_take(mask, val2, 2, out);
    int untouched = 1;
    for (int d = 0; d < D; d++) if (out[d] != 0.0f) untouched = 0;
    check(untouched, "when the backend's take fails the output must be left alone");
    check(G.is_cnt[0] == 0 && G.issue_open == 0, "a failed take must still close the group");
    fake_vk_take_ok = 1;

    /* Prefill: interleaved residents, misses, an all-miss token and failure.
     * First-seen expert order is 1,0, so their fake values are 1,2. */
    {
        int ids[6] = {1,2,0,1,2,2};
        float weights[6] = {.5f,1.f,.25f,.75f,1.f,1.f};
        float bx[3*D], by[3*D]; uint8_t done[6];
        for (int j = 0; j < 3*D; j++) bx[j] = 1.f+j/D;
        setenv("QT_PREFILL_GPU_REDUCE","0",1);
        check(!qt_batch_gpu_reduce(), "GPU reduction can be disabled");
        setenv("QT_PREFILL_GPU_REDUCE","1",1);
        check(qt_batch_gpu_reduce(), "GPU reduction is available on one Vulkan device");
        uint64_t hits = G.hits[0], misses = G.miss;
        check(qt_issue_batch_reduce(0,ids,3,2,bx,weights,by,done), "reduced batch succeeds");
        check(done[0] && !done[1] && done[2] && done[3] && !done[4] && !done[5], "done marks only resident pairs");
        int correct = 1;
        for (int d = 0; d < D; d++) if (by[d] != .5f || by[D+d] != 2.5f || by[2*D+d] != 0.f) correct = 0;
        check(correct, "gather order, weights and missing tokens reach the backend correctly");
        check(G.hits[0]-hits == 3 && G.miss-misses == 3, "batch statistics count pairs");
        check(!G.issue_open, "reduced batch releases the uploader guard");
        fake_vk_take_ok = 0;
        check(!qt_issue_batch_reduce(0,ids,3,2,bx,weights,by,done), "failed backend returns failure");
        check(!done[0] && !done[2] && !done[3] && !G.issue_open, "failure clears claims and releases guard");
        fake_vk_take_ok = 1;
        for (int p = 0; p < 6; p++) ids[p] = 2;
        int calls = fake_vk_prefill_calls;
        check(qt_issue_batch_reduce(0,ids,3,2,bx,weights,by,done), "all-miss batch succeeds with zeros");
        correct = 1;
        for (int j = 0; j < 3*D; j++) if (by[j] != 0.f) correct = 0;
        check(correct && calls == fake_vk_prefill_calls, "all-miss batch needs no GPU submit");
        check(!qt_issue_batch_reduce(0,ids,1,2,bx,weights,by,done), "decode stays on its old path");
    }

    /* ---- 6. injected resident trunk upload failures fall back to CPU ---- */
    check(G_lmh.dev_ok == 1, "COLI_LMHEAD_GPU=0 should have reached the lm_head gate (the fake rejects dense uploads below)");
    static int8_t lmq[16 * 8]; static float lms[8];
    int before = fake_vk_uploads;
    check(qt_lmhead_init(lmq, lms, 16, 8) == 0, "qt_lmhead_init must return 0 after upload failure");
    check(G_lmh.on == 0, "a failed lm_head upload must stay on the CPU");
    float ly[8], lx[16] = { 0 };
    check(qt_lmhead_matmul(ly, lx, 16, 8) == 0, "unplaced lm_head must return 0");
    check(fake_vk_uploads == before, "failed trunk upload must not create a tensor");

    /* Dense handles must also propagate the injected upload failure. */
    check(qt_dense_init(lmq, lms, 16, 8, 0) < 0, "qt_dense_init must report upload failure");
    check(qt_dense_count() == 0, "a refused dense handle must not be registered");
    float dy[8];
    check(qt_dense_matmul(0, dy, lx, 16, 8) == 0, "unplaced dense matmul must return 0");
    check(fake_vk_uploads == before, "failed dense upload must not create a tensor either");

    /* ---- 7. shutdown -------------------------------------------------- */
    /* every resident expert's three tensors go back through coli_vk_tensor_free
     * before the device does, or vkDestroyDevice reports them as leaked */
    int resident = 0;
    for (size_t i = 0; i < (size_t)G.nl * G.ne; i++) resident += G.slot[i].resident;
    int frees = fake_vk_frees;
    qt_shutdown();
    check(resident > 0 && fake_vk_frees - frees == 3 * resident,
          "shutdown must free every resident expert through the Vulkan backend");
    check(fake_vk_shutdowns == 1, "coli_vk_shutdown should be called exactly once");
    check(G.on == 0, "the tier should be off after shutdown");

    /* ---- 8. fp8 streaming is CUDA-only: refused at init ------------------ */
    static float lut[256];
    int inits = fake_vk_inits;
    check(qt_init_fp8(NL, NE, D, IH, 2, TOPK, lut) == 0, "qt_init_fp8 must be refused on Vulkan");
    check(G_fp8_stream == 0, "a refused fp8 init must not leave the streaming mode armed");
    check(fake_vk_inits == inits + 1, "the fp8 refusal should come from the decode-table publish, after the backend came up");

    /* ---- 9. no VK_EXT_memory_budget: auto budget falls back to 4 GB ------ */
    fake_vk_budget_known = 0;
    setenv("VK_EXPERT_GB", "auto", 1);
    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0 /* per-row */, 0 /* int8 */)) {
        printf("  FAIL: the tier should start without a memory-budget extension\n");
        return 1;
    }
    check(G.wfmt == 1, "an int8 container is fmt 1 on Vulkan as on CUDA");
    check(G.budget[0] == ((size_t)4 << 30) - ((size_t)1 << 30),
          "without VK_EXT_memory_budget the auto budget should be the 4 GB fallback minus 1 GB headroom");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_vk_fake: %d failures\n", fails); return 1; }
    printf("test_qwen36_tier_vk_fake: ok\n");
    return 0;
}
