/* qwen36_mtp.h — multi-token prediction (MTP) for the Qwen3.6 engine, included
 * by qwen36.c. QWEN_MTP=<gguf> loads the model's single trained MTP block
 * (llama.cpp's nextn layer: e.g. ggml-org/Qwen3.6-35B-A3B-GGUF
 * mtp-Qwen3.6-35B-A3B-Q4_0.gguf) and greedy generation drafts one token ahead:
 *
 *   MTP row at position p (as llama.cpp's qwen35moe graph_mtp):
 *     x = eh_proj( [ enorm(embed(tok_p)) , hnorm(h_{p-1}) ] )      h = main hidden AFTER output_norm
 *     x += attn(attn_norm(x))            (gated attention, own KV cache)
 *     x += moe(post_attention_norm(x))   (256 routed experts top-8 + gated shared expert)
 *     logits_{p+1} = lm_head(shared_head_norm(x))
 *   The head and the token embedding are the main model's (the file's copies
 *   are the same matrices in Q4_0).
 *
 *   Loop: a = argmax(main logits); draft d from the MTP row (a, h); the main
 *   model runs [a, d] as one S = 2 step. argmax at a == d -> both tokens stand;
 *   otherwise d is dropped: KV length back, DeltaNet state and conv ring back to
 *   the copies captured right after token a (deltanet_phased, g_dn_cap_*).
 *   Greedy output is the plain greedy output (verified token by token).  With
 *   sampling (QWEN_TEMP > 0) every emitted token is drawn from the main model's
 *   logits (cli_pick) and the draft stands only if it equals that draw, so the
 *   output follows the model's distribution exactly; the draft itself stays argmax.
 *
 * The MTP block runs on the CPU (dense int8 / int4 like the trunk, experts
 * grouped per expert so the prompt catch-up is a few batched GEMMs). */

#include <stdio.h>

/* ---------------- minimal GGUF v3 reader (tensor infos + f32/f16/Q4_0 data) ---------------- */
typedef struct { char name[96]; int nd; uint64_t ne[4]; uint32_t type; uint64_t off; } GgTensor;
typedef struct { FILE *f; uint64_t base; int n; GgTensor *t; } Gguf;

static int gg_skip(FILE *f, uint32_t t) { return gguf_skip_val(f, t); }
static int gg_open(Gguf *g, const char *path) {
    memset(g, 0, sizeof *g);
    g->f = fopen(path, "rb");
    if (!g->f) return 0;
    uint32_t magic, ver, align = 32; uint64_t nt, nkv;
    if (fread(&magic, 4, 1, g->f) != 1 || magic != 0x46554747u || fread(&ver, 4, 1, g->f) != 1 ||
        fread(&nt, 8, 1, g->f) != 1 || fread(&nkv, 8, 1, g->f) != 1) return 0;
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t kl; char key[256]; uint32_t t;
        if (fread(&kl, 8, 1, g->f) != 1 || kl >= sizeof key || fread(key, 1, kl, g->f) != kl) return 0;
        key[kl] = 0;
        if (fread(&t, 4, 1, g->f) != 1) return 0;
        if (!strcmp(key, "general.alignment") && t == 4) { if (fread(&align, 4, 1, g->f) != 1) return 0; }
        else if (!gg_skip(g->f, t)) return 0;
    }
    g->t = calloc(nt, sizeof(GgTensor)); g->n = (int)nt;
    for (int i = 0; i < g->n; i++) {
        uint64_t nl; GgTensor *x = &g->t[i];
        if (fread(&nl, 8, 1, g->f) != 1 || nl >= sizeof x->name || fread(x->name, 1, nl, g->f) != nl) return 0;
        x->name[nl] = 0;
        if (fread(&x->nd, 4, 1, g->f) != 1 || x->nd > 4) return 0;
        for (int k = 0; k < 4; k++) x->ne[k] = 1;
        for (int k = 0; k < x->nd; k++) if (fread(&x->ne[k], 8, 1, g->f) != 1) return 0;
        if (fread(&x->type, 4, 1, g->f) != 1 || fread(&x->off, 8, 1, g->f) != 1) return 0;
    }
    long pos = ftell(g->f);
    g->base = ((uint64_t)pos + align - 1) / align * align;
    return 1;
}
static GgTensor *gg_find(Gguf *g, const char *name) {
    for (int i = 0; i < g->n; i++) if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}
static float gg_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) { if (!m) u = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; u = s | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
/* elements [e0, e0 + n) of tensor t (row-major, ne0 fastest) as f32; Q4_0 needs e0, n % 32 == 0 */
static float *gg_f32(Gguf *g, GgTensor *t, uint64_t e0, uint64_t n) {
    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;
    if (t->type == 0) {           /* F32 */
        fseek(g->f, (long)(g->base + t->off + e0 * 4), SEEK_SET);
        if (fread(out, 4, n, g->f) != n) { free(out); return NULL; }
    } else if (t->type == 1) {    /* F16 */
        uint16_t *h = malloc(n * 2);
        fseek(g->f, (long)(g->base + t->off + e0 * 2), SEEK_SET);
        if (!h || fread(h, 2, n, g->f) != n) { free(h); free(out); return NULL; }
        for (uint64_t i = 0; i < n; i++) out[i] = gg_h2f(h[i]);
        free(h);
    } else if (t->type == 2) {    /* Q4_0: blocks of 32 = f16 d + 16 bytes, x = (nibble - 8) * d */
        if (e0 % 32 || n % 32) { free(out); return NULL; }
        uint64_t nb = n / 32; uint8_t *raw = malloc(nb * 18);
        fseek(g->f, (long)(g->base + t->off + e0 / 32 * 18), SEEK_SET);
        if (!raw || fread(raw, 18, nb, g->f) != nb) { free(raw); free(out); return NULL; }
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < (int64_t)nb; b++) {
            const uint8_t *blk = raw + b * 18; uint16_t hd; memcpy(&hd, blk, 2);
            float d = gg_h2f(hd); float *o = out + b * 32;
            for (int j = 0; j < 16; j++) {
                o[j] = (float)((int)(blk[2 + j] & 15) - 8) * d;
                o[j + 16] = (float)((int)(blk[2 + j] >> 4) - 8) * d;
            }
        }
        free(raw);
    } else { free(out); return NULL; }
    return out;
}

/* ---------------- the MTP block ---------------- */
typedef struct {
    int on, E, K, I;
    Layer L;                         /* q,k,v,o (QW), qn, kn, in_ln (attn_norm), post_ln -- colibri's (1 + w) norms */
    QW eh;                           /* [D x 2D] */
    float *enorm, *hnorm, *head_norm;
    float *router;                   /* [E][D] f32 */
    float *sh_gate;                  /* [D] f32 */
    QW sh_g, sh_u, sh_d;
    QW *eg, *eu, *ed;                /* [E] */
    float *Kc, *Vc; int stride;      /* own KV cache [KV][stride][hd] */
    float **Kp, **Vp;                /* pointer tables: the model's layers + this block */
    long drafts, accepted;
} Mtp;
static Mtp g_mtp;

/* GGUF norms hold the effective multiplier (HF's 1 + w); colibri's rmsnorm_row adds the 1 */
static float *mtp_norm(Gguf *g, const char *name, int n) {
    GgTensor *t = gg_find(g, name);
    if (!t || (int)(t->ne[0] * t->ne[1]) != n) return NULL;
    float *v = gg_f32(g, t, 0, (uint64_t)n);
    if (v) for (int i = 0; i < n; i++) v[i] -= 1.f;
    return v;
}
static int mtp_qw(Gguf *g, const char *name, int I, int O, uint64_t e0, const char *tag, QW *out) {
    GgTensor *t = gg_find(g, name);
    if (!t || (int)t->ne[0] != I) return 0;
    float *w = gg_f32(g, t, e0, (uint64_t)I * O);
    if (!w) return 0;
    out->w = NULL; out->q4 = NULL; out->sg = NULL; out->ng = 0;
    qw_quantize(w, I, O, tag, out);
    free(w);
    return 1;
}
static int mtp_load(Model *m, const char *path) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, I = c->inter, Ish = c->shared_inter;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    Gguf g;
    if (!gg_open(&g, path)) { fprintf(stderr, "[mtp] %s: not a readable GGUF\n", path); return 0; }
    char nm[128]; int L = -1;
    for (int i = 0; i < g.n; i++) if (!strncmp(g.t[i].name, "blk.", 4) && strstr(g.t[i].name, ".nextn.eh_proj")) L = atoi(g.t[i].name + 4);
    if (L < 0) { fprintf(stderr, "[mtp] %s: no nextn block\n", path); return 0; }
    double t0 = now_s();
    Mtp *p = &g_mtp; memset(p, 0, sizeof *p);
    p->E = E; p->K = c->topk; p->I = I;
    int ok = 1;
#define N(s) (snprintf(nm, sizeof nm, "blk.%d.%s", L, s), nm)
    ok &= !!(p->enorm = mtp_norm(&g, N("nextn.enorm.weight"), D));
    ok &= !!(p->hnorm = mtp_norm(&g, N("nextn.hnorm.weight"), D));
    ok &= !!(p->head_norm = mtp_norm(&g, N("nextn.shared_head_norm.weight"), D));
    ok &= !!(p->L.in_ln = mtp_norm(&g, N("attn_norm.weight"), D));
    ok &= !!(p->L.post_ln = mtp_norm(&g, N("post_attention_norm.weight"), D));
    ok &= !!(p->L.qn = mtp_norm(&g, N("attn_q_norm.weight"), hd));
    ok &= !!(p->L.kn = mtp_norm(&g, N("attn_k_norm.weight"), kvd));
    ok &= mtp_qw(&g, N("nextn.eh_proj.weight"), 2 * D, D, 0, "mtp", &p->eh);
    ok &= mtp_qw(&g, N("attn_q.weight"), D, H * qdim, 0, "mtp", &p->L.q);
    ok &= mtp_qw(&g, N("attn_k.weight"), D, KV * kvd, 0, "mtp", &p->L.k);
    ok &= mtp_qw(&g, N("attn_v.weight"), D, KV * kvd, 0, "mtp", &p->L.v);
    ok &= mtp_qw(&g, N("attn_output.weight"), H * hd, D, 0, "mtp", &p->L.o);
    ok &= mtp_qw(&g, N("ffn_gate_shexp.weight"), D, Ish, 0, "mtp", &p->sh_g);
    ok &= mtp_qw(&g, N("ffn_up_shexp.weight"), D, Ish, 0, "mtp", &p->sh_u);
    ok &= mtp_qw(&g, N("ffn_down_shexp.weight"), Ish, D, 0, "mtp", &p->sh_d);
    GgTensor *tr = gg_find(&g, N("ffn_gate_inp.weight")), *ts = gg_find(&g, N("ffn_gate_inp_shexp.weight"));
    ok &= tr && ts && (p->router = gg_f32(&g, tr, 0, (uint64_t)E * D)) && (p->sh_gate = gg_f32(&g, ts, 0, (uint64_t)D));
    p->eg = calloc(E, sizeof(QW)); p->eu = calloc(E, sizeof(QW)); p->ed = calloc(E, sizeof(QW));
    for (int e = 0; ok && e < E; e++) {
        ok &= mtp_qw(&g, N("ffn_gate_exps.weight"), D, I, (uint64_t)e * I * D, "mtpexp", &p->eg[e]);
        ok &= mtp_qw(&g, N("ffn_up_exps.weight"), D, I, (uint64_t)e * I * D, "mtpexp", &p->eu[e]);
        ok &= mtp_qw(&g, N("ffn_down_exps.weight"), I, D, (uint64_t)e * D * I, "mtpexp", &p->ed[e]);
    }
#undef N
    fclose(g.f); free(g.t);
    if (!ok) { fprintf(stderr, "[mtp] %s: block %d incomplete or shapes differ from the model\n", path, L); return 0; }
    p->on = 1;
    fprintf(stderr, "[mtp] block %d from %s: %d experts, loaded in %.1f s\n", L, path, E, now_s() - t0);
    return 1;
}

/* own KV rows with the model's row stride (attention() indexes K[layer] by max_t) */
static void mtp_kv_ensure(Model *m) {
    Cfg *c = &m->c; Mtp *p = &g_mtp;
    if (p->stride != m->max_t || !p->Kc) {
        free(p->Kc); free(p->Vc);
        size_t n = (size_t)c->kv_heads * m->max_t * c->k_head_dim;
        p->Kc = calloc(n, sizeof(float)); p->Vc = calloc(n, sizeof(float)); p->stride = m->max_t;
    }
    free(p->Kp); free(p->Vp);
    p->Kp = malloc(sizeof(float *) * (c->n_layers + 1)); p->Vp = malloc(sizeof(float *) * (c->n_layers + 1));
    for (int i = 0; i < c->n_layers; i++) { p->Kp[i] = m->K[i]; p->Vp[i] = m->V[i]; }
    p->Kp[c->n_layers] = p->Kc; p->Vp[c->n_layers] = p->Vc;
}

static float sigm(float z) { return 1.f / (1.f + expf(-z)); }
/* routed experts (softmax top-K, renormalised) + sigmoid-gated shared expert,
 * rows grouped per expert so a prompt catch-up runs as batched GEMMs */
static void mtp_moe(const float *x, int R, float *out) {
    Mtp *p = &g_mtp; int D = g_mtp.L.q.I, E = p->E, K = p->K, I = p->I;
    int *idx = malloc(sizeof(int) * (size_t)R * K); float *val = falloc((int64_t)R * K);
    float *lg = falloc((int64_t)R * E);
    #pragma omp parallel for schedule(static) if(R > 1)
    for (int r = 0; r < R; r++) {
        const float *xr = x + (int64_t)r * D; float *l = lg + (int64_t)r * E;
        for (int e = 0; e < E; e++) { const float *w = p->router + (int64_t)e * D; float a = 0; for (int i = 0; i < D; i++) a += xr[i] * w[i]; l[e] = a; }
        softmax_row(l, E);
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) { int taken = 0; for (int j = 0; j < k; j++) taken |= idx[r * K + j] == e; if (!taken && l[e] > bv) { bv = l[e]; best = e; } }
            idx[r * K + k] = best; val[r * K + k] = bv;
        }
        float sm = 0; for (int k = 0; k < K; k++) sm += val[r * K + k];
        for (int k = 0; k < K; k++) val[r * K + k] /= sm;
    }
    memset(out, 0, sizeof(float) * (size_t)R * D);
    /* shared expert on all rows */
    {
        int Ish = p->sh_g.O;
        float *g = falloc((int64_t)R * Ish), *u = falloc((int64_t)R * Ish), *y = falloc((int64_t)R * D);
        matmul_d(g, x, &p->sh_g, R, D, Ish); matmul_d(u, x, &p->sh_u, R, D, Ish);
        for (int64_t i = 0; i < (int64_t)R * Ish; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
        matmul_d(y, g, &p->sh_d, R, Ish, D);
        for (int r = 0; r < R; r++) {
            const float *xr = x + (int64_t)r * D; float a = 0; for (int i = 0; i < D; i++) a += xr[i] * p->sh_gate[i];
            float sg = sigm(a); for (int i = 0; i < D; i++) out[(int64_t)r * D + i] += sg * y[(int64_t)r * D + i];
        }
        free(g); free(u); free(y);
    }
    /* routed experts, grouped */
    int *rows = malloc(sizeof(int) * (size_t)R * K);
    float *xe = falloc((int64_t)R * K * D), *ge = falloc((int64_t)R * K * I), *ue = falloc((int64_t)R * K * I), *ye = falloc((int64_t)R * K * D);
    for (int e = 0; e < E; e++) {
        int n = 0;
        for (int i = 0; i < R * K; i++) if (idx[i] == e) rows[n++] = i;
        if (!n) continue;
        for (int j = 0; j < n; j++) memcpy(xe + (int64_t)j * D, x + (int64_t)(rows[j] / K) * D, sizeof(float) * D);
        matmul_d(ge, xe, &p->eg[e], n, D, I); matmul_d(ue, xe, &p->eu[e], n, D, I);
        for (int64_t i = 0; i < (int64_t)n * I; i++) ge[i] = ge[i] / (1.f + expf(-ge[i])) * ue[i];
        matmul_d(ye, ge, &p->ed[e], n, I, D);
        for (int j = 0; j < n; j++) {
            float w = val[rows[j]]; float *o = out + (int64_t)(rows[j] / K) * D; const float *yr = ye + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += w * yr[i];
        }
    }
    free(idx); free(val); free(lg); free(rows); free(xe); free(ge); free(ue); free(ye);
}

/* R MTP rows at positions pos0 .. pos0+R-1: tokens tok[r] with main hidden hprev[r]
 * (= h at position pos0+r-1). logits (vocab) of the last row when logits != NULL. */
static void mtp_forward(Model *m, const int *tok, const float *hprev, int R, int pos0, float *logits) {
    Cfg *c = &m->c; Mtp *p = &g_mtp; int D = c->hidden;
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D), *tmp = falloc((int64_t)R * D);
    memset(cat, 0, sizeof(float) * (size_t)R * 2 * D);   /* (every row is written below; silences -Wmaybe-uninitialized) */
    for (int r = 0; r < R; r++) {
        rmsnorm_row(cat + (int64_t)r * 2 * D, m->embed + (int64_t)tok[r] * D, p->enorm, D, c->eps);
        rmsnorm_row(cat + (int64_t)r * 2 * D + D, hprev + (int64_t)r * D, p->hnorm, D, c->eps);
    }
    matmul_d(x, cat, &p->eh, R, 2 * D, D);
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.in_ln, D, c->eps);
    float **K0 = m->K, **V0 = m->V;
    m->K = p->Kp; m->V = p->Vp;
    attention(m, &p->L, c->n_layers, nrm, R, pos0, tmp);
    m->K = K0; m->V = V0;
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.post_ln, D, c->eps);
    mtp_moe(nrm, R, tmp);
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    if (logits) {
        float *last = falloc(D);
        rmsnorm_row(last, x + (int64_t)(R - 1) * D, p->head_norm, D, c->eps);
        if (!qt_lmhead_matmul(logits, last, D, c->vocab)) matmul_d(logits, last, &m->lm_head, 1, D, c->vocab);
        free(last);
    }
    free(cat); free(x); free(nrm); free(tmp);
}

static int argmax_v(const float *l, int n) { int b = 0; for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i; return b; }

/* greedy generation with one MTP draft per main step; returns tokens generated */
static int generate_mtp(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c; int D = c->hidden, V = c->vocab;
    if (np + n_new + 2 > QWEN36_ATTN_MAX_CTX) { fprintf(stderr, "[ctx] prompt too long for MTP\n"); exit(1); }
    m->max_t = np + n_new + 2;
    reset_recurrent(m); ensure_kv(m); m->kv_len = 0;
    mtp_kv_ensure(m);
    int eos_ids[4], n_eos = getenv("STOP_EOS") && getenv("STOP_EOS")[0] == '1' ? serve_eos_ids(eos_ids, 4) : 0;
    /* capture buffers for the DeltaNet state right after the verify step's first token */
    float **cr = calloc(c->n_layers, sizeof(float *)), **cg = calloc(c->n_layers, sizeof(float *));
    for (int i = 0; i < c->n_layers; i++) if (!c->is_attn[i]) {
        cr[i] = falloc((int64_t)c->dn_vheads * c->dn_kdim * c->dn_vdim);
        cg[i] = falloc((int64_t)c->dn_conv_dim * (c->dn_convk - 1));
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *H = falloc((int64_t)np * D);
    g_step_hid = H;
    float *logit = step(m, prompt, np, 0);
    g_step_hid = NULL;
    if (np > 1) mtp_forward(m, prompt + 1, H, np - 1, 1, NULL);        /* MTP catch-up over the prompt */
    float *hprev = falloc(D); memcpy(hprev, H + (int64_t)(np - 1) * D, sizeof(float) * D); free(H);
    int a = cli_pick(logit, V); free(logit);
    if (g_ttft < 0) g_ttft = now_s() - g_gen_t0;
    int len = np, made = 0, pend = -1;
    double t_draft = 0, t_verify = 0;           /* pend: accepted draft whose MTP row is still owed */
    float *hpend = falloc(D), *dlog = falloc(V), *H2 = falloc(2 * (int64_t)D), *L2 = falloc(2 * (int64_t)V);
    float *hrows = falloc(2 * (int64_t)D);
    for (;;) {
        int is_eos = 0; for (int e = 0; e < n_eos; e++) is_eos |= a == eos_ids[e];
        if (is_eos) break;
        out[len] = a; made++;
        if (g_stream) { stream_token(a); fflush(stdout); }
        if (made >= n_new) { len++; break; }
        /* draft: MTP rows [ (pend, hpend) @ len-1 ,] (a, hprev) @ len */
        int toks[2], R = 0;
        if (pend >= 0) { toks[R] = pend; memcpy(hrows + (int64_t)R * D, hpend, sizeof(float) * D); R++; }
        toks[R] = a; memcpy(hrows + (int64_t)R * D, hprev, sizeof(float) * D); R++;
        double tq0 = now_s();
        mtp_forward(m, toks, hrows, R, len - (R - 1), dlog);
        double tq1 = now_s(); t_draft += tq1 - tq0;
        int d = argmax_v(dlog, V);
        /* QWEN_MTP_TEST_REJECT=1: every draft deliberately wrong -- each step takes the
         * roll-back path, and the output must still be the plain greedy text */
        static int force_rej = -1;
        if (force_rej < 0) force_rej = getenv("QWEN_MTP_TEST_REJECT") && getenv("QWEN_MTP_TEST_REJECT")[0] == '1';
        if (force_rej) d = (d + 1) % V;
        g_mtp.drafts++;
        /* verify [a, d] at len, len + 1 */
        int pair[2] = {a, d};
        g_step_hid = H2; g_step_logits = L2; g_dn_cap_rec = cr; g_dn_cap_ring = cg; g_dn_cap_after = 0; g_dn_cap_n = 0;
        float *lg = step(m, pair, 2, len);
        t_verify += now_s() - tq1;
        g_step_hid = NULL; g_step_logits = NULL; g_dn_cap_after = -1;
        free(lg);
        int c1 = cli_pick(L2, V);
        if (c1 == d && !force_rej) {   /* forced test: never accept, even if d+1 hits */
            g_mtp.accepted++;
            int d_eos = 0; for (int e = 0; e < n_eos; e++) d_eos |= d == eos_ids[e];
            if (d_eos) { len++; break; }
            out[len + 1] = d; made++;
            if (g_stream) { stream_token(d); fflush(stdout); }
            if (made >= n_new) { len += 2; break; }
            pend = d; memcpy(hpend, H2, sizeof(float) * D);
            memcpy(hprev, H2 + D, sizeof(float) * D);
            a = cli_pick(L2 + V, V);
            len += 2;
        } else {
            /* drop d: state back to right after a */
            int ndn = 0; for (int i = 0; i < c->n_layers; i++) ndn += !c->is_attn[i];
            if (g_dn_cap_n != ndn) { fprintf(stderr, "[mtp] DeltaNet state capture missed (%d of %d layers)\n", g_dn_cap_n, ndn); exit(1); }
            for (int i = 0; i < c->n_layers; i++) if (!c->is_attn[i] && !qt_dn_rollback(i)) {   /* GPU: swap in its capture */
                memcpy(m->DN_rec[i], cr[i], sizeof(float) * (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim);
                memcpy(m->DN_conv[i], cg[i], sizeof(float) * (size_t)c->dn_conv_dim * (c->dn_convk - 1));
                qt_dn_drop(i);
            }
            m->kv_len = len + 1;
            if (m->kvp.len > len + 1) m->kvp.len = len + 1;
            pend = -1;
            memcpy(hprev, H2, sizeof(float) * D);
            a = c1;
            len += 1;
        }
    }
    for (int i = 0; i < c->n_layers; i++) { free(cr[i]); free(cg[i]); }
    free(cr); free(cg); free(hprev); free(hpend); free(dlog); free(H2); free(L2); free(hrows);
    fprintf(stderr, "[mtp] drafts %ld, accepted %ld (%.1f %%), %.2f tokens per main step\n",
            g_mtp.drafts, g_mtp.accepted, g_mtp.drafts ? 100.0 * g_mtp.accepted / g_mtp.drafts : 0.0,
            g_mtp.drafts ? (double)(g_mtp.drafts + g_mtp.accepted) / g_mtp.drafts : 1.0);
    if (g_mtp.drafts) fprintf(stderr, "[mtp] per main step: draft %.2f ms, verify (S=2) %.2f ms\n",
                              1e3 * t_draft / g_mtp.drafts, 1e3 * t_verify / g_mtp.drafts);
    return made;
}
