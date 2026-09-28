#!/bin/bash
# Benchmark fuer das colibri-Issue (28.09.2026): je Arm 1 Aufwaermlauf (verworfen) + 3 Laeufe, Median.
# Ausgabe: results.tsv (arm, prompt, run, prompt_tokens, ttft_s, decode_tok_s, mtp_accept)
set -u
cd ~/bench/colibri-1338
if journalctl -k -b 0 | grep -q BO_VA; then echo "BO_VA"; exit 1; fi
OUT=~/bench/colibri-mr/issue; R=$OUT/results.tsv; : > $R
BASE="COLI_VULKAN=1 VK_EXPERT_GB=24 HEAT_FILE=out/heat24.bin QT_PREFILL_BATCH=1 OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 SNAP=$HOME/models/qwen36_i4 STOP_EOS=0"
I4="COLI_DENSE_BITS=4 COLI_IMATRIX=$HOME/bench/colibri-mr/agent-int4/imatrix_unsloth.gguf"
MTP="QWEN_MTP=$HOME/models/qwen36-mtp/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
PR=~/bench/colibri-mr/pr1338-build/c/qwen36; NEW=~/src/colibri-dev/c/qwen36
arm() {  # name binary "env" prompt ntok
  local name=$1 bin=$2 env=$3 pf=$4 n=$5
  for run in 0 1 2 3; do
    env $BASE $env N_NEW=$n $bin 256 4 $OUT/$pf > /dev/null 2> $OUT/last.err
    python3 - "$name" "$pf" "$run" "$OUT/last.err" >> $R <<'P'
import re,sys
name,pf,run,f=sys.argv[1:]; t=open(f).read()
pt=re.search(r'prompt tokens: (\d+)',t); tt=re.search(r'TTFT: ([0-9.]+)',t); m=re.search(r'Speed: [0-9.]+ tok/s \(([0-9.]+)s for (\d+)',t)
a=re.search(r'accepted \d+ \(([0-9.]+) %\)',t)
if not (tt and m): print(f"{name}\t{pf}\t{run}\tFEHLER"); sys.exit()
tot,n=float(m.group(1)),int(m.group(2)); tt=float(tt.group(1))
print(f"{name}\t{pf}\t{run}\t{pt.group(1) if pt else '?'}\t{tt:.2f}\t{(n-1)/(tot-tt):.2f}\t{a.group(1) if a else ''}")
P
  done
}
for pf in pshort.txt p6k.txt; do
  arm pr1338 $PR "" $pf 256
  arm int8 $NEW "" $pf 256
  arm int4 $NEW "$I4" $pf 256
  arm int4-mtp $NEW "$I4 $MTP" $pf 256
done
echo FERTIG
