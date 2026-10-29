@echo off
rem Every evaluation behind the README, one after the other (about 30 minutes on an RTX 2060).
cd /d %~dp0..
set C=courier\target\release\courier.exe
%C% eval latency --out results/latency.json > results\latency.log 2>&1
%C% eval latency --no-speculation --out results/latency-no-speculation.json > results\latency-no-speculation.log 2>&1
%C% eval latency --real-wake --out results/latency-real-wake.json > results\latency-real-wake.log 2>&1
%C% eval endpoint --out results/endpoint.json > results\endpoint.log 2>&1
%C% eval tools --out results/tools-3b.json > results\tools-3b.log 2>&1
%C% eval tools --set bench/tools-heldout.jsonl --out results/tools-heldout-3b.json > results\tools-heldout-3b.log 2>&1
%C% eval tools --no-grammar --out results/tools-3b-nogrammar.json > results\tools-3b-nogrammar.log 2>&1
%C% eval tools --no-grammar --set bench/tools-heldout.jsonl --out results/tools-heldout-3b-nogrammar.json > results\tools-heldout-3b-nogrammar.log 2>&1
%C% eval tools --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --out results/tools-7b.json > results\tools-7b.log 2>&1
%C% eval tools --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --set bench/tools-heldout.jsonl --out results/tools-heldout-7b.json > results\tools-heldout-7b.log 2>&1
echo finished > results\evals.done
