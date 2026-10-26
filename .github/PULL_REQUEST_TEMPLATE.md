## What changes, and why

<!-- One logical change. What it fixes or adds, and the evidence: a test, a record, a measurement. -->

## What you ran

<!-- The commands and their results: make check, make test-make-room-ci, a probe or an E2 point, as the change needs. -->

## Checklist

- [ ] Each commit is signed off (`git commit -s`, the Developer Certificate of Origin) and has an imperative subject.
- [ ] No file under `benchmarks/results` is edited; a new measurement has a dated directory of its own.
- [ ] A changed wire type or metric is regenerated (`make schema`, `make metrics`).
- [ ] A changed number in a document cites the record, test or script it comes from.
- [ ] A change to make-room's scheduler patches comes from #63350 or #63351, as `python/flatshare_vllm/README.md` describes.
