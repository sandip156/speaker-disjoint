RESULTS — v1.0 (2026-10-14)
E2 — headline: cross-corpus speaker generalization
Model: ilprl-docse/NwachaMuna-NepConformer-Aug — their published SOTA (apache-2.0),their reported result: CER 17.59% / WER 52.61% (seen-speaker test set)
Our evaluation: same released model, zero training, on spk04 (127 utts, 0.10 h,speaker-disjoint, 15/18 domains)
CER 26.75% | WER 73.62%
Generalization gap: +9.16 CER absolute (+52% relative)
Decoding: greedy (model.transcribe), batch 16, T4 Colab
Normalization (identical for refs and hyps): NFC; strip danda । ? . , ! ; :and ZWJ/ZWNJ; collapse whitespace. Devanagari has no case.
E5 — per-domain CER (test domains with n>=10 only; bootstrap 95% CI, 1000 resamples)
domain	n	CER%	95% CI
tourist_places_culture	15	26.92	[22.9, 31.0]
hotels_accommodation	14	29.50	[23.9, 33.9]
food_restaurants	12	28.03	[23.4, 33.8]
guides_services	12	31.60	[25.9, 38.9]
animals_nature	12	15.34	[7.2, 23.6]
health_body	10	19.08	[14.1, 25.6]
Pattern: tourism/logistics domains (27–32%) worse than common-vocabulary domains(15–19%); consistent with entity-dense, agglutinated content. Descriptive only:domains and speaker are entangled (single test speaker).
Utterance-length effect: weak (shorter half 25.9% vs longer half 27.8%) — ruledout as the main driver of domain differences.
QC audit trail
All 431 utterances manually checked (audio vs transcript) by first author.
Post-hoc worst-10 audit: 8/10 references verified; ids 64, 345 = one-wordspeaker disfluencies (references retained as intended text; documented).
Voice verification: spk04 coherent (ids 74, 237); distinct from spk05/06 (SPLITS.md).
Corpus frozen: checksums_v1.0.txt (SHA-256, 437 files), MANIFEST INTACT.
Artifacts
e2_results_v1.tsv (per-utterance: id, speaker, domain, ref, hyp, cer)
e5_domain_results_v1.tsv (per-domain with CIs)
../sandiptourism/: master.tsv, speakers.tsv, split.tsv, SPLITS.md, README.md,teacher_check.tsv, audio_check.tsv, checksums_v1.0.txt
Historical: e2_results.tsv + e5_domain_results.tsv (v0.9-era, 197-utt test set:CER 27.33%) — superseded by the v1.0 revision, retained for provenance.
Notes for the paper
Historical result (27.33% on pre-revision 197-utt set) may be mentioned as aprior measurement; v1.0 numbers are primary.
Their model used zero tourism-domain training data → E5 contrasts reflect textproperties, not missing domain training (that question is E6, future work).