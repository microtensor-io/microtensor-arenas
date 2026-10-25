# Changelog

- Defined as a testnet draft. Envelope group mt-5 with max input 4,096 and max output 1,024 tokens; the edge bonus applies, so a system that also passes mt-3 (CPU) or mt-2 (Phone) earns up to 0.20. Bases Qwen2.5 Coder 7B and 1.5B, Qwen2.5 1.5B, Phi 4 mini and Llama 3.2 3B. Escalation through the relay (Claude Haiku 5.5, GLM 5.3 Flash) until frontier models are served on our network. Empty pools burn.
- Item pass line is 0.90: finance work needs near exact fields. Calibration against the reference model and a frontier model happens before the arena goes live.
- Training data under CC BY 4.0 only: VRDU Ad-buy Forms and Registration Forms, CORD v2 and FATURA. DocILE (research only) and RealKIE (non commercial) are excluded.
- Hidden items each epoch come from two sources that cannot be memorised: synthetic invoices, receipts and purchase orders rendered fresh from many layouts, currencies and languages with OCR noise, where the answer is known from how the document was built; and a smaller slice of newly filed public records (FCC ad invoices, FARA registrations) published after the commit window, labelled by hand.
- Still to build before it goes live: the `field_extraction` grader with the per type matching rules, the twin generator (same fields, other layout), the hidden item generator, and lower class tests that use the items whose input fits that class.
