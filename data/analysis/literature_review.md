# Literature Review — Novelty Assessment

## Core Question: Is our work novel?

### VERDICT: YES. Our core claims are genuinely novel.

---

## Papers that MUST be cited

| Paper | Year | Why cite |
|-------|------|---------|
| Zha, Gao & Ni | 2003 | Original NDBI formula + acknowledged limitations |
| Haas & Ahlstrom | 2021 | Documented detection failure at 30 refugee settlements in Uganda (F1=0.24). They show THAT it fails; we explain WHY |
| Pesaresi et al. | 2016 | GHS layer authors acknowledge "scattered huts in rural areas built with straw/clay not distinguishable from background" |
| Radoux et al. | 2016 | Formal sub-pixel detection limits for Sentinel-2 |
| Rasul et al. | 2018 | NDBI fails in arid zones (bare soil confusion) |
| Corbane et al. | 2021 | CNN for global settlements from S2; still fails on small/natural-material buildings |
| Kuffer, Pfeffer, Sliuzas | 2016 | "Slums from Space" review: limited generalizability across regions |
| Quinn et al. | 2018 | "geographic and environmental differences between camps" affect ML transfer |
| Sprohnle et al. | 2017 | Only direct Sentinel-2 + refugee camp paper (feasibility study) |
| Wernicke | 2023 | DLR thesis: DL + S2 for refugee camp extents |
| Wurm, Stark, Zhu | 2017 | GLCM texture + Sentinel-2 for slum mapping |
| Friedrich et al. | 2021 | Sentinel-2 products inadequately detect refugee settlements |
| Gella et al. | 2022 | Mask R-CNN cross-site transfer degradation for dwelling detection |

---

## Novelty Assessment: Claim by Claim

### 1. NDBI gap predicts detectability (r=0.91)
**FULLY NOVEL.** No paper uses NDBI contrast (camp minus background) as a predictor of detection performance. Literature documents NDBI limitations qualitatively but nobody has formalized the relationship.

### 2. Environmental dependence (arid works, humid fails)
**NOVEL quantification.** Qualitative awareness exists (Quinn 2018, Gella 2022) but nobody has formally shown detectability is a function of biome/climate with quantitative evidence and physical mechanism.

### 3. Texture signal inversion between biomes
**COMPLETELY NOVEL.** No paper documents that structural features (GLCM, gradient) invert direction between arid and humid for built-up detection. This is the most novel finding.

### 4. CNN failure at 10m cross-country (AUC=0.279)
**NOVEL at 10m.** VHR degradation documented (Gella 2022: MIoU 0.78→0.69) but not catastrophic. Our AUC=0.279 at 10m and framing as physics limitation is new.

### 5. Sentinel-2 for refugee camps (systematic study)
**Only 3 prior papers exist** (Sprohnle 2017, Wernicke 2023, Masoni 2021). None do cross-country analysis or explain failure mechanisms. We fill a significant gap.

---

## Methods Verification

| Method | Standard? | Concern | Mitigation |
|--------|-----------|---------|------------|
| LOCO evaluation | Yes (LOGO-CV) | n=7 small | L2CO (n=42) + per-camp (n=101) |
| NDBI formula | Yes (Zha 2003) | Bare soil confusion | Documented as core finding |
| Cohen's d | Valid, non-standard in RS | J-M distance more typical | Interpretable for broader audience |
| Bootstrap CI (n=7) | Valid in principle | Under-coverage at n<20 | Permutation test + L2CO compensate |
| Permutation test (n=7) | Yes, STRONGEST test | p-resolution limited to 1/5040 | p=0.005 is conclusive |
| class_weight='balanced' | Yes | None | Standard implementation |
| GLCM at 10m | Yes, published | Window size matters | 7x7 to 9x9 supported by lit |
| FFT for settlements | Valid, less common | Novel application | Physically motivated |
| Percentile norm (p2-p98) | Yes | None | Per-channel, train-only |
| Planetary Computer | Yes | Processing version issues | Fixed date range limits exposure |

### Key methodological recommendation:
- Emphasize permutation test (p=0.005) as primary statistical evidence
- Report bootstrap CI but note n=7 limitation
- Emphasize L2CO (n=42, r=0.872) as most robust validation

---

## What the literature says about our specific results

### South Sudan failure
- Pesaresi et al. (2016) explicitly mention "scattered huts built with traditional materials (straw/clay) not distinguishable from background"
- Haas & Ahlstrom (2021) documented F1=0.15-0.17 for S2-based products in Uganda
- But NOBODY has shown this is because NDBI gap is negative (-0.077)

### CNN adds nothing at 10m
- Corbane et al. (2021) showed even global-scale CNN fails on "small building structures made of natural materials"
- But nobody has directly compared CNN vs LogReg(NDBI) to prove signal is purely physical

### Environmental adaptivity
- Rasul et al. (2018) recommended different indices for different climates — but manually, not as formal framework
- No paper proposes switching detection pathway based on environment
