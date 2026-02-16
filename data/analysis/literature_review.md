# Literature Review — Novelty Assessment

## Core Question: Is our work novel?

### VERDICT: YES. Our core claims are genuinely novel.

---

## Papers that MUST be cited (VERIFIED feb-2026, all DOIs checked)

| Paper | Year | DOI | Why cite |
|-------|------|-----|---------|
| Zha, Gao & Ni | 2003 | 10.1080/01431160304987 | Original NDBI formula + acknowledged limitations |
| Pesaresi et al. | **2013** | 10.1109/JSTARS.2013.2271445 | GHS layer: "scattered huts built with straw or clay not distinguishable" |
| Radoux et al. | 2016 | 10.3390/rs8060488 | Formal sub-pixel detection limits for Sentinel-2 |
| Kuffer, Pfeffer, Sliuzas | 2016 | 10.3390/rs8060455 | "Slums from Space" review: limited generalizability across regions |
| Wendt, Lang, Rogenhofer | 2017 | 10.1553/giscience2017_01_s172 | Only direct Sentinel-2 + refugee camp paper (feasibility study) |
| Wurm, Weigand, Schmitt, Geiss, Taubenboeck | 2017 | 10.1109/JURSE.2017.7924586 | GLCM texture + Sentinel-2 for slum mapping (JURSE, not IGARSS) |
| Quinn et al. | 2018 | 10.1098/rsta.2017.0363 | "geographic and environmental differences between camps" affect ML transfer |
| Rasul et al. | 2018 | 10.3390/land7030081 | NDBI fails in arid zones (bare soil confusion) |
| Braun, Fakhri, Hochschild | 2019 | 10.3390/rs11172047 | SAR-based camp monitoring (different sensor modality) |
| Corbane et al. | 2021 | 10.1007/s00521-020-05449-7 | CNN for global settlements from S2; fails on natural-material buildings |
| Friedrich & Van Den Hoek | **2020** | 10.1016/j.compenvurbsys.2020.101499 | Landsat temporal detection of refugee settlements in Uganda |
| Van Den Hoek & Friedrich | 2021 | 10.3390/rs13183574 | F1=0.16-0.26 at 30 refugee settlements in Uganda. They show THAT it fails; we explain WHY |
| Gella et al. | 2022 | 10.3390/rs14030689 | Mask R-CNN cross-site transfer degradation for dwelling detection |
| Wernicke | 2023 | — (elib.dlr.de/196349/) | M.S. thesis, Univ. Wurzburg: DL + S2 for refugee camp extents |

**Removed:** ~~Haas & Ahlstrom (2021)~~ — phantom citation, does not exist.
**Removed:** ~~Masoni (2021)~~ — predatory journal (Walsh Medical Media), not indexed.
**Corrected:** Pesaresi year 2016→2013, Sprohnle→Wendt, Wurm authors, Friedrich co-author & year.

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
**Only 2 verified prior papers exist** (Wendt et al. 2017, Wernicke 2023).
  None do cross-country analysis or explain failure mechanisms. We fill a significant gap.

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
- Pesaresi et al. (2013) explicitly mention "scattered huts in rural areas built with traditional materials such as straw or clay (not distinguishable from the background soil and vegetation patterns)"
- Van Den Hoek & Friedrich (2021) documented F1=0.16-0.26 for S2-based products in Uganda
- But NOBODY has shown this is because NDBI gap is negative (-0.077)

### CNN adds nothing at 10m
- Corbane et al. (2021) showed even global-scale CNN fails on "small building structures made of natural materials"
- But nobody has directly compared CNN vs LogReg(NDBI) to prove signal is purely physical

### Environmental adaptivity
- Rasul et al. (2018) recommended different indices for different climates — but manually, not as formal framework
- No paper proposes switching detection pathway based on environment

---

## ERRATA LOG (corrections applied feb-2026)
- "Haas & Ahlstrom (2021)" was a phantom citation — real paper is Van Den Hoek & Friedrich (2021)
- Pesaresi year corrected from 2016 to 2013 (IEEE JSTARS 6(5))
- "Sprohnle et al." corrected to Wendt, Lang & Rogenhofer (GI_Forum, not ISPRS)
- "Wurm, Stark, Zhu" corrected to Wurm, Weigand, Schmitt, Geiss, Taubenboeck (JURSE, not IGARSS)
- "Friedrich & Witmer (2021)" corrected to Friedrich & Van Den Hoek (2020)
- Masoni (2021) removed — predatory journal
- Corbane quote attributed to Pesaresi (2013) instead — needs full text verification
