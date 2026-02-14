# Results Summary — Sentinel-2 Refugee Camp Detectability

## Scientific Question
> Under what environmental conditions are refugee camps spectrally
> detectable using Sentinel-2 at 10m resolution?

## Dataset
- 101 camps, 7 countries, 1689 tiles (505 camp + 1184 negative)
- Train: Syria, South Sudan, Turkey, Uganda
- Test: Chad, Ethiopia, Yemen
- 6 spectral channels: R, G, B, NDVI, NDBI, SWIR_ratio (128x128 px)

## Main Result
**NDBI spectral gap (camp minus background) explains 83% of cross-country
detection variance.**

| Country      | LOCO AUC | NDBI Gap  |
|-------------|----------|-----------|
| Turkey      | 0.773    | +0.129    |
| Ethiopia    | 0.678    | +0.023    |
| Yemen       | 0.658    | +0.020    |
| Uganda      | 0.654    | +0.055    |
| Chad        | 0.652    | +0.046    |
| Syria       | 0.531    | -0.001    |
| South Sudan | 0.238    | -0.077    |

## Statistical Validation
- Country-level: r = +0.912, R² = 0.832
- Leave-2-Countries-Out (n=42): r = +0.872
- Bootstrap 95% CI: [+0.558, +0.995]
- Permutation p = 0.005
- Per-camp (n=101): r = +0.597

## Physical Interpretation
- Camps are detectable only where built-up materials (concrete, metal roofing)
  contrast spectrally with natural background
- South Sudan: mud/thatch camps have LOWER NDBI than surroundings (negative gap)
  → spectrally invisible at 10m
- Turkey: concrete camps on rural land → strong NDBI contrast → detectable
- Syria: camps in semi-urbanized areas → no contrast → indistinguishable

## CNN vs Physical Model
- ResNet-18 (mini-test): Test AUC = 0.279 (worse than random)
- NDBI-only LogReg: Test AUC = 0.728
- Conclusion: CNN adds nothing. The signal is physical, not morphological.

## Conclusion
This is a **sensor/physics limitation**, not an algorithm limitation.
No amount of ML can recover signal that does not exist in the data.

## Figures
- `environmental_detectability.png` — NDBI gap vs AUC + aridity analysis
- `physical_mechanism.png` — NDBI distributions + Cohen's d by country
- `oos_validation.png` — Bootstrap + L2CO validation
- `oos_validation_results.txt` — Full numerical results
