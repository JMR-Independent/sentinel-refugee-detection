# Proyecto Completo: Detectabilidad Ambiental de Campos de Refugiados con Sentinel-2

> Documento unificado de referencia. Contiene: qué hicimos, cómo, por qué,
> todos los resultados, verificaciones metodológicas, revisión de literatura,
> y fuentes. Última actualización: 2026-02-14.

---

## 1. Pregunta Científica

> ¿Bajo qué condiciones ambientales son espectralmente detectables los campos
> de refugiados usando Sentinel-2 a 10 m de resolución?

**Lo que NO es este proyecto:**
- No es un detector de campos de refugiados
- No es un paper de ML/deep learning
- No propone una arquitectura nueva

**Lo que SÍ es:**
- Una caracterización de los **límites físicos de observabilidad** del sensor
- Una explicación de **por qué fallan** todos los intentos previos de detección
- Una demostración de que la detectabilidad es una propiedad del **entorno**, no del algoritmo

---

## 2. Dataset

### 2.1 Campos de refugiados
- **101 campos** en 7 países
- Fuentes: UNHCR/HDX (datos oficiales) + OpenStreetMap Overpass API + listas conocidas
- Archivo: `data/labels/all_camps_merged.csv` (102 líneas incl. header)

| País         | N campos | Fuente principal |
|-------------|----------|-----------------|
| Syria       | 28       | UNHCR/HDX + OSM |
| Ethiopia    | 26       | UNHCR/HDX       |
| Chad        | 19       | UNHCR/HDX       |
| South Sudan | 10       | UNHCR/HDX + OSM |
| Turkey      | 6        | UNHCR/HDX       |
| Uganda      | 6        | UNHCR/HDX       |
| Yemen       | 6        | UNHCR/HDX       |

### 2.2 Tiles Sentinel-2
- **1708 tiles** planificados, **1689 descargados** (98.9%)
- 505 tiles de campo + 1184 tiles negativos (sin campo)
- 24 tiles fallaron permanentemente (corruptos en Planetary Computer)
- Archivo: `data/labels/full_dataset_all_locations.csv` (1709 líneas incl. header)

### 2.3 Grilla por campo
- 5 tiles por campo: centro + 4 adyacentes (esquinas excluidas)
- Tamaño de tile: 128 x 128 píxeles a 10 m = 1.28 km x 1.28 km
- Estrategia de etiquetado: `point_buffer` con radio de 500 m

### 2.4 Muestras negativas
- 4 categorías: rural, urbano, árido/desierto, informal (hard negatives)
- 3 negativos por campo por categoría
- Distancia mínima: 10 km de cualquier campo conocido

### 2.5 Split geográfico
- **Train:** Syria, South Sudan, Turkey, Uganda
- **Test:** Chad, Ethiopia, Yemen
- Nunca se mezclan tiles del mismo país entre train y test

### 2.6 Bandas espectrales
Bandas descargadas (raw):
- B02 (Blue, 10m), B03 (Green, 10m), B04 (Red, 10m)
- B08 (NIR, 10m), B11 (SWIR1, 20m nativo), B12 (SWIR2, 20m nativo)

Canales del modelo (derivados):
- R (B04), G (B03), B (B02)
- NDVI = (B08 - B04) / (B08 + B04)
- NDBI = (B11 - B08) / (B11 + B08)
- SWIR_ratio = B12 / B11

B11 y B12 se resamplean de 20m a 10m con interpolación bilineal.

### 2.7 Parámetros de descarga
- Fuente: Microsoft Planetary Computer (STAC API)
- Nivel: Sentinel-2 L2A (corrección atmosférica)
- Rango de fechas: 2022-01-01 a 2023-12-31
- Cloud cover máximo: 20%
- Rate limiting: 0.5s entre tiles, 3s cada 20 tiles
- Normalización: percentil 2-98 por canal, solo desde train split

---

## 3. Métodos

### 3.1 Evaluación: Leave-One-Country-Out (LOCO)
Para cada país C de los 7:
1. Excluir C del dataset
2. Entrenar modelo con los 6 países restantes
3. Testear en C
4. Reportar AUC en C

Esto garantiza que **nunca hay filtración geográfica** entre train y test.

### 3.2 Modelo espectral
- **Algoritmo:** LogisticRegression (scikit-learn)
- **Features:** NDBI mean, NDBI std, fracción de píxeles "built-up" (NDBI > 0)
- **class_weight:** 'balanced' (ajuste automático por frecuencia de clase)
- **Regularización:** C = 0.1

### 3.3 Modelo textural
- **Algoritmo:** LogisticRegression (scikit-learn)
- **Features:** 17 features estructurales extraídas de la banda Red (B04, 10m nativo)
- **class_weight:** 'balanced', C = 0.1
- **Normalización:** StandardScaler ajustado solo en train

**5 canales texturales:**

| Canal | Cálculo | Features extraídas |
|-------|---------|-------------------|
| Gradient magnitude | Sobel (scipy.ndimage) | mean, std, p90 |
| Edge density | Canny threshold=0.1 + convolución 5x5 | edge_frac, edge_density_std |
| Directional entropy | Bloques 8x8, histograma orientación 8 bins | mean, std, min |
| GLCM | 16 niveles, distancia=2, 4 ángulos | homogeneity, contrast, correlation, dissimilarity, energy |
| FFT radial power | FFT 2D, promediado radial | low, mid, high, mid_ratio, peak |

### 3.4 NDBI spectral gap (la métrica clave)
```
ΔNDBI = mean(NDBI_camp_tiles) - mean(NDBI_negative_tiles)
```
Calculado **por país**. Mide el contraste espectral entre los materiales de
construcción del campo y el suelo/vegetación natural circundante.

- ΔNDBI > 0: campo más "built-up" que alrededores (hormigón, metal)
- ΔNDBI ≈ 0: campo espectralmente indistinguible
- ΔNDBI < 0: campo MENOS "built-up" que alrededores (barro sobre barro)

### 3.5 CNN (ResNet-18) — experimento de contraste
- ResNet-18 pretrained ImageNet, primera conv modificada: 3→6 canales
- FC: 512 → 1 (clasificación binaria, sigmoid)
- Entrenado en mini-test (n=166 tiles)
- Dos variantes: frozen FC y unfrozen layer4+FC

### 3.6 Adaptive dual-pathway (experimento final)
Protocolo para cada país test C:
1. Excluir C
2. Correr LOCO interno con modelo espectral y textural en los 6 restantes
3. Aprender τ óptimo (threshold del NDBI gap) maximizando mean AUC
4. Calcular NDBI gap de C
5. Si gap > τ → vía espectral; si gap ≤ τ → vía textural
6. Reportar AUC en C

### 3.7 Validación estadística
- **Bootstrap CI:** 10,000 remuestras de los n=7 pares (NDBI gap, AUC)
- **Permutation test:** 10,000 permutaciones (7! = 5040 posibles)
- **Leave-2-Countries-Out (L2CO):** 42 evaluaciones (C(7,2) = 21 pares × 2)
- **Per-camp correlation:** n=101 campos individuales
- **Cohen's d:** Tamaño de efecto para discriminación textural

---

## 4. Resultados

### 4.1 Resultado principal: NDBI gap predice detectabilidad

| País         | LOCO AUC | NDBI Gap  | Clima      |
|-------------|----------|-----------|------------|
| Turkey      | 0.773    | +0.129    | Semiárido  |
| Ethiopia    | 0.678    | +0.023    | Semiárido  |
| Yemen       | 0.658    | +0.020    | Árido      |
| Uganda      | 0.654    | +0.055    | Húmedo     |
| Chad        | 0.652    | +0.046    | Árido      |
| Syria       | 0.531    | -0.001    | Semiárido  |
| South Sudan | 0.238    | -0.077    | Húmedo     |

**Mean LOCO AUC:** 0.612 ± 0.14

**Correlación NDBI gap vs AUC:**
- Country-level (n=7): **r = +0.912, R² = 0.832**
- Leave-2-Countries-Out (n=42): **r = +0.872, R² = 0.760**
- Bootstrap 95% CI: [+0.558, +0.995]
- Permutation p-value: **0.005** (solo 25/5040 permutaciones producen r ≥ 0.912)
- Per-camp (n=101): r = +0.597, R² = 0.356
- P(r < 0) en bootstrap: 0.0204

**Estabilidad L2CO por país:**
```
chad             mean=0.642 ± 0.058  (n=6 evaluaciones)
ethiopia         mean=0.679 ± 0.037
south_sudan      mean=0.235 ± 0.005  ← consistentemente indetectable
syria            mean=0.527 ± 0.023
turkey           mean=0.750 ± 0.055  ← consistentemente detectable
uganda           mean=0.654 ± 0.012
yemen            mean=0.660 ± 0.019
```

### 4.2 Mecanismo físico

**Turquía (gap = +0.129):** Campos de hormigón/contenedores sobre suelo rural.
Fuerte contraste espectral. AUC = 0.773.

**South Sudan (gap = -0.077):** Campos de barro/paja sobre suelo de barro.
"Barro sobre barro" — el campo tiene MENOR NDBI que los alrededores (gap negativo).
El campo es espectralmente invisible. AUC = 0.238.

**Syria (gap ≈ 0):** Campos en zonas semi-urbanizadas. El NDBI del campo es
idéntico al del entorno urbano circundante. Sin contraste. AUC = 0.531.

### 4.3 CNN no aporta nada a 10 m

| Modelo | Val AUC | Test AUC |
|--------|---------|----------|
| ResNet-18 frozen FC | 0.964 | 0.294 |
| ResNet-18 unfrozen layer4+FC | — | 0.279 |
| NDBI-only LogReg | — | **0.728** |
| Random Forest (band stats) | — | 0.535 |

**Conclusión:** CNN memoriza el entrenamiento (Val AUC 0.964) pero no
generaliza (Test AUC 0.279, peor que azar). LogReg con solo NDBI supera
a CNN. La señal es **física (contraste espectral)**, no morfológica.
A 10 m un campo es ~12×12 píxeles — no hay estructura resoluble.

### 4.4 Análisis textural

**South Sudan — discriminación por features estructurales:**

| Feature | Cohen's d | p-value | Dirección |
|---------|-----------|---------|-----------|
| GLCM dissimilarity | -0.70 | 0.0002 | camp < neg |
| GLCM homogeneity | +0.68 | 0.0003 | camp > neg |
| Gradient magnitude | -0.66 | 0.0003 | camp < neg |
| Edge fraction | -0.59 | 0.0017 | camp < neg |
| GLCM contrast | -0.59 | 0.0002 | camp < neg |
| GLCM energy | +0.50 | 0.0051 | camp > neg |
| GLCM correlation | +0.48 | 0.0016 | camp > neg |

**Interpretación:** Campos en South Sudan son MÁS LISOS y HOMOGÉNEOS que
la vegetación circundante. Efecto de "suelo despejado" — detectable aunque
el material (barro) sea espectralmente idéntico.

**LOCO espectral vs textural por país:**

| País | Espectral | Textural | Diferencia | Ganador |
|------|-----------|----------|------------|---------|
| Chad | 0.568 | 0.536 | -0.032 | espectral |
| Ethiopia | 0.732 | 0.611 | -0.121 | espectral |
| **S. Sudan** | **0.356** | **0.542** | **+0.186** | **textural** |
| Syria | 0.663 | 0.426 | -0.237 | espectral |
| Turkey | 0.566 | 0.629 | +0.063 | textural |
| Uganda | 0.694 | 0.634 | -0.060 | espectral |
| Yemen | 0.706 | 0.584 | -0.122 | espectral |

**Medias:** Espectral = 0.612, Textural = 0.566, Oracle = 0.648

### 4.5 Inversión de señal textural entre biomas

Descubrimiento clave: la **dirección** de la señal textural se invierte:

- **Árido:** campos son MÁS texturados que alrededores (estructuras sobre suelo desnudo)
- **Húmedo:** campos son MÁS LISOS que alrededores (suelo despejado en vegetación)

Esto significa que **no existe una firma estructural universal de campo a 10 m**.
Un clasificador textural global funciona PEOR que uno espectral (0.566 vs 0.612)
porque los features apuntan en direcciones opuestas según el bioma.

### 4.6 Experimento final: gating adaptativo con τ aprendido

| País | τ aprendido | NDBI gap | Vía elegida | AUC |
|------|------------|----------|-------------|-----|
| Chad | -0.075 | +0.0455 | espectral | 0.568 |
| Ethiopia | -0.075 | +0.0233 | espectral | 0.732 |
| S. Sudan | -0.100 | -0.0770 | espectral | 0.356 |
| Syria | +0.130 | -0.0009 | textural | 0.426 |
| Turkey | -0.075 | +0.1288 | espectral | 0.566 |
| Uganda | -0.075 | +0.0549 | espectral | 0.694 |
| Yemen | -0.075 | +0.0201 | espectral | 0.706 |

**Resumen:**
```
Adaptativo (τ desde training):  0.578
Solo espectral:                  0.612
Solo textural:                   0.566
Oracle (mejor por país):         0.648
```

**Por qué falla el gating automático:** South Sudan es el ÚNICO país con
gap negativo. Cuando se excluye del entrenamiento, ningún otro país provee
señal de que "gap negativo → usar textural". Con n=7 países, es imposible
aprender τ. El entorno DEBE caracterizarse a priori.

### 4.7 Resumen de modelos
```
Espectral LOCO mean AUC:    0.612
Textural LOCO mean AUC:     0.566
Oracle (best per country):  0.648
Ensemble (sigmoid gap):     0.619
Adaptive (τ OOS):           0.578
CNN cross-country:          0.279
```

---

## 5. Conclusiones Científicas

1. **NDBI spectral gap explica 83% de la varianza** de detección cross-country (r=0.91).
   Es una limitación del SENSOR/FÍSICA, no algorítmica.

2. **Features estructurales (GLCM, gradiente, FFT) recuperan señal parcial**
   en South Sudan: AUC 0.356 → 0.542. Pero no son universales.

3. **La dirección de la señal estructural se INVIERTE entre biomas:**
   árido = campos más texturados; húmedo = campos más lisos.
   No existe firma estructural universal a 10m.

4. **El gating automático falla** con n=7 porque South Sudan es único.
   La caracterización ambiental debe hacerse a priori, no inferirse.

5. **CNN no aporta nada a 10 m.** La señal es física (material de construcción
   vs suelo), no morfológica. A 10 m/px no hay geometría resoluble.

6. **Oracle muestra potencial** (AUC = 0.648) si se sabe qué vía usar.
   Pero requiere conocimiento ambiental, no más ML.

**En una frase:** No descubrimos cómo detectar mejor campamentos.
Descubrimos cuándo **no existe nada que detectar** con Sentinel-2.

---

## 6. Verificación de Métodos

### 6.1 LOCO — ¿Es estándar?
**SÍ.** Leave-One-Group-Out CV es gold standard para generalización geográfica.
Implementado como `LeaveOneGroupOut` en scikit-learn. Usado en poverty mapping
(Jean et al., Nature Communications 2020; PNAS 2025), marine RS
(iSLOOCV, Taylor & Francis 2022). Nuestro L2CO (n=42) agrega robustez.

### 6.2 NDBI fórmula — ¿Es correcta?
**SÍ.** NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) es la fórmula original de
Zha, Gao & Ni (2003), IJRS 24(3), 583-594. Nuestro uso de B11/B08 de
Sentinel-2 es el análogo espectral correcto de TM5/TM4 de Landsat.

### 6.3 Cohen's d — ¿Es válido en RS?
**VÁLIDO pero no estándar.** En RS se usa más J-M distance o Transformed
Divergence. Cohen's d es más interpretable para audiencia amplia.
Valores d = 0.5-0.7 son "efecto mediano a grande" por convención de Cohen.
Defendible para paper corto.

### 6.4 Bootstrap CI con n=7 — ¿Es fiable?
**LIMITADO.** Efron & Tibshirani recomiendan n ≥ 30. A n=7, coverage real
del CI 95% es ~81-83% (R-bloggers analysis; Univ. Minnesota notes).
BCa intervals también sub-cubren a n pequeño.

**Mitigación:** El permutation test (p=0.005) es más fuerte. L2CO (n=42,
r=0.872) provee validación independiente. Recomendación: reportar bootstrap
CI pero enfatizar permutation test.

### 6.5 Permutation test con n=7 — ¿Es válido?
**SÍ — el test MÁS fuerte para n pequeño.** Solo requiere exchangeability
bajo H0, no asume distribución. Con n=7 hay 5040 permutaciones posibles.
p=0.005 significa que solo 25/5040 permutaciones producen r ≥ 0.912.
Resolución de p limitada a 1/5040 = 0.0002. Para nuestros propósitos,
p=0.005 es concluyente. Validado por Yu (2020, arXiv), UVA Library guide.

### 6.6 class_weight='balanced' — ¿Es correcto?
**SÍ.** Estándar para clasificación desbalanceada. Ajusta pesos inversamente
proporcional a frecuencia de clase. Nuestro ratio 505:1184 (~1:2.3) es
moderadamente desbalanceado, apropiado para este approach.

### 6.7 GLCM a 10 m — ¿Es significativo?
**SÍ.** Publicado en Nature Scientific Reports 2024 (detección de fuego),
MDPI RS 2022 (vegetación), Wurm et al. 2017 (slums). Ventanas 7×7 a 9×9
son apropiadas a 10 m (70-90 m en terreno). Nuestros parámetros: 16 niveles,
distancia=2, 4 ángulos.

### 6.8 FFT para settlements — ¿Tiene precedente?
**VÁLIDO, menos común.** Publicado para detección de features antropogénicas
en DTMs (MDPI RS 2025), change detection con decoders FFT (Taylor & Francis
2025). Nuestro uso a 128×128 tiles (tamaño estándar FFT) es físicamente
motivado: grillas de refugios producen picos en frecuencia.

### 6.9 Percentile normalization (p2-p98) — ¿Es estándar?
**SÍ.** Recomendado por Sentinel Hub para deep learning con imágenes
satelitales. Per-channel, stats calculados solo desde train split,
con safety checks para rangos cero.

### 6.10 Planetary Computer — ¿Es fuente válida?
**SÍ.** Fuente estándar de investigación. Provee Sentinel-2 L2A completo.
Issue conocido: inconsistencia en versión de procesamiento Sen2Cor entre
regiones. Mitigado por nuestro rango fijo de fechas.

---

## 7. Revisión de Literatura y Evaluación de Novedad

### 7.1 Papers que DEBEN citarse (VERIFICADOS feb-2026)

| # | Citación correcta | Journal / Venue | DOI | Relevancia |
|---|-------------------|----------------|-----|------------|
| 1 | Y. Zha, J. Gao, S. Ni (2003) | Int. J. Remote Sens., 24(3), 583-594 | 10.1080/01431160304987 | Fórmula original NDBI + limitación bare soil reconocida |
| 2 | M. Pesaresi et al. **(2013)** | IEEE J-STARS, 6(5), 2102-2131 | 10.1109/JSTARS.2013.2271445 | GHS layer: "scattered huts built with straw or clay not distinguishable" |
| 3 | J. Radoux et al. (2016) | Remote Sens., 8(6), 488 | 10.3390/rs8060488 | Límites formales de detección sub-pixel para Sentinel-2 |
| 4 | M. Kuffer, K. Pfeffer, R. Sliuzas (2016) | Remote Sens., 8(6), 455 | 10.3390/rs8060455 | "Slums from Space" review: generalización limitada |
| 5 | L. Wendt, S. Lang, E. Rogenhofer **(2017)** | GI_Forum, 1, 172-182 | 10.1553/giscience2017_01_s172 | Único paper directo S2 + campo de refugiados (feasibility) |
| 6 | M. Wurm, M. Weigand, A. Schmitt, C. Geiss, H. Taubenboeck **(2017)** | **JURSE 2017**, 1-4 | 10.1109/JURSE.2017.7924586 | GLCM + S2 para slums (el más cercano a nuestra textura) |
| 7 | J. Quinn et al. (2018) | Phil. Trans. R. Soc. A, 376(2128), 20170363 | 10.1098/rsta.2017.0363 | "geographic and environmental differences between camps" |
| 8 | A. Rasul et al. (2018) | Land, 7(3), 81 | 10.3390/land7030081 | NDBI falla en zonas áridas (confusión con suelo desnudo) |
| 9 | A. Braun, F. Fakhri, V. Hochschild (2019) | Remote Sens., 11(17), 2047 | 10.3390/rs11172047 | SAR para monitoreo de campos (Sentinel-1, modalidad diferente) |
| 10 | C. Corbane et al. (2021) | Neural Comput. & Appl., 33, 6697-6720 | 10.1007/s00521-020-05449-7 | CNN global settlements S2; falla con edificaciones de material natural |
| 11 | J. Van Den Hoek, H.K. Friedrich **(2021)** | Remote Sens., 13(18), 3574 | 10.3390/rs13183574 | F1=0.16-0.26 en 30 campos Uganda. Documentan QUE falla; nosotros POR QUÉ |
| 12 | H.K. Friedrich, J. Van Den Hoek **(2020)** | Comput. Environ. Urban Syst., 82, 101499 | 10.1016/j.compenvurbsys.2020.101499 | Detección temporal Landsat en Uganda (BFAST) |
| 13 | G.W. Gella et al. (2022) | Remote Sens., 14(3), 689 | 10.3390/rs14030689 | Mask R-CNN cross-site: degradación MIoU en transferencia |
| 14 | K. Wernicke (2023) | M.S. thesis, Univ. Wurzburg | — (elib.dlr.de/196349/) | DL + S2 para extensiones de campos |

**NOTA:** Masoni (2021) eliminado — publicado en journal predatorio (Walsh Medical Media),
no indexado en WoS ni Scopus. Solo quedan 2 papers S2+campos: Wendt 2017 y Wernicke 2023.

### 7.1.1 Errores corregidos en esta verificación

| Error | Antes (INCORRECTO) | Ahora (CORRECTO) |
|-------|-------------------|-------------------|
| Cita fantasma | "Haas & Ahlstrom (2021)" | No existe. Paper real: Van Den Hoek & Friedrich (2021) |
| Pesaresi año | 2016 | **2013** (IEEE JSTARS 6(5)) |
| Pesaresi URL | ieee.org/document/7306961 (paper ajeno) | ieee.org/document/6578177 |
| Sprohnle autores | "Sprohnle et al." | **Wendt, Lang & Rogenhofer** |
| Sprohnle venue | ISPRS | **GI_Forum** |
| Wurm autores | "Wurm, Stark, Zhu" | **Wurm, Weigand, Schmitt, Geiss, Taubenboeck** |
| Wurm venue | IGARSS | **JURSE 2017** |
| Friedrich co-autor | "Witmer" | **Van Den Hoek** |
| Friedrich año | 2021 | **2020** |
| Corbane quote | Comillas en "small building structures..." | Parafrasear sin comillas (quote es de Pesaresi 2013) |
| Masoni journal | Citado como fuente | **Eliminado** (journal predatorio) |

### 7.2 Evaluación de novedad: claim por claim

**Claim 1: NDBI gap predice detectabilidad (r = 0.91)**
- **TOTALMENTE NOVEL.** Nadie usó el contraste NDBI camp-vs-background como
  predictor de performance. La literatura documenta limitaciones del NDBI
  cualitativamente pero nadie formalizó la relación.

**Claim 2: Dependencia ambiental cuantificada (árido funciona, húmedo falla)**
- **NOVEL en cuantificación.** Quinn (2018) y Gella (2022) mencionan
  cualitativamente que diferencias geográficas afectan transferencia.
  Nadie mostró formalmente que detectabilidad = f(bioma) con evidencia
  cuantitativa y mecanismo físico.

**Claim 3: Inversión de señal textural entre biomas**
- **COMPLETAMENTE NOVEL.** No existe NINGÚN paper que documente que features
  estructurales (GLCM, gradiente) invierten dirección entre árido y húmedo
  para detección de built-up. Es el hallazgo más novedoso.

**Claim 4: CNN falla a 10 m cross-country (AUC = 0.279)**
- **NOVEL a 10 m.** A VHR hay degradación documentada (Gella 2022: MIoU
  0.78→0.69) pero no catastrófica. AUC=0.279 a 10 m y framing como
  limitación física es nuevo.

**Claim 5: Sentinel-2 para campos de refugiados (estudio sistemático)**
- **Solo 2 papers previos verificados:** Wendt et al. 2017, Wernicke 2023.
  (Masoni 2021 eliminado — journal predatorio.) Ninguno hace análisis
  cross-country ni explica mecanismos de fallo. Llenamos un vacío significativo.

### 7.3 Cómo se conecta con la literatura

**Zha et al. (2003)** dijeron que NDBI no separa built-up de bare soil.
En 2003 eso era una limitación. En nuestro paper eso pasa a ser la
**variable causal que predice detectabilidad**: ΔNDBI ≈ separabilidad ≈ AUC.

**Pesaresi et al. (2013)** describieron literalmente South Sudan:
"scattered huts in rural areas built with traditional materials such as
straw or clay (not distinguishable from the background soil and vegetation
patterns)." Pero lo mencionan como problema sin cuantificar. Nosotros
mostramos: ΔNDBI < 0 → AUC ≈ 0.24. Convertimos una observación anecdótica
en una ley empírica.

**Van Den Hoek & Friedrich (2021)** documentaron que GHS-BUILT-S2 tiene
F1=0.16 (post-2016) a F1=0.26 (pre-2016) en 30 campos de Uganda, pero no
explican el mecanismo. Nosotros mostramos que falla cuando ΔNDBI ≈ 0.
Su paper se transforma en evidencia externa de nuestro mecanismo.

**Corbane et al. (2021)** entrenaron CNN globalmente en 277 AOIs para
settlements. Su producto (GHS-BUILT-S2) es exactamente el que Van Den Hoek
& Friedrich muestran que falla en campos. CNN a escala global sigue
fallando con edificaciones de material natural en entornos rurales.

**La literatura muestra que CNN falla, NDBI falla, built-up layers fallan,
slum detectors no generalizan. Pero nadie preguntó: ¿existe siquiera una
señal detectable a 10 m? Nosotros sí.**

---

## 8. Figuras Generadas

| Archivo | Contenido |
|---------|-----------|
| `figure_main.png` / `.pdf` | **Figura principal (3 paneles, 300 DPI):** (a) NDBI gap vs AUC scatter r=0.91, (b) ROC South Sudan espectral vs textural, (c) Cohen's d inversión por país |
| `environmental_detectability.png` | 3 paneles: (a) NDBI gap vs AUC, (b) Aridity vs AUC, (c) Barras por país |
| `physical_mechanism.png` | 2 paneles: (a) NDBI gap por país, (b) Cohen's d effect size |
| `oos_validation.png` | 3 paneles: (a) Country-level scatter, (b) Bootstrap distribution, (c) L2CO scatter |
| `texture_discrimination.png` | 6 paneles: features texturales camp vs no-camp por país |
| `texture_south_sudan.png` | 6 paneles: distribuciones de South Sudan |
| `dual_pathway_comparison.png` | Barras AUC espectral vs textural por país |
| `spectral_separation.png` | Separación espectral entre clases |
| `visual_validation_rgb.png` | Composites RGB de tiles de ejemplo |
| `channel_comparison.png` | Comparación de canales |
| `seasonal_comparison.png` | Comparación estacional |

---

## 9. Estructura del Repositorio

```
sentinel-refugee-detection/
├── configs/default.yaml              # Todos los hiperparámetros
├── src/
│   ├── __init__.py
│   ├── data.py                       # Dataset, normalización, manifests
│   ├── model.py                      # ResNet-18 (6 canales input)
│   ├── train.py                      # Training loop + evaluación
│   └── utils.py                      # Download, índices, geo utilities
├── notebooks/
│   ├── 01_download_labels.ipynb      # Labels: UNHCR + OSM
│   ├── 02_download_sentinel2.ipynb   # Descarga S2 (Colab)
│   ├── 03_prepare_tiles.ipynb        # Tile + normalizar + split
│   ├── 04_train_model.ipynb          # CNN training (Colab)
│   ├── 05_generalization.ipynb       # Test cross-country
│   └── 06_validation.ipynb           # Métricas + figuras
├── data/
│   ├── labels/
│   │   ├── all_camps_merged.csv      # 101 campos
│   │   └── full_dataset_all_locations.csv  # 1708 tiles
│   ├── sentinel2/                    # 1689 .npy tiles (6ch, 128x128)
│   └── analysis/                     # Figuras + resultados + este documento
├── download_full_dataset.py          # Batch downloader v1
├── download_full_v2.py               # v2 con resume + flushed output
└── download_retry.py                 # Retry fallidos con rate limiting
```

---

## 10. Historial de Git

```
faa0223 Add literature review: novelty assessment and methods verification
29bdd84 Final results: adaptive gating + main figure for paper
e111c37 Texture analysis: structural features recover signal in humid biomes
2bf31c6 Add comprehensive results summary for paper writing
eb291a8 OOS validation: NDBI gap explains 83% of cross-country detection variance
a4c9890 Full dataset analysis: environmental detectability of refugee camps
60b9989 Mini-test complete: 166/170 tiles downloaded, quantitative analysis done
0dbb312 Pre-download checklist: geographic validation and strategic camp selection
1841afb Add multi-source camp labels: UNHCR/HDX + OSM + known camps
ee12f2b Exclude corner tiles by default, add mini-test mode and dataset summary
1a075f4 Address 8 reviewer points: labels, validation gate, hard negatives
2c16c75 Full pipeline: src modules + 6 notebooks with expert improvements
b578bbb Initial project structure: configs, requirements, README
```

---

## 11. Fuentes y Referencias Verificadas

### Referencias en formato IEEE (VERIFICADAS feb-2026)

```
[1]  Y. Zha, J. Gao, and S. Ni, "Use of normalized difference built-up
     index in automatically mapping urban areas from TM imagery," Int. J.
     Remote Sens., vol. 24, no. 3, pp. 583-594, 2003,
     doi: 10.1080/01431160304987.

[2]  M. Pesaresi et al., "A global human settlement layer from optical
     HR/VHR RS data: Concept and first results," IEEE J. Sel. Topics Appl.
     Earth Observ. Remote Sens., vol. 6, no. 5, pp. 2102-2131, Oct. 2013,
     doi: 10.1109/JSTARS.2013.2271445.

[3]  J. Radoux et al., "Sentinel-2's potential for sub-pixel landscape
     feature detection," Remote Sens., vol. 8, no. 6, Art. no. 488, 2016,
     doi: 10.3390/rs8060488.

[4]  M. Kuffer, K. Pfeffer, and R. Sliuzas, "Slums from space--15 years
     of slum mapping using remote sensing," Remote Sens., vol. 8, no. 6,
     Art. no. 455, 2016, doi: 10.3390/rs8060455.

[5]  L. Wendt, S. Lang, and E. Rogenhofer, "Monitoring of refugee and
     camps for internally displaced persons using Sentinel-2 imagery--A
     feasibility study," GI_Forum, vol. 1, pp. 172-182, 2017,
     doi: 10.1553/giscience2017_01_s172.

[6]  M. Wurm, M. Weigand, A. Schmitt, C. Geiss, and H. Taubenboeck,
     "Exploitation of textural and morphological image features in
     Sentinel-2A data for slum mapping," in Proc. Joint Urban Remote
     Sens. Event (JURSE), Dubai, UAE, 2017, pp. 1-4,
     doi: 10.1109/JURSE.2017.7924586.

[7]  J. A. Quinn, M. M. Nyhan, C. Navarro, D. Coluccia, L. Bromley, and
     M. Luengo-Oroz, "Humanitarian applications of machine learning with
     remote-sensing data: Review and case study in refugee settlement
     mapping," Phil. Trans. R. Soc. A, vol. 376, no. 2128, Art. no.
     20170363, 2018, doi: 10.1098/rsta.2017.0363.

[8]  A. Rasul et al., "Applying built-up and bare-soil indices from Landsat
     8 to cities in dry climates," Land, vol. 7, no. 3, Art. no. 81, 2018,
     doi: 10.3390/land7030081.

[9]  A. Braun, F. Fakhri, and V. Hochschild, "Refugee camp monitoring and
     environmental change assessment of Kutupalong, Bangladesh, based on
     radar imagery of Sentinel-1 and ALOS-2," Remote Sens., vol. 11,
     no. 17, Art. no. 2047, 2019, doi: 10.3390/rs11172047.

[10] C. Corbane et al., "Convolutional neural networks for global human
     settlements mapping from Sentinel-2 satellite imagery," Neural Comput.
     & Appl., vol. 33, pp. 6697-6720, 2021,
     doi: 10.1007/s00521-020-05449-7.

[11] H. K. Friedrich and J. Van Den Hoek, "Breaking ground: Automated
     disturbance detection with Landsat time series captures rapid refugee
     settlement establishment and growth," Comput. Environ. Urban Syst.,
     vol. 82, Art. no. 101499, 2020,
     doi: 10.1016/j.compenvurbsys.2020.101499.

[12] J. Van Den Hoek and H. K. Friedrich, "Satellite-based human settlement
     datasets inadequately detect refugee settlements: A critical assessment
     at thirty refugee settlements in Uganda," Remote Sens., vol. 13,
     no. 18, Art. no. 3574, 2021, doi: 10.3390/rs13183574.

[13] G. W. Gella et al., "Mapping of dwellings in IDP/refugee settlements
     from very high-resolution satellite imagery using a mask region-based
     convolutional neural network," Remote Sens., vol. 14, no. 3, Art.
     no. 689, 2022, doi: 10.3390/rs14030689.

[14] K. Wernicke, "Deep learning for refugee camps--Mapping settlement
     extents with Sentinel-2 imagery and semantic segmentation," M.S.
     thesis, Dept. Remote Sens., Univ. Wurzburg, Wurzburg, Germany, 2023.
     [Online]. Available: https://elib.dlr.de/196349/
```

### Erratas eliminadas
- ~~"Haas & Ahlstrom (2021)"~~ → No existe. Paper real: [12] Van Den Hoek & Friedrich
- ~~"Sprohnle et al. (2017)"~~ → Autores reales: [5] Wendt, Lang & Rogenhofer
- ~~"Wurm, Stark, Zhu (2017)"~~ → Autores reales: [6] Wurm, Weigand, Schmitt, Geiss, Taubenboeck
- ~~"Friedrich & Witmer (2021)"~~ → Co-autor real: Van Den Hoek; año real: 2020
- ~~Pesaresi et al. (2016)~~ → Año real: 2013
- ~~Masoni (2021)~~ → Eliminado (journal predatorio, no indexado)

### Fuentes de datos
- Microsoft Planetary Computer: https://planetarycomputer.microsoft.com/
- UNHCR Data: https://data.unhcr.org/
- OpenStreetMap Overpass API: https://overpass-api.de/
- Sentinel-2 L2A: ESA Copernicus via Planetary Computer STAC

### Herramientas y software
- Python 3.x con numpy, scipy, scikit-learn, scikit-image, rasterio
- pystac-client + planetary-computer para acceso STAC
- PyTorch + torchvision para CNN (ResNet-18)
- matplotlib para figuras

---

## 12. Posicionamiento Científico

### Lo que aporta este trabajo

La literatura existente muestra piezas del rompecabezas:
- NDBI tiene limitaciones (Zha 2003, Rasul 2018)
- Hay límites de resolución (Radoux 2016)
- Productos globales fallan en campos (Van Den Hoek & Friedrich 2021)
- CNN no generaliza entre regiones (Gella 2022)
- Materiales naturales son invisibles (Pesaresi 2013)

**Nadie conectó:**
```
física del sensor → contraste espectral → separabilidad estadística → performance ML
```

**Ese es nuestro aporte.**

No descubrimos un detector mejor. Descubrimos que la cadena
**entorno → señal disponible → detector posible** determina todo,
y que esa cadena no es inferible desde la imagen misma.

### Implicación práctica
Antes de diseñar un pipeline de detección de campos de refugiados con
Sentinel-2, hay que calcular ΔNDBI del entorno. Si ΔNDBI ≤ 0, la detección
espectral no es viable. No vale la pena invertir en modelos más complejos.

### Extensión futura
Sentinel-1 SAR responde a rugosidad/geometría, no reflectancia. Podría
recuperar señal geométrica en biomas húmedos donde Sentinel-2 falla.
Los campos tienen estructura regular (grillas de refugios) que produce
backscatter característico incluso cuando son espectralmente invisibles.

---

## 13. Guías IEEE para Publicación

### 13.1 Opciones de venue

| Característica | IEEE GRSL | IEEE JSTARS |
|---------------|-----------|-------------|
| Límite páginas | **5 páginas** (estricto) | Sin límite duro; overlength $200/pág después de 6 |
| Longitud típica | 3-5 páginas | 10-15 páginas |
| Figuras | 3-5 típico | Sin límite explícito |
| Abstract | Hasta 250 palabras | 150-250 palabras |
| Keywords | 3-5 | 3-4 |
| Open Access | Opcional ($2,645) | Obligatorio CC BY 4.0 |
| APC | Gratis para miembros GRSS (desde jun 2025) | $1,800 |
| Revisión | ~30 días | ~10 semanas |
| Portal | IEEE Author Portal | ScholarOne |

**Recomendación:** GRSL si podemos comprimir a 5 páginas. JSTARS si necesitamos
más espacio para los resultados texturales + validación.

### 13.2 Formato de referencias IEEE
- Numeradas secuencialmente `[1]`, `[2]`, etc.
- En texto: `[1]`, `[1], [3]`, `[1]-[5]`
- Autores: `A. B. Apellido` (iniciales + apellido)
- Título artículo: entre comillas, sentence case
- Journal: en itálica, abreviado
- DOI al final: `doi: 10.xxxx/xxxxx.`
- Máximo 6 autores; si más, "et al."
- Ver sección 11 para las 14 referencias ya formateadas en IEEE style.

### 13.3 Figuras — requisitos técnicos
- **DPI mínimo:** 300 dpi color/grayscale, 600 dpi line art
- **Ancho mínimo:** 1050 px (1 columna), 2150 px (ancho página)
- **Formatos aceptados:** PNG, PDF, EPS, TIFF
- **NO aceptado:** JPEG (excepto fotos de autor), GIF, BMP
- **Tipografía en figuras:** ~9-10 pt, Helvetica/Times New Roman/Arial
- **Tamaño máximo:** 7.16 x 8.8 pulgadas (182 x 220 mm)
- **NOTA:** Nuestras figuras actuales están a 300 DPI en PNG — VÁLIDO para GRSL.
  Para JSTARS necesitaríamos 600 DPI o regenerar en PDF/EPS.

### 13.4 Idioma
- **American English** (referencia: Merriam-Webster)
- "analyze" no "analyse", "color" no "colour", "modeling" no "modelling"
- Gramática: referencia Chicago Manual of Style
- IEEE recomienda edición profesional para autores no-nativos

### 13.5 Requisitos adicionales
- ORCID obligatorio para todos los autores
- Data/code sharing: recomendado pero no obligatorio
- Repositorios recomendados: IEEE DataPort, Zenodo, figshare
- Code: Code Ocean para cápsulas reproducibles
- Estructura recomendada: Title, Abstract, Keywords, Introduction,
  Methodology, Results, Discussion, Conclusion, References

### 13.6 Checklist pre-submission
- [ ] Abstract ≤ 250 palabras, sin abreviaciones ni ecuaciones
- [ ] 3-5 keywords
- [ ] Figuras en PNG/PDF/EPS, ≥ 300 DPI, texto 9-10 pt
- [ ] Referencias en formato IEEE numerado con DOIs
- [ ] American English consistente
- [ ] ORCID de todos los autores
- [ ] Dos versiones: 2-columnas (final) + 1-columna (revisión)
- [ ] Data availability statement (recomendado)
- [ ] Todas las citas verificadas contra DOIs
