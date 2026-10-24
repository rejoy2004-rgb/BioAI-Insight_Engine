# Data dictionary - `data/cancer.csv`

**Provenance: not documented in the repository.** The column names, 699 rows, 16 missing `Bare.nuclei` values and 458/241 class split match the well-known *Wisconsin Breast Cancer (Original)* biopsy data (also distributed as `biopsy` in R's MASS and `BreastCancer` in mlbench; originally collected by Dr. W. H. Wolberg, University of Wisconsin Hospitals). That match is my inference from the schema and counts, **not a verified provenance statement** - the owner should add the exact source and licence they downloaded it from.

Each row describes a fine-needle aspirate (FNA) of a breast lump, scored by a pathologist.

| Column | Type | Range | Meaning (standard description of the original attributes) | Used as |
|---|---|---|---|---|
| `Id` | int | - | Sample code number. 54 values repeat across rows. | **Not a feature.** Only used to keep rows with the same `Id` on one side of every split. |
| `Cl.thickness` | int | 1-10 | Clump thickness | feature |
| `Cell.size` | int | 1-10 | Uniformity of cell size | feature |
| `Cell.shape` | int | 1-10 | Uniformity of cell shape | feature |
| `Marg.adhesion` | int | 1-10 | Marginal adhesion | feature |
| `Epith.c.size` | int | 1-10 | Single epithelial cell size | feature |
| `Bare.nuclei` | float | 1-10 | Bare nuclei; **16 missing** | feature (median-imputed *inside* the training pipeline) |
| `Bl.cromatin` | int | 1-10 | Bland chromatin (spelling as in the file) | feature |
| `Normal.nucleoli` | int | 1-10 | Normal nucleoli | feature |
| `Mitoses` | int | 1-10 | Mitoses | feature |
| `Class` | int | {0, 1} | **0 = benign (458), 1 = malignant (241)**. Encoding inferred from the class counts (the file has no legend). | target |

Feature order used everywhere (training, saved bundle, API): the order in the table above.

## Data-quality facts (computed by `src/preprocessing.data_quality_report`, also shown in the UI)

* 699 rows; 34.5 % malignant (moderate imbalance, handled with CV on ROC-AUC/recall and a recall-targeted threshold).
* 16 missing values, all in `Bare.nuclei`. **No row is dropped**; training imputes with the training-fold median. Inference *rejects* a request with a missing value rather than guessing.
* 236 rows duplicate another row's feature vector (the original dataset contains repeated samples). They are kept (deleting them would change the study population) but their effect is measured: ROC-AUC on the test rows whose feature vector never occurs in training is reported next to the headline number.
* 54 rows share an `Id` with another row -> grouped splitting.
* No constant features. Values outside 1-10, infinities, non-numeric columns, missing columns or unexpected targets raise `DataValidationError`.

## Limits of the data

Single-institution, 1990s-era data; ordinal pathologist scores (not images); no demographics, so subgroup performance cannot be assessed. Features describe an *existing* lump that has already been aspirated - this is not a screening or risk-prediction dataset.
