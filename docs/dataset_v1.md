# Dataset V1

## Sources
| File | Path |
|---|---|
| Label source | `PHA_Jung _lvl1_fraction_v0.csv` |
| Metadata | `crosswalk_image_habitat_v0.csv` |

## Statistics
| Metric | Value |
|---|---|
| Total records | 122,961 |
| Images not matched | 2,742 |
| Unique species | 8,885 |
| Unique families | 240 |
| Unique genres | 2,013 |

## Label Columns
`1`, `2`, `3`, `4`, `8`, `12`, `14`, `9_11`, `5_13_15`, `6_7`

---

# Cross-Validation — cross_version_1_FAM

## Configuration
| Parameter | Value |
|---|---|
| Folds | 3 |
| Split strategy | Family |
| Random state | 42 |

## Per-Fold Statistics
| Fold | Train images | Val images | Train species | Val species | Train families | Val families | Train genres | Val genres |
|---|---|---|---|---|---|---|---|---|
| 0 | 85,071 | 37,890 | 6135 | 2750 | 157 | 83 | 1407 | 606 |
| 1 | 68,361 | 54,600 | 4951 | 3934 | 163 | 77 | 1095 | 918 |
| 2 | 92,490 | 30,471 | 6684 | 2201 | 160 | 80 | 1524 | 489 |

