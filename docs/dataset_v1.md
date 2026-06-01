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


---

# Cross-Validation — cross_version_1_GENRE

## Configuration
| Parameter | Value |
|---|---|
| Folds | 3 |
| Split strategy | Genre |
| Random state | 42 |

## Per-Fold Statistics
| Fold | Train images | Val images | Train species | Val species | Train families | Val families | Train genres | Val genres |
|---|---|---|---|---|---|---|---|---|
| 0 | 82,251 | 40,710 | 5960 | 2925 | 214 | 158 | 1338 | 675 |
| 1 | 79,728 | 43,233 | 5812 | 3073 | 205 | 160 | 1342 | 671 |
| 2 | 83,943 | 39,018 | 5998 | 2887 | 204 | 159 | 1346 | 667 |

---

# Cross-Validation — cross_version_1_SPE

## Configuration
| Parameter | Value |
|---|---|
| Folds | 3 |
| Split strategy | Species |
| Random state | 42 |

## Per-Fold Statistics
| Fold | Train images | Val images | Train species | Val species | Train families | Val families | Train genres | Val genres |
|---|---|---|---|---|---|---|---|---|
| 0 | 81,966 | 40,995 | 5944 | 2941 | 224 | 200 | 1678 | 1202 |
| 1 | 81,930 | 41,031 | 5927 | 2958 | 229 | 196 | 1694 | 1190 |
| 2 | 82,026 | 40,935 | 5899 | 2986 | 224 | 199 | 1677 | 1186 |
