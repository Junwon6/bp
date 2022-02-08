| ID                                  | CASE                                |
| :---------------------------------- | :---------------------------------- |
| DATASET                             | d00, d01, d02                       |
| DETAIL                              | s00, s01                            |
| MODEL                               | m01, m02, m03, m04, m05             |
| EXPERIMENT                          | e01 ~ e10                           |
| DATAFRAME                           | df01, df02, df03, df04, df05, df06  |

| DATABASE ID                         | DESCRIPTION                         |
| :---------------------------------- | :---------------------------------- |
| d00                                 | VITAL-UQ                            |
| d01                                 | MIMIC-II                            |
| d02                                 | E4                                  |

| DETAIL ID                           | DESCRIPTION                         |
| :---------------------------------- | :---------------------------------- |
| s00                                 | raw slice                           |
| s01                                 | unit slice                          |

| MODEL ID                            | DESCRIPTION                                      |
| :---------------------------------- | :----------------------------------------------- |
| m01                                 | 1-layer lstm                                     |
| m02                                 | multi-layer cnn                                  |
| m03                                 | 2-layer lstm                                     |
| m04                                 | 1-layer bidirectional lstm                       |
| m05                                 | stacked, residual, bidirectional lstm            | 

| EXPERIMENT ID                       | DESCRIPTION                                                                            |
| :---------------------------------- | :------------------------------------------------------------------------------------- |
| e01                                 |  df01                                                                                  |
| e02                                 |  df01 + df02                                                                           |
| e06                                 |  df04                                                                                  |
| e07                                 |  df04 + df05                                                                           |
| e08                                 |  df06                                                                                  |
| e09                                 |  df04 + df06                                                                           |
| e10                                 |  df04 + df05 + df06                                                                    |

| DATAFRAME ID                        | DESCRIPTION                                                                            |
| :---------------------------------- | :------------------------------------------------------------------------------------- |
| df01                                | raw signal                                                                             |
| df02                                | differential values                                                                    |
| df03                                | static feature                                                                         |
| df04                                | unit pattern signal                                                                    |
| df05                                | unit differential values                                                               |
| df06                                | unit static feature                                                                    |
