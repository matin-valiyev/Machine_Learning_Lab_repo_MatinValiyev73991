# Data Card: UCI Adult Dataset

- **Source:** scikit-learn fetch_openml (name='adult', version=2), originally from the UCI Machine Learning Repository (1994 Census database).
- **License:** Open Access (Public Domain / CC0).
- **Size:** 48,842 rows and 14 feature columns (prior to duplicate dropping).
- **Processed Data Hash (SHA256):** c88faa037b22ef144728bf419ac720315df3a9b4385f4f290620ca55ada77b8f
- **Known Biases:** The dataset reflects 1994 societal norms and employment demographics, heavily skewing toward male representation in higher-income brackets and specific racial majorities. The target label definition (>$50K) is historically static and does not account for inflation or regional cost-of-living adjustments.
- **Privacy Concerns:** While anonymized, the data originates from real census records. Re-identification risks are theoretically possible if cross-referenced with external historical records, though no direct identifiers (names, exact addresses) are present.