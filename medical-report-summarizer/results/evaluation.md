# Evaluation results

Average over 5 reports. ROUGE values are F1 scores (0 to 1, higher is better).

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Fact coverage | Faithful | Avg words | Readability | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lead baseline | 0.320 | 0.157 | 0.219 | - | - | 77 | 53.3 | 0.00 |
| extractive | 0.525 | 0.302 | 0.376 | 52.9% | 5/5 | 76 | 40.2 | 0.02 |
| hybrid | 0.429 | 0.192 | 0.273 | 36.1% | 5/5 | 44 | 23.8 | 3.03 |
| abstractive | 0.392 | 0.155 | 0.266 | 41.4% | 5/5 | 51 | 21.9 | 2.03 |
