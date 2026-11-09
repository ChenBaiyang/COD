# COD
Baiyang Chen, Zhong Yuan, Dezhong Peng, Xiaoliang Chen, and Hongmei Chen, "[Consistency-guided semi-supervised outlier detection in heterogeneous data using fuzzy rough sets](Paper/2024-COD.pdf)," Applied Soft Computing, vol. 165, article 112070, 2024. DOI: [10.1016/j.asoc.2024.112070](https://doi.org/10.1016/j.asoc.2024.112070).

## Abstract
Outlier detection aims to find objects that behave differently from the majority of the data. Semi-supervised detection methods can utilize the supervision of partial labels, thus reducing false positive rates. However, most of the current semi-supervised methods focus on numerical data and neglect the heterogeneity of data information. In this paper, we propose a consistency-guided outlier detection algorithm (COD) for heterogeneous data with the fuzzy rough set theory in a semi-supervised manner. First, a few labeled outliers are leveraged to construct label-informed fuzzy similarity relations. Next, the consistency of the fuzzy decision system is introduced to evaluate attributes’ contributions to knowledge classification. Subsequently, we define the outlier factor based on the fuzzy similarity class and predict outliers by integrating the classification consistency and the outlier factor. The proposed algorithm is extensively evaluated on 20 freshly proposed datasets. Experimental results demonstrate that COD is better than or comparable with the leading outlier detectors.

## Framework
![image](Paper/COD_Framework.png)

## Datasets
The paper evaluates 20 datasets; 15 are provided here: 2 categorical, 2 mixed, and 11 numerical datasets. Sample counts range from 351 to 11,183, with anomaly ratios from 2.0% to 35.9%.

Each NPZ contains `X` (features), `y` (1=anomaly, 0=normal, used to simulate known labels and compute evaluation metrics), and `nominals` (a Boolean column mask). Numerical columns are already min-max normalized; categorical codes need not lie in [0, 1]. The experiment script removes columns with standard deviation at most `1e-6`, using the supplied mask for all remaining columns.


| No 	|   Datasets  	| #Samples 	| #Attributes 	| #Outlier 	| %Outlier 	|   Category  	|   DataType  	|
|:--:	|:-----------:	|:--------:	|:-----------:	|:--------:	|:--------:	|:-----------:	|:-----------:	|
|  1 	|  annthyroid 	|   7200   	|      6      	|    534   	|   7.4%   	|  Healthcare 	|  Numerical  	|
|  2 	|  Arrhythmia 	|    452   	|     279     	|    66    	|   14.6%  	|   Medical   	|    Mixed    	|
|  3 	|   breastw   	|    683   	|      9      	|    239   	|   35.0%  	|  Healthcare 	|  Numerical  	|
|  4 	|    cardio   	|   1831   	|      21     	|    176   	|   9.6%   	|  Healthcare 	|  Numerical  	|
|  5 	|  Ionosphere 	|    351   	|      32     	|    126   	|   35.9%  	| Oryctognosy 	|  Numerical  	|
|  6 	| mammography 	|   11183  	|      6      	|    260   	|   2.3%   	|  Healthcare 	|  Numerical  	|
|  7 	|  Mushroom1  	|   4429   	|      22     	|    221   	|   5.0%   	|    Botany   	| Categorical 	|
|  8 	|  Mushroom2  	|   4781   	|      22     	|    573   	|   12.0%  	|    Botany   	| Categorical 	|
|  9 	|     musk    	|   3062   	|     166     	|    97    	|   3.2%   	|  Chemistry  	|  Numerical  	|
| 10 	|  optdigits  	|   5216   	|      64     	|    150   	|   2.9%   	|    Image    	|  Numerical  	|
| 11 	|  PageBlocks 	|   5393   	|      10     	|    510   	|   9.5%   	|   Document  	|  Numerical  	|
| 12 	|     Sick    	|   3613   	|      29     	|    72    	|   2.0%   	|   Medical   	|    Mixed    	|
| 13 	|   Waveform  	|   3443   	|      21     	|    100   	|   2.9%   	|   Physical  	|  Numerical  	|
| 14 	|     Wilt    	|   4819   	|      5      	|    257   	|   5.3%   	|    Botany   	|  Numerical  	|
| 15 	|    yeast    	|   1484   	|      8      	|    507   	|   34.2%  	|   Biology   	|  Numerical  	|


## Installation

Python 3.8+ with NumPy and scikit-learn:

```bash
python -m pip install -r requirements.txt
```

COD runs on CPU using NumPy float64.

## Usage

```python
import numpy as np
from model import COD

with np.load("data/breastw.npz") as data:
    X, y, nominals = data["X"], data["y"], data["nominals"]

# Simulate five known anomaly labels for this example.
known = np.random.RandomState(2023).choice(np.flatnonzero(y), 5, replace=False)
train_y = np.zeros(len(X), dtype=int)
train_y[known] = 1
model = COD().fit(X, train_y, nominals)
scores = model.predict_score()  # Scores for all fitted rows, in input order.
```

`X` has shape `(n, m)`; `(n, m, 1)` is also accepted. Provide `nominals` as a Boolean/0-1 column mask. `fit` applies min-max normalization to numerical columns over the fitted universe and leaves categorical codes unchanged. Constant numerical columns map to zero; input arrays are not modified.

`train_y=1` identifies a known outlier; `0` means unlabeled. Hidden anomaly labels must remain zero. `fit` returns the model, and larger scores indicate stronger outlier evidence.

COD is transductive: `predict_score()` returns scores for the fitted universe. Fit a new model to analyze a different universe.

- `n_neg=100`: maximum candidate inliers per attribute, selected from unlabeled rows by ascending mean distance to the full universe; ties follow input row order.
- `batch_size=256`: relation rows per block. Every block uses the full universe as its reference.
- `normalize=True`: apply min-max preprocessing. Set `normalize=False` to use numerical inputs already normalized to [0, 1].

Numerical radii maximize Eq. (6) over `0.001, 0.002, ..., 0.999`, choosing the smallest radius on ties. Singleton outlier factors are weighted by classification consistency (Eqs. 12-14). Learned `radii_` (`-1` for categorical columns) and `consistencies_` are available on the fitted model.

## Experiments

Run the 15 bundled datasets with ten repetitions each:

```bash
python run_COD.py
```

Select datasets, seeds, and an output directory:

```bash
python run_COD.py --datasets breastw mushroom1 arrhythmia --seeds 2023 --output-dir outputs/run1
```

Defaults are five known outliers, 100 candidate inliers per attribute, and seeds `2023` through `2032`. Known outliers participate in fitting but are excluded from evaluation. Metrics are AUROC and average precision (AP).

Normalization is enabled by default. Pass `--no-normalize` to keep the supplied pre-normalized values unchanged.

Each run creates a new directory under `outputs/`; existing directories are refused. `results_COD.csv` contains full-precision per-seed metrics and known-outlier indices. `summary_COD.csv` contains dataset means, population standard deviations (`ddof=0`), and the unweighted mean across datasets.

`resuts_COD.csv` contains historical aggregate results. Results from this script differ from that file and the published tables.

## Citation
If you find the code or datasets useful in your research, please consider citing:
```
@article{Chen2024COD,
  title = {Consistency-guided semi-supervised outlier detection in heterogeneous data using fuzzy rough sets},
  journal = {Applied Soft Computing},
  volume = {165},
  pages = {112070},
  year = {2024},
  issn = {1568-4946},
  author = {Baiyang Chen and Zhong Yuan and Dezhong Peng and Xiaoliang Chen and Hongmei Chen},
  doi = {10.1016/j.asoc.2024.112070}
}
```

## License

The code is provided under the [MIT License](LICENSE).

## Contact
If you have any questions, please contact farstars@qq.com.
