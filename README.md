# COD
Baiyang Chen and Zhong Yuan* and Dezhong Peng and Xiaoliang Chen and Hongmei Chen, "[Consistency-guided semi-supervised outlier detection in heterogeneous data using fuzzy rough sets](https://doi.org/10.1016/j.asoc.2024.112070)," Applied Soft Computing, vol. 103, p. 112070, 2024, DOI: 10.1016/j.asoc.2024.112070

## Abstract
Outlier detection aims to find objects that behave differently from the majority of the data. Semi-supervised detection methods can utilize the supervision of partial labels, thus reducing false positive rates. However, most of the current semi-supervised methods focus on numerical data and neglect the heterogeneity of data information. In this paper, we propose a consistency-guided outlier detection algorithm (COD) for heterogeneous data with the fuzzy rough set theory in a semi-supervised manner. First, a few labeled outliers are leveraged to construct label-informed fuzzy similarity relations. Next, the consistency of the fuzzy decision system is introduced to evaluate attributes’ contributions to knowledge classification. Subsequently, we define the outlier factor based on the fuzzy similarity class and predict outliers by integrating the classification consistency and the outlier factor. The proposed algorithm is extensively evaluated on 20 freshly proposed datasets. Experimental results demonstrate that COD is better than or comparable with the leading outlier detectors.

## Framework


## Datasets
We use 15 public datasets to assess the model performances, including 2 nominal, 2 mixed, and 11 numerical datasets. The number of samples in a dataset ranges from 351 to 11183, and the ratio of anomalies varies between 2% and 35.9%. The details of the datasets are provided in below table: 


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


## Environment
* cudatoolkit=11.6.0
* numpy=1.23.5
* pandas=1.5.3
* python=3.8.16
* pytorch=1.12.1
* scikit-learn=1.2.0
* scipy=1.9.3
* torchaudio=0.12.1
* torchvision=0.13.1


## Usage
To reproduce the results in the paper:
```
To be updated later.
```
To reproduce the examples in the paper:
```
To be updated later.
```
To run COD on customized datastes with default parameters:
```
To be updated later.
```
To run COD on customized datastes with parameter tuning:
```
To be updated later.
```

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
  author = {Baiyang Chen and Zhong Yuan and Dezhong Peng and Xiaoliang Chen and Hongmei Chen}
  doi = {10.1016/j.asoc.2024.112070},
  }
```
or:

Baiyang Chen, Zhong Yuan*, Dezhong Peng, et al., "[Consistency-guided semi-supervised outlier detection in heterogeneous data using fuzzy rough sets](https://doi.org/10.1016/j.asoc.2024.112070)," Applied Soft Computing, vol. 103, p. 112070, 2024, DOI: 10.1016/j.asoc.2024.112070

## Contact
If you have any questions, please contact farstars@qq.com.
