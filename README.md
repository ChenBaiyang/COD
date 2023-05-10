# COD
The code and dataset for paper "Consistency-guided semi-supervised outlier detection in heterogeneous data using fuzzy rough sets".

## Datasets
We use 15 public datasets to assess the model performances, including 2 nominal, 2 mixed, and 11 numerical datasets. The number of samples in a dataset range from 351 to 11183, and the ratio of anomalies varies between 2% and 35.9%. The details of the datasets are provided in below table: 


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
