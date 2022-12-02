
## How to run


There are different python and bash files to run different DP algorithms on MNIST dataset.


#### Vanilla/Original Model

```
python Vanilla_model.py 
```


#### PATE
1. Train all teacher models:
```
bash run_all_teachers_PATE.sh
```

2. Train Student model:
```
python PATE_Student.py
```

3. Train Query-based Student PATE model:

```
bash run_querybased_PATE.sh 
```


#### DP-SGD

```
python DPSGD.py
```


#### LP-2ST
(Runs multiple times)

```
bash run_multiple_times_LPMST.sh 
```


#### ALIBI
(Runs multiple times)

```
bash run_multiple_times_ALIBI.sh
```


## Requirements:


| Package  | Version |
| ------------- | ------------- |
| absl-py  | 1.2.0  |
| numpy  | 1.21.2  |
| scipy  | 1.7.3  |
| simple-parsing  | 0.0.20  |
| six  | 1.16.0  |
| torch  | 1.10.0  |
