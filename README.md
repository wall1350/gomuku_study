# AlphaZero-Gomoku-forbidden-step （gomoku_study using tensorflow）

Forked from [zouyih/AlphaZero_Gomoku-tensorflow](https://github.com/zouyih/AlphaZero_Gomoku-tensorflow) with some changes:  

* rewrite the forbidden move detect (not yet)
* added a GUI

## How to Run
* Play with AI
```
python human_play.py
```
* Play with parallel AI (-np : set number of processings, take care of OOM !)
```
mpiexec -np 3 python -u human_play_mpi.py 
```
* Train from scratch
```
python train.py
```
* Train in parallel
```


### Example of Game

