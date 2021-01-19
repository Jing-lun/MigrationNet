# MigrationNet: Underground Pipeline Interpretation with PyTorch

![dataset](https://www.dropbox.com/s/tv0ne4bgiql7nco/tgrs_models.tar.gz?dl=0)


Built based on the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch.


## Usage
**Note : Use Python 3**
### Prediction

You can easily test the output with our dataset:

`python predict_mat.py -i path/to/test -o predict.png -m path/to/checkpoint
`



### Training

```shell script
> python train_mat.py -f path/to/checkpoint -e 200 -b 1 -l 0.000005 -s 0.25 -x path/to/train -y path/to/gt
```
