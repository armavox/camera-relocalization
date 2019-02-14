# camera-relocalization 
_braingarden.ai testcase_


Transfer learning of few CNN models for camera position regression.

### Dependencies
- torch 1.0.0
- torchvision 0.2.1
- numpy 1.15.4
- pandas 0.23.4
- matplotlib 3.0.2
- PIL 5.4.1
- tqdm 4.29.1
- pretrainedmodels 0.7.4

Full conda/pip environment stored in `brg_test_env.yml`. To install use `$conda env create -f brg_test_env.yml` 
(conda version > 4.6.2).

### Data

Put image folder and csv file into camera_relocalization_sample_dataset folder or specifiy both paths in the main.py options.

### Training

- To train model and watch Train/Val loss use `python main.py`
- To see all possible options, including model choose (GoogleNet or PosNet), learning rate, batch size, etc. use: `python main.py --help`

Examples:
- Training on small dataset (0.1 part of full) to watch convergence:
  ```
  python main.py --model PoseNet --epochs 30 --plot-loss -b 8 --small-dataset
  ```

- Resume training of previously started model:
  ```
  python main.py --model PoseNet -e 30 --batch-size 8 --small-dataset --resume --plot-loss
  ```
### Inference

TODO

### Example

Mostly similar to plain code example presented in the `2019-02-14-posnet.ipynb` Jupyter notebook. 
Difference is that in the notebook transferring of AlexNet and Inception presented, despite of the main.py code, where is 
only Inception net in GoogleNet and PosNet realizations.

Figures for slightly trained model are presented in the `figs/` folder 
