# Progressive Graph Learning for Open-Set Domain Adaptation

### Requirements
- Python 3.6
- Pytorch 1.3


### Datasets
The links of datasets will be released afterwards,
- Syn2Real-O (VisDA-18) https://github.com/VisionLearningGroup/visda-2018-public/tree/master/openset
- VisDA-17 https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification
- Office-home https://www.hemanthdv.org/officeHomeDataset.html


### Training
The general command for training is,
```
python3 train.py
```
Change arguments for different experiments:
- dataset: "home" / "visda" / "visda18"
- batch_size: mini_batch size
- beta: The ratio of known target sample and Unk target sample in the pseudo label set
- EF : Enlarging Factor α
- num_layers: GNN's depth
- adv_coeff: adversarial loss coefficient γ
- node_loss: node classification loss μ
For the detailed hyper-parameters setting for each dataset, please refer to Section 5.2 and Appendix 3.  

Remember to change dataset_root to suit your own case

The training loss and validation accuracy will be automatically saved in './logs/', which can be visualized with tensorboard.
The model weights will be saved in './checkpoints'

### Without Pseudo-labeling Results

VisDA-18

Plane | Bike | Bus | Car | Horse | Knife | Motorcycle | Person | Plant | SkateB | Train | Truck | Unk | OS^* | OS |
------|------| --- | --- | ----- | ----- | ---------- | ------ | ----- | ------ | ----- | ----- | --- | ---- | -- |
0.640 | 0.695|0.501|0.509| 0.795 | 0.126 | 0.945      | 0.585  | 0.742 | 0.588  | 0.702 | 0.081 | 0.542|0.573 | 0.575|


### TODO List
- [X] Update the GradReverse layer for Pytorch 1.4

- [ ] Update detail config file for datasets

     - [ ] VisDA-18
     - [ ] VisDA-17
     - [ ] Office-home
     
- [ ] Fix progress bar



