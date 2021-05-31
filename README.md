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

### Graph Learning without Pseudo-labeling Results (ResNet-50)
VisDA-18 (alpha=1, beta=0.6)

Plane | Bike | Bus | Car | Horse | Knife | Motorcycle | Person | Plant | SkateB | Train | Truck | Unk | OS^* | OS |
------|------| --- | --- | ----- | ----- | ---------- | ------ | ----- | ------ | ----- | ----- | --- | ---- | -- |
0.437 | 0.807|0.588|0.646| 0.857 | 0.155 | 0.943      | 0.355  | 0.879 | 0.250  | 0.712 | 0.126 | 0.437 |0.553 | 0.563|

Office-Home

Src|R    |     |     |A    |     |     |C    |     |     |P    |     |     |     |
---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
Tar| A   | C   | P   | C   | P   | R   | R   | P   | A   | A   | C   | R   | Avg.|
OS |0.722|0.499|0.763|0.505|0.523|0.826|0.727|0.622|0.599|0.589|0.446|0.752|0.639|
OS*|0.733|0.506|0.777|0.511|0.632|0.840|0.739|0.631|0.607|0.567|0.449|0.765|0.649|
### TODO List
- [X] Update the GradReverse layer for Pytorch 1.4

- [ ] Update detail config file for datasets

     - [ ] VisDA-18
     - [ ] VisDA-17
     - [ ] Office-home
     
- [ ] Fix progress bar



