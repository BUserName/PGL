# Progressive Graph Learning for Open-Set Domain Adaptation

### Requirements
- Python 3.6
- Pytorch 1.3


### Datasets
The links of datasets will be released afterwards,
- Syn2Real-O (VisDA-18)
- VisDA-17
- Office-home


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




