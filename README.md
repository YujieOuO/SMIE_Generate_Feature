# SMIE_Generate_Feature
The pipeline to general ST-GCN visual features for [SMIE](https://github.com/YujieOuO/SMIE)

After splitting the data of different unseen classes using [split.py](https://github.com/YujieOuO/SMIE/blob/main/split.py), 
utilize gen_feature.py to obtain the corresponding visual features for ST-GCN.

Modify gen_config.py to select a different split_id for different datasets and customize the storage location.

```bash
$ python gen_feature.py
```
