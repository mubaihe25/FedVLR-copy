# FedVLR

[![Static Badge](https://img.shields.io/badge/AAAI-N/A-red?style=plastic&logo=iclr&labelColor=%2386C166&color=grey)](https://openreview.net/forum?id=BQByYBJ9Wl) | [![Static Badge](https://img.shields.io/badge/OpenReview-FedVLR-red?style=plastic&logo=OpenReivew&labelColor=%23FCFAF2&color=grey)](https://openreview.net/forum?id=BQByYBJ9Wl) | [![Static Badge](https://img.shields.io/badge/arxiv-2410.08478-red?style=plastic&logo=arxiv&logoColor=white&labelColor=%23C73E3A&color=grey)](https://arxiv.org/abs/2410.08478)

> This project is the code of Our Paper "**Federated Vision-Language-Recommendation with Personalized Fusion**"

## Requirements

1. The code is implemented with `Python ~= 3.8` and `torch~=2.4.0+cu118`;
2. Other requirements can be installed by `pip install -r requirements.txt`.
3. For multimodal dataset processing, please refer to [mtics/MMRec_Dataset_Preprocessing](https://github.com/mtics/MMRec_Dataset_Preprocessing)

## Quick Start

1. Put datasets into the path `[parent_folder]/datasets/`;

2. For quick start, please run:
    ``````
    python main.py --alias MMFedRAP --dataset movielens --data_file ml-100k.dat \
        --lr 1e-3 --l2_reg 1e-5 --seed 0
    ``````

3. All multimodal FedRecs start with 'MM'.

## Thanks

In the implementation of this project, we drew upon the following resources: [MMRec](https://github.com/enoche/MMRec), [RecBole](https://github.com/RUCAIBox/RecBole) and [Tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec?tab=readme-ov-file). 
We sincerely appreciate their open-source contributions!

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))