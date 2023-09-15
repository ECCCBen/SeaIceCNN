# SeaIceCNN

This repository contains the techniques and code referenced in the following publication:


>  Benoit Montpetit, Benjamin Deschamps, Joshua King & Jason Duffe (2023) Assessing the Parameterization of RADARSAT-2 Dual-polarized ScanSAR Scenes on the Accuracy of a Convolutional Neural Network for Sea Ice Classification: Case Study over Coronation Gulf, Canada, Canadian Journal of Remote Sensing, 49:1, DOI: 10.1080/07038992.2023.2247091 


Open-Access Publication: [![Static Badge](https://img.shields.io/badge/Canadian_Journal_of_Remote_Sensing-blue)](https://doi.org/10.1080/07038992.2023.2247091)

Open-Access Dataset: [![DOI](zenodo.svg)](link to zenodo)
use the auto-update badge thingy https://gist.github.com/seignovert/ae6771f400ca464d294261f42900823a

## Abstract

Arctic amplification has many impacts on sea-ice extent, thickness, and flux. It becomes critical to monitor sea-ice conditions at finer spatio-temporal resolution. We used a simple convolutional neural network (CNN) on the RADARSAT-2 dual-polarized ScanSAR wide archive available over Coronation Gulf, Canada, to assess which SAR parameter improves model performances to classify sea ice from water on a large volume of data covering 11 years of ice and surface water conditions. An overall accuracy of 90.1% was achieved on 989 scenes of 100% ice cover or ice-free conditions. An accuracy of 86.3% was achieved on the last year of data (134 scenes) which was kept out of the training process to test the model on an independent dataset. A better accuracy is obtained at lower incidence angles and the HH polarization provides the most information to classify ice from water. To achieve the best accuracy, the incidence angle and the noise equivalent sigma-nought had to be included as input to the model. A comparison done with the ASI passive microwave product shows similar errors in total sea ice concentration when using the Canadian Ice Service regional charts as reference. Nonetheless, errors from both datasets differ and the CNN outputs show greater potential to reduce masked areas, given the better spatial resolution, enabling data classification closer to land and identify features not captured by the ASI dataset.

<p align="center">
    <img src="https://www.tandfonline.com/na101/home/literatum/publisher/tandf/journals/content/ujrs20/2023/ujrs20.v049.i01/07038992.2023.2247091/20230912/images/medium/ujrs_a_2247091_f0007_c.jpg">
</p>

<p align="center">
    <i>Figure 7. (a) CIS regional chart valid July 13, 2020; (b) ASI SIC product of July 13, 2020; (c) MODIS Terra image of July 14; (d) RSAT-2 RGB: HH-HV-HV composite image of July 13,2020; and (e) CNN model output for July 13, 2020.)</i>
</p>

> **Warning**
> Access to RADARSAT-2 data products is not included with this repository. RADARSAT-2 Data and Products © Maxar Technologies Ltd. (2016-2020) – All Rights Reserved. RADARSAT is an official mark of the Canadian Space Agency.

## Environment Configuration

Use [miniconda](https://docs.conda.io/projects/miniconda/en/latest/), [mamba](https://mamba.readthedocs.io/en/latest/) or [anaconda](https://www.anaconda.com/download) to recreate the runtime environment:


```
conda env create -n seaicecnn -f environment.yml
conda activate seaicecnn
```
> **Warning** 
> The provided environment.yml file was generated on Windows 10 and may behave differently on Linux or Mac systems.

> **Warning** 
> if you want to train the model yourself, it is suggested to use an Nvidia GPU as the environment is setup to make use of CUDA acceleration for Tensorflow. You may still train the model without a GPU but it will take considerably longer.

## Data Preparation

To download the datasets used by the notebooks, use the following link
[zenodo link](zenodo) and place the data inside the correct folder:

```
Data
├── stuff
├── things
|   └── some file
└── others
```


## Exploring the Notebooks

After setting up the environment and data, you may wish to look first at the Table of Contents in [index notebook filename](./index.ipynb) to discover which parts of the code interest you. In order to launch the Table of Contents notebook on your local system, use the following command while inside the activated `seaicecnn` environment:

```
jupyter notebook <name-of-index.ipynb>
```

# Markdown Examples

Image (no control over alignment - check the fig7 html above for alignment-tweaking capability)
![this is an image](https://www.tandfonline.com/na101/home/literatum/publisher/tandf/journals/content/ujrs20/2023/ujrs20.v049.i01/07038992.2023.2247091/20230912/images/medium/ujrs_a_2247091_f0007_c.jpg)

Equation inline using `$` notation $\sqrt{3x - 1}+(1+x)^2$

Equation block using `$$` notation

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$