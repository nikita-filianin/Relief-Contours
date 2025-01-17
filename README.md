# Relief-Contours

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data storage](#data-storage)
- [User interactions](#user-interactions)

## Description
This repository contains a ready-to-use UNet model trained on HiRISE images of the Martian North Pole. The model itself is wrapped in a pipeline that takes care of PDS database access, image pre-processing, segmentation, and contour derivation. User input can be provided either through CLI or by directly editing the [config.yaml](./Configs/config.yaml).

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Relief-Contours.git
    ```
2. Install PyTorch and make sure cuda is enabled:

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    ```console
    $ python
    >>> import torch
    >>> torch.cuda.is_available()
    True
    ```
3. Create a virtual environment & install the rest of the required dependencies

    ```
    conda env create -f contours.yml
    ```

## Data storage
Folders for storing input data (.JP2), intermediate results & final contours (.npy) are created automatically according to the following logic. It is strongly suggested that this structure is maintained to avoid unexpected behavior. 
```
├───data
│   ├───images
│   │   ├───JP2----------------------> original HiRISE .JP2 products
│   │   └───png----------------------> downscaled .png counterparts
│   └───results
│       ├───npy----------------------> np.ndarray(s) of scarp coordinates
│       └───plots--------------------> images that illustrate the found contours
└───Relief-Contours
```

## User interactions
All necessary arguments are set to reasonable default values in [config.yaml](./Configs/config.yaml), which can either be edited directly, or by passing the desired changes through CLI:

```console
$ python main.py --help
usage: main.py [-h] [--imID] [--STORAGE_PATH] [--CONFIG_PATH] [--MODEL_PATH] [--BAND] 
               [--RESIZE_FACTOR] [--TILE_SIZE] [--STATS] [--NUM_CHANNELS] [--NUM_CLASSES]
               [--THRESHOLD] [--GAP_SIZE] [--ISLAND_SIZE] [--CNT_DENSITY]
```
We can categorize arguments according to the functions they are associated with:
<details>
<summary>General</summary>

These basic arguments are used to initialize the folder tree, load default constants, load the model and retrieve the image(s) from PDS database.

| Argument | Dtype | Values  | Description |         
|----------|-------|---------|-------------|
| `imID` | str | 'aaa_bbbbbb_cccc' | ID of a HiRISE image to be processed. |
| `STORAGE_PATH`  | str | ... | path to the storage folder |
| `CONFIG_PATH` | str | ... | path to the config.yaml file, which contains the default argument values |
| `MODEL_PATH`  | str | ... | path to UNet .pth weights file. User may change this if he/she wants to use his/her own pretrained weights |
</details>

<details>
<summary>Gdal</summary>

Since .JP2 HiRISE products are too large to load into python, we use [osgeo. gdal](https://gdal.org/api/python_bindings.html) to first downscale them & then to convert them to .png format.

| Argument  | Dtype | Values | Description |         
|-----|-------|---------|-------------|
| `BAND` | int | $\\{1, 2, 3\\}$ | Band to read from the .JP2 image. Since we're primarily dealing with single channel (RED) HiRISE images this value is set to 1. |
| `RESIZE_FACTOR` | float | $x \in (0,1]$ | The downscaling factor applied to the .JP2 image. This helps significantly reduce computational times. |
| `STATS` | bool | $\\{true , false\\}$| Call to display .JP2 image metadata | 

</details>

<details>
<summary>Tiling</summary>

Even after downscaling the image is still too large to be segmented by UNet in one go. It is common practice to partition the image into smaller patches, segment each one of them separately and then recombine the outputs into a complete full-image mask.

| Argument  | Dtype | Values | Description |         
|-----|-------|---------|-------------|
| `TILE_SIZE` | int | $x \\% 2 = 0$ |  Defines the HxW of the patches (tiles) that are cut from the downscaled .png. Strongly suggested that this value matches the dimensions of the patches used during UNet training |

</details>

<details>
<summary>UNet</summary>

In our case, UNet operates on .png images of dimensions `TILE_SIZE`x`TILE_SIZE` that have only 1 channel & 1 associated class. These arguments should mirror the dataset used in training. 

| Argument  | Dtype | Values | Description |         
|-----|-------|---------|-------------|
| `NUM_CHANNELS` | int | $\\{1, 2, 3\\}$ | Number of channels in an input image provided to UNet |
| `NUM_CLASSES` | int | $\\{1, \infty \\}$ | Number of classes in the output mask |
| `THRESHOLD` | float |$x \in (0,1)$ | Threshold for the output mask; everything below this value is set to 0, everything above to 1 |
</details>

<details>
<summary>Skimage</summary>

Skimage is used for dealing with noisy predictions, i.e. it helps to clean up the masks by filling gaps and removing islands. Values for both arguments represent the pixel volume of an area to be removed/filled.

| Arg | Dtype | Values | Description |         
|-----|-------|---------|-------------|
| `GAP_SIZE` | int | ... | Any gap in the mask smaller than this value will be filled |
| `ISLAND_SIZE` | int | ... | Any island in the mask smaller than this value will be removed |
</details>

<details>
<summary>Contour detection </summary>

Final contours are derived using an OpenCV.

| Argument  | Dtype | Values | Description |         
|-----|-------|---------|-------------|
| `CNT_DENSITY` | int | ... |  Gap between knot points along the contour (pixels); the higher the value, the less points are kept, the smoother the contour |

</details>
