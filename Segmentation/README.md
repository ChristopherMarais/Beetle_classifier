# How to use Segmentation script

## Set up the environment and open the guide notebook:
1. Download and install Anaconda [here](https://www.anaconda.com/products/distribution)


2. Download this folder (Segmentation) and all its contents. The most important files to have are:
  - `Ambrosia.py`
  - `environemnt.yml`
  - `guide_notebook.ipynb`
  - `example_dataset.zip` (unzip this in any directory this is just an example dataset to practice on.)


3. Create a new anaconda environment using the `environemnt.yml` file using the following command in the anaconda command prompt terminal:<br />
```
conda env create -f environment.yml
```
This will create an environment called `Seg_310`.

4. Use the anaconda navigator GUI to activate the Seg_310 environment and launch jupyterlab<br /><br />

OR<br /><br />

Use the anaconda command prompt terminal to activate the environment and launch Jupyterlab using the following commands:

```
conda activate Seg_310

jupyter lab
```
This will open jupyter lab in your browser. Navigate to where you downloaded the `guide_notebook.ipynb` in jupyterlab and open it
