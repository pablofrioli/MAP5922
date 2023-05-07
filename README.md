# fin_ml
  This package of libraries was developed for the classes of the discipline MAP5922 - Machine Learning Applied to Finance from USP (Universidade de São Paulo), 2023/1st semester cohort.<br>

  In the fin_ml library, you will find the transformers classes. One of the them is in frac_diff_transf, which applies fraction differentiation to series of data organized in a Pandas DataFrame, and can be called as a Pipeline in a ColumnTransformer from sckit-learn.

# Installation
  After downloading the files of this repository to your PC, run the following commands from the folder you have put it in:

* python setup.py bdist_wheel
* pip install dist/fin_ml-0.2-py3-none-any.whl

# Requirements
* numpy 1.21.4
* pandas 1.2.3
* ststsmodels 0.13.5 (Apenas para fazer o teste ADF)
* scikit-learn 1.2.2 ou posterior (Apenas para criar o transformer) 

# Examples
  In the 'notebook' folder there are an examples of how to use transformers and other functions in a Jupyter Notebook files.

  Enjoy!

# Directory
<pre>
│  setup.py
└─ fin_ml
     │  notebook
     │    └─ frac_diff_ex.ipynb
     └─ transformers
          │  __init__.py
          └─ frac_diff_transf.py
</pre>

### setup.py
  Instructions to python when installing the package.<br>

### frac_diff_ex.ipynb
  Jupyter notebook with an example using the library frac_diff_transf.

### __init__.py
  File to indicate this folder contains libraries to be added to fin_ml.

### frac_diff_transf.py
  Module that contains the class FracDiff which have the functions to perform the fraction differentiation.

# References
*  https://www.ostirion.net/post/stock-price-fractional-differentiation-best-fraction-finder
*  Chapter 5 from Advances in Financial Machine Learning - Marcos M. Lopez de Prado - 1st Edition

# Author
[Pablo Oliveira - LindedIn](https://br.linkedin.com/in/pablo-oliveira-msc-cqf-88365716)<br>
[Pablo Oliveira - GitHub](https://github.com/pablofrioli)

# License
fin_ml is under [Apache v2 license](LICENSE).
