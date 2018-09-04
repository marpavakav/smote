# smote
This repository includes a SMOTE function with additional functionality and one python notebook example

Background: SMOTE (Synthetic Minority Over-sampling Technique) is a method explained in [Chawla et al.](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html) in which points of a minority class get generated in a random distance between each minority point and a selected amount of its closest neighbour points of the same minority class.

We use the term *smote_ratio* to refer to the size of the generated data compared to the origial minority data. e.g. for smote_ratio=2 the generated data would be 2 x times the size of the original minority class data.

### mysmote.py
The function mysmote.py replicates the SMOTE method as described in Chawla et al. with our additional functionality of using the nearest neihgbours multiple times when the smote_ratio is larger than the amount of nearest neighbours selected.

### smote_application.ipynb
The notebook smote_application.ipynb implements mysmote.py on a specific example. It shows that using  using an SVM classification model on the original test file where the two classes are imbalanced, all minority class points get misclassified. When the smote method is implemented, with a high enough smote_ratio, the minority points classification gets improved.

### yeast6_data.dat
File yeast6_data.dat is used in the notebook and has been taken from the [Keel dataset](http://sci2s.ugr.es/keel/imbalanced.php)
reference: J. Alcalá-Fdez, A. Fernandez, J. Luengo, J. Derrac, S. García, L. Sánchez, F. Herrera. KEEL Data-Mining Software Tool: Data Set Repository, Integration of Algorithms and Experimental Analysis Framework. Journal of Multiple-Valued Logic and Soft Computing 17:2-3 (2011) 255-287
