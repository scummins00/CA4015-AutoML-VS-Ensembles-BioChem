# CA4015-AutoML-VS-Ensembles-BioChem
This repository is to host the Jupyter Book and accompanying files for the second assignment in CA4015, "AutoML vs Ensembles in Biochemistry".
In this assignment, we are tasked with comparing the poerformance of Automatic Machine Learning Frameworks with traditional Machine Learning methods
such as '*Random Forest*' on a common biochemistry task involving data collected by a Gas Chromatograph Mass Spectrometer.

## Repository
The repository for this assignment can be found [here](https://github.com/scummins00/CA4015-AutoML-VS-Ensembles-BioChem).

## Jupyter Book
This repository uses Jupyter Book to host its contents online. Find the online book [here](https://scummins00.github.io/CA4015-AutoML-VS-Ensembles-BioChem/).
If you're having trouble using Jupyter Book please view the section on [Building the Jupyter Book](#building-the-jupyter-book).

### Building the Jupyter Book
1. Make changes to the contents in the `main` branch.
2. Rebuild the book with `jupyter-book build book/`
3. Push your new content to the hosting repository.
4. Use `ghp-import -n -p -f book/_build/html` to push new content to the *gh-pages* branch.

## Building a PDF
This repository supports the creation of a latex pdf file (*.tex file*). This assignment required the submission of a .pdf file rather than a Jupyter Book.
Please find [the final submitted .pdf](https://github.com/scummins00/CA4015-AutoML-VS-Ensembles-BioChem/blob/main/ca4015_assignment_2.pdf) file above.