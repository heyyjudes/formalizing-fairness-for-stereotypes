## Fairness with respect to Stereotype Predictors: Impossibilities and Best Practices

This repository contains the official code and experiments for our TMLR Paper [Fairness with respect to Stereotype Predictors: Impossibilities and Best Practices](https://openreview.net/forum?id=FPJKZDzdsW&). 


**Abstract**: As AI systems increasingly influence decision-making from consumer recommendations to educational opportunities, their accountability becomes paramount. This need for oversight has driven extensive research into algorithmic fairness, a body of work that has examined both allocative and representational harms. However, numerous works examining representational harms such as stereotypes encompass many different concepts measured by different criteria, yielding many, potentially conflicting, characterizations of harm. The abundance of measurement approaches makes the mitigation of stereotypes in downstream machine learning models highly challenging. Our work introduces and unifies a broad class of auditors through the framework of \textit{stereotype predictors}. We map notions of fairness with respect to these predictors to existing notions of group fairness. We give guidance, with theoretical foundations, for selecting one or a set of stereotype predictors and provide algorithms for achieving fairness with respect to stereotype predictors under various fairness notions. We demonstrate the effectiveness of our algorithms with different stereotype predictors in two empirical case studies.

---

## Recreating Experiments
- [`synthetic.ipynb`](synthetic.ipynb): Run synthetic experiments to evaluate stereotype bias and mitigation strategies in controlled settings.

### Datasets
We use two main datasets in our paper: 
- `BiasBios`: We use code form the original [BiasBios](https://github.com/microsoft/biosbias) to scrape ~47k progessional biographies. 
- `NELS`: [National Education Longitudinal Study](https://nces.ed.gov/surveys/nels88/) is a national study of 8th graders with follow up surveys until they reached high school. We include the processed data we used for our experiments in the data folder. 

### Code

We provide two main Jupyter notebooks to reproduce all results from the paper:


- [`Bios.ipynb`](Bios.ipynb): Reproduce our experiments on the BiosBias dataset three different race-based stereotypes. 
- [`NELS.ipynb`](NELS.ipynb): Reproduce our experiments on the National Education Longtitudinal Survey dataset with three different gender based stereotypes. 

These notebooks use our code in `models.py` which implement multicalibration algorithms and `data.py` which handles the data processing. 

### Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

### Setup

1. Clone this repository:
    ```
    git clone https://github.com/heyyjudes/formalizing-fairness-for-stereotypes.git
    cd formalizing-fairness-for-stereotypes
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the notebooks:
    - Open either `synthetic.ipynb`, `Bios.ipynb` or `NELS.ipynb` in Jupyter and follow the instructions to reproduce the results.

---

