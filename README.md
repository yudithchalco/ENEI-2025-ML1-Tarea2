## Assignment: Logistic Regression and Multiclass Extensions

**Deadline:** Sunday, October 12th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

### Part A. Binary Logistic Regression from Scratch

1. **Dataset**
   Use the **Heart Disease dataset** from the UCI repository (available via `sklearn.datasets.fetch_openml("heart-disease-uci", as_frame=True)`).

   * Task: predict whether a patient has heart disease (`target` column).
   * Standardize numeric features, one-hot encode categorical ones.
   * Split into 70% train / 30% test.

2. **Model Derivation and Implementation**

   * Recall:
     [
     p(y=1|x) = \sigma(x^\top \beta) = \frac{1}{1 + e^{-x^\top \beta}}
     ]
     and
     [
     \ell(\beta) = \sum_i [y_i\log p_i + (1 - y_i)\log(1 - p_i)]
     ]
   * Derive its gradient:
     [
     \nabla_\beta \ell = ???
     ]
   * Implement gradient descent to maximize the log-likelihood (or equivalently, minimize the negative log-likelihood).
   * Show convergence plots for at least two learning rates.

3. **Evaluation**

   * Compute accuracy, precision, recall, F1 score in the test set.
   * Compare with `sklearn.linear_model.LogisticRegression`.

---

### Part B. Multiclass Logistic Regression via One-vs-All (OvA)

4. **Dataset**
   Use the **Wine dataset** (`from sklearn.datasets import load_wine`).

   * There are 3 wine cultivars (classes) with 13 chemical features.
   * Standardize all features.

5. **OvA Implementation**

   * Build **three binary classifiers**, each distinguishing one class vs. all others.
   * Use your binary logistic regression optimizer from Part A.
   * For prediction:

     * Compute probabilities from each classifier.
     * Assign each observation to the class with the highest predicted probability.
   * Report confusion matrix and accuracy.

6. **Comparison**

   * Fit `LogisticRegression(multi_class="ovr")` from sklearn.
   * Compare coefficients and accuracy to your own implementation.

---

### Part C. Multinomial (Softmax) Logistic Regression from Scratch

7. **Theory**

   * For (K) classes:
     [
     p(y=k|x) = \frac{e^{x^\top \beta_k}}{\sum_{j=1}^{K} e^{x^\top \beta_j}}
     ]
     and
     [
     \ell(\beta) = \sum_i \sum_k \mathbf{1}(y_i=k) \log p(y_i=k)
     ]
   * Derive the gradient:
     [
     \frac{\partial \ell}{\partial \beta_k} = ???
     ]

8. **Implementation**

   * Implement gradient descent updating all class weight vectors simultaneously.
   * Include a `softmax` function with numerical stability (`z -= np.max(z, axis=1, keepdims=True)` before exponentiation).
   * Monitor log-likelihood convergence.

9. **Evaluation**

   * Use the same Wine dataset.
   * Compute accuracy, per-class precision/recall, and confusion matrix.
   * Compare to `LogisticRegression(multi_class="multinomial", solver="lbfgs")`.

---

### Deliverables

You must fork the [original repository](https://github.com/RodrigoGrijalba/ENEI-2025-ML1-Tarea2), and turn in a link to your group's repository with:

* A Jupyter notebook (in the `src` folder) with:

  * Binary logistic regression (from scratch and sklearn)
  * OvA and multinomial implementations
  * Gradient derivations
  * Convergence and comparison plots
* A short (~600 words) write-up explaining:

  * How the gradient differs between binary, OvA, and multinomial forms
  * How numerical stability issues may arise in softmax
  * When OvA and multinomial approaches diverge in predictions

