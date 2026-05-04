#AI-Based Surrogate Model for LEO Collision Risk Prediction
Overview

This repository contains the code used to develop the surrogate model for predicting relative collision risk in Low Earth Orbit (LEO). The model is based on an XGBoost regression framework trained on binned orbital data derived from a LEO object catalog.

The surrogate model is designed to approximate a relative collision risk metric, defined as the product of a collision probability proxy and combined object mass. This approach enables fast prediction and supports subsequent analysis, including explainability and optimization (not included in this repository).

This code corresponds to the implementation used in the study:

AI-assisted framework for collision risk prediction and optimization in Low Earth Orbit
