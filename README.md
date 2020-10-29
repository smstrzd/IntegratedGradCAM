# Integrated Grad-CAM (submitted to ICASSP2021)

## Integrated Grad-CAM: Sensitivity-Aware Visual Explanation of Deep Convolutional Networks via Integrated Gradient-Based Scoring 

**Abstract**

Visualizing the features captured by Convolutional Neural Networks (CNNs) is one of the conventional approaches to interpret the predictions made by these models in numerous image recognition applications. Grad-CAM is a popular solution that provides such a visualization by combining the activation maps obtained from the model. However, the average gradient-based terms deployed in this method underestimates the contribution of the representations discovered by the model to its predictions. Addressing this problem, we introduce a solution to tackle this issue by computing the path integral of the gradient-based terms in Grad-CAM. We conduct a thorough analysis to demonstrate the improvement achieved by our method in measuring the importance of the extracted representations for the CNN's predictions, which yields to our method's administration in object localization and model interpretation.

**Architecture**

<p align="center">
    <img src="docs/IGCAM-arch.svg" alt="Architecture of Integrated Grad-CAM" />
    <br>
    <em>Fig: Schematic of the proposed method considering that the baseline image is set to black and the path connecting the baseline and the input is set as a straight line.</em>
</p>

---

Open the [demo notebook](Integrated_Grad_CAM.ipynb) in Google Colab to quickly replicate our results.
