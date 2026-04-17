# CMPE 401 Project 2: Time Series Modeling - LSTM_Forecasting
**Author:** Matthew Ofina

## 1. Project Overview
This project explores two fundamental deep learning approaches for time-series modeling using Keras: 
* **Classification:** Utilizing a Transformer architecture on the FordA dataset.
* **Forecasting:** Utilizing a Long Short-Term Memory (LSTM) network on the Jena Climate dataset.

The objective is to reproduce baseline model performances and engineer structural modifications to improve the forecasting capabilities of the LSTM architecture against short-term weather volatility.

---

## 2. Usage Instructions & Reproducibility
The source code for both models is contained within Jupyter Notebooks (`.ipynb`). To reproduce the training process and experimental results:

**Environment Setup:**
1. Clone this repository to your local machine:
   `https://github.com/Matt0fina/LSTM_Forecasting.git`
2. Open the notebooks using Google Colab, Kaggle, or a local Jupyter environment configured with TensorFlow/Keras.
3. *Note for Local Execution:* GPU acceleration is highly recommended for the LSTM modifications. Ensure CUDA and cuDNN are properly configured if running locally.

**Execution:**
* Run all cells in `transformer_classification.ipynb` to execute the baseline FordA classification.
* Run all cells in `lstm_forecasting.ipynb` to execute the baseline Jena Climate forecasting.

---

## 3. Key Findings: Baseline Evaluation

**Model 1: Transformer Classification**
* **Result:** The baseline model exhibited severe sensitivity to random weight initialization, failing to converge and flatlining at a loss of ~0.693 (equivalent to random guessing). First time running the baseline model resulted in the training stopping at epoch 25. The model was reset and ran again but with the training being stopped at epoch 11. The results were as follows:

| Architecture | Loss | Sparse Categorical Accuracy | Val Loss | Val Sparse Categorical Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.6927 | 0.5149 | 0.6934 | 0.5035 |

**Model 2: LSTM Forecasting**
* **Result:** The baseline model (120-step sequence length, 32 hidden units) converged stably with a validation loss of `~0.1250`. Visual analysis of single-step predictions confirmed the model could work out trends but exhibited high error variance during spikes and large changes in data.

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/15812250-d533-4e4c-a134-be3bbedcce3a" /> 
<img width="556" height="455" alt="image" src="https://github.com/user-attachments/assets/6512a892-cdc4-4f16-a2aa-024650c7ca4b" /><img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/04e30b9d-0d64-419d-a182-6c7239148fae" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/4e60d35e-cb78-4a5f-8f53-d53743804e7e" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/d71a9edd-8c18-4996-96c6-466400f125ed" />
<img width="547" height="455" alt="image" src="https://github.com/user-attachments/assets/3cc933b4-5cbd-42a4-9a88-3515201f4aaa" />

---

## 4. Key Findings: Experimental Benchmark (LSTM)
To improve upon the baseline forecasting model, three targeted architectural modifications were engineered and evaluated. 

### Performance Summary
| Architecture | Sequence Length | Hidden Units | Layers | Final Val Loss | Result Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 120 | 32 | 1 | `~0.1250` | Handled trends well but struggled with spikes and large variance. |
| **Modification 1** | 240  | 32 | 1 | `~0.1147` | **Best Performance.** Better context reduced error. |
| **Modification 2** | 240  | 64 | 1 | `~0.1270` | Capacity may be too high and began memorizing noise. |
| **Modification 3** | 240  | 64 | 2 | `~0.1277` | Overfitting and training loss dropped while validation stalled. |

The visual results of the modifications can be seen below:

**Modification 1: Increase Sequence Length**

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/95779d74-514a-47f5-93b9-f4f2d904839a" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/0d9f2411-57ea-4312-8da8-63edbe4f87fe" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/e65c9637-bfd9-4439-949b-6c00b51a30a0" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/3eb7ec24-0756-4564-a4ab-8c80a6517bae" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/dd502c1f-5d2c-4f5f-be01-3106751ffe32" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/2bb03f5d-032b-4877-9f27-9af6a5177328" />


**Modification 2: Increase hidden LSTM layer inputs**

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/a7c91ab5-9658-4e9e-be7b-e9fe0780354d" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/d00ba5d3-8a44-4d74-b691-153f97b56bba" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/f479931a-0f16-4f03-86cc-2016108346b9" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/3ad9df5b-0259-47d0-929a-7e97334363fc" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/f1f0708d-3825-47c9-b6ec-1df844192a6e" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/038a69b9-d2e6-4303-8769-e527d7eabafe" />


**Modification 3: Add another hidden layer**

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/25049384-840d-44c0-8c86-254aa51102a8" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/1ddd4e52-9516-4552-bf6b-a41ebd2da22b" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/2246d67e-b525-4c66-84b9-8d12ac09b750" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/68a3e59c-3b46-437b-931e-3094aca7a40f" /><img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/caf42a0f-edc4-4a99-9a50-5db1e9dfe73e" />
<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/8da8ae76-50dc-456d-a08f-e954cab7a0bf" />

As can be seen, the vaidation loss increases when increasing the parameters with the hidden layers but the Single Step Prediction plots have accurate prediction, indicating overfitting.

### Engineering Analysis
* **Context over Capacity:** Expanding the historical observation window from -120 days to -240 days provided the most significant performance gain. The added temporal context prevented the LSTM from overreacting to immediate, short-term temperature spikes.
* **Overfitting via Over-parameterization:** Expanding the hidden state to 64 units and stacking a second LSTM layer ballooned the parameter count to 51,521. This excess capacity resulted in distinct overfitting. The model memorized the training data's specific noise rather than generalizing, leading to degraded prediction accuracy on unseen data.

---

## 5. Project Reflection

**Which model did you find easier to understand and why?**
The LSTM forecasting model was easier to understand for me. The initial visual intuition after running the baseline model was easy to understand as it was just a line graph. I could see if the model made a mistake with the prediction relative to the actual future and "see" how large the mistake was. Being able to visualize how well the model performed makes it very easy to understand. It was also easy to understand how the prediction works by feeding data of the past and seeing if the model could predict the output. A direct input to output mapping.

**What improvement did you try, and what did you learn from it?**
I learned that simply adding more layers and parameters does not inherently improve time-series forecasting. The highest-capacity model exhibited clear signs of overfitting, proving that our simpler architecture (Sequence Length 240, Hidden Size 32) found a much better balance between capacity and generalization. Overfitting is unwanted as the model just starts "memorizing" the answers instead of actually learning. If it is fed new, unseen data, the model would perform poorly and have a large error in predicting the future.
