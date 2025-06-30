![image](https://github.com/user-attachments/assets/3c2a5517-c75e-4c5a-9fbf-b6824e1fea21)# DAR-NN-based-FEM-SCA
In this repository, we propose a Domain Adversarial ReFeture Nueral Network (DAR-NN) to facilitate "noisy-clean" adaptation for far field EM traces captured at long distances.

To reconstruct key-dependent features from noisy and distorted far-field EM traces, which are captured over long distances or within complex environments, we propose a new DANN scheme called Domain-Adversarial ReFeture Nueral Network (DAR-NN). By collaboratively utilizing two other deep-learning classifiers as regularization terms of the DAE model, it aims to rebuild key-dependent features for further attacks, thereby achieving a more efficient FEM-SCA. The main idea behind the proposed DAR-NN model is to build the mapping from the "noisy" to "groundtruth" traces while using a CNN classifier pre-trained for the attack and a MLP discriminator as regularization terms. However, it is impractical to capture far-field EM traces truly "clean". The training process of the proposed DAR-NN model is described in Fig. 1.

![image](https://github.com/user-attachments/assets/1d053434-33b5-4d22-b896-c504827446be)

