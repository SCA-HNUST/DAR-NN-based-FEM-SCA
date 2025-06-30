# DAR-NN-based-FEM-SCA

In this repository, we propose a Domain Adversarial ReFeture Nueral Network (DAR-NN) to facilitate "noisy-clean" adaptation for far field EM traces captured at long distances.

By collaboratively utilizing two other deep-learning classifiers as regularization terms of the DAE model, the DAR-NN model aims to rebuild key-dependent features for further attacks, thereby achieving a more efficient FEM-SCA. 

The main idea behind the proposed DAR-NN model is to build the mapping from the "noisy" to "groundtruth" traces while using a CNN classifier pre-trained for the attack and a MLP discriminator as regularization terms. However, it is impractical to capture far-field EM traces truly "clean". The training process of the proposed DAR-NN model is described in Fig. 1.

![image](https://github.com/user-attachments/assets/1d053434-33b5-4d22-b896-c504827446be)

At the attack stage, the trained reconstructor in DAR-NN is used independently as the filter to map the captured ”victim” traces to ”reconstructed” traces. Next, the newly generated ”reconstructed” traces are used to derive the secret key of the victim device by using the adapted classifier in DAR-NN (see Fig. 3).

![image](https://github.com/user-attachments/assets/ce7f54bc-953c-4050-9ca5-b19e8bfdcd26)

## Database

The first dataset is a far-field EM side-channel dataset, in which the traces are captured at 15 m distance in an office corridor environment from 10 nRF52 DK implementation of TinyAES.
