# 1. Proposed project title.

Representation learning for neutral hydrogen mapping.


# 2. Introduction and Background.

Resolving the phase structure of neutral hydrogen (Hi) gas is crucial for understanding the life cycle of the interstellar medium (ISM). The fraction of cold neutral medium (F_cnm) and opacity correction factor (R_hi) are two important values that decide the phase structure of neutral hydrogen.  
However, accurate measurements of R_hi and F_cnm are limited by the availability of background continuum sources for measuring Hi absorption. In traditional methods, optical depth information needed to be provided in order to get the R_hi and F_cnm. In most of the cases, optical depth information does not exist and the accurate prediction of R_hi and F_cnm is impossible in traditional methods.

# 3. Problem statement.

As accurate measurements of Hi temperature and density are limited by the availability of background continuum sources for measuring Hi absorption. As a result, the prediction of R_hi and F_cnm is impossible using traditional methods.

Due to the property of the spectra, applying deep convolutional neural networks is a solution for predicting the parameters that we care about without needing so much background information.  
Recently, a study has been put forward by Claire E. Murray, et al where they applied a 2-layer convolutional neural network to predict R_hi and F_cnm based on synthetic spectra data [1]. 
To be more specific, they first used synthetic spectra data to train the CNN model and then tested the model on 58 ground truth samples. 
However, there are still some problems in this field that need to be explored and solved.

 - The CNN model in the study is simple. More sophisticated design of CNN may leverage the prediction accuracy of R_hi and F_cnm. 
 - The CNN model in the study tends to over-interpret sharp spectral edges, which results in an under-prediction of f_cnm. This phenomenon can be considered as an overfitting problem and may be caused by the relatively simple CNN structure.
 - the value of R-hi and F_cnm in an area usually has a closed relationship with the surrounding area’s R_hi and F_cnm values. The study calculated the values only based on the spectra of the area, which may cause accuracy reduction. We would like to explore the prediction of F_cnm and R_hi using spectra data and surrounding information together.
 - As CNN usually requires large training set and CNN model in the study is lacked of interpretability, one important aspect of the research is to make the model be interpretable to the astronomer.
In summary, there is a need for a better convolutional neural network architecture and data processing technique for the better prediction of R_hi and F_cnm in improving the prediction accuracy and reducing the overfitting problem.

# 4. Objectives.

The objective of the research is to explore new convolutional neural network architectures, which can achieve higher regression accuracy in the R_hi and F_cnm prediction and solving the problem of overfitting in spectra data with sharp edges. Particularly, the study has the following sub-objectives:
 - To provide a review of R_hi and F_cnm calculation methods.
 - Perform regression (using these features) to determine R_hi and F_cnm with the absence of background information. To be more specific, we would like to explore convolutional neural network architectures or new mechanism for leveraging prediction accuracy and reducing overfitting problems.
 - Exploring 2D representation methods from 1D signals. To be more specific, the signal of spectra is a one-dimensional data. We would like to develop method that can represent 1D data with 2D representation and compare model fitting performance with two different data formats. One example is the spectrogram used in audio or ECG signal transformation.
 - As the noise in the universe is huge, which may affect the values in each channel if the spectra data. Extracting the feature and recognize the noise in an spectra vector is important. One potential solution is using self-supervised learning (comparative learning) method to train CNN and make the output as close as possible based on real vector and vector that contains gaussian noise. 
The result of this study will be valuable to astronomy research as well as related data processing fields in developing better practice and tools for handling 1D data regression problems.



# 5. Methodology.

The primary research method for this study is literature review, theoretical and mathematical analysis and comparative experiments. In the first stage of the research, we will first review different types of research in one dimensional data processing using convolutional neural networks and data processing techniques. As the research [1] contains the drawbacks of over-interpreting sharp spectral edges, which results in an under-prediction of f_cnm. One direction is to design CNN that can focus on border area of the signal so as to leverage the model accuracy in Hi property prediction.
In the second stage of the research, we would like to focus on the 2D representation of 1D signal. Deep dive the representation method of transforming 1D signal into 2D dimensions (e.g. spectrogram or line plot) and put forward new methods of dimensional transformation. Then, we would like to further compare the model performance of 1D and 2D convolutional neural network. The final report will be based on both tasks.


# 6. Raw data and resources.

The research will mainly be based on synthetic spectra data with noise in the spectral (velocity) axis, which was put forward by E. Saury et al in 2014 (E. Saury, et, al 2014).

The training set is a data cube with a size of (101, 512, 512) includes (512x512) spectra and their ground-truth cold atomic hydrogen gas fraction (FCNM) and opacity correction factor (R), where F_cnm range: from ~0.0 to 1 and R_hi range: >= 1 (E. Saury, et, al 2014).  For model testing, we will use 58 HI spectra samples from [1] as our testing set. This 58 samples contain 30 spectra samples from 21-SPONGE (Murray et al. 2015, 2018b) survey and 28 samples from Millennium Survey (Heiles & Troland 2003a,b). 

Each testing sample is a 1 × 414 channels of spectra with corresponding R_hi and F_cnm. In order to prevent the uncertainty of the CNN, ten-fold validation will be used in our experiments. 

Model training will be based on pytorch. Laptop equipped with GTX1080 will be used for CNN trainings. 

| Data Files  | Usage | Data Shape | description | 
| ----------- | ----------- | ----------- | ----------- | 
| Tb_n01_pw02_vs12_512_thick_noise_1.0_K_beam_1.45_pix_.fits    | Training (spectra)       | 512 x 512 x 101 | Synthetic spectra data for training |
| fcnm_map_n01_pw02_vs12_Tcut_500K_512px.fits   | Training (ground truth)        | 512 x 512 | F_cnm ground truth for training |
| R_map_n01_pw02_vs12_Tcut_500K_512px.fits   | Training (ground truth)        | 512 x 512 | R_hi ground truth for training |
| training_data_kok14_galfa.pickle -> X_test | Model Testing | 58 x 414 | spectra data for testing | 
| training_data_kok14_galfa.pickle -> Y_test | Model Testing | 58 x 2 | F_cnm and R_hi ground truth for testing | 

# 7. Timeline.
![TimeLine](docs/timeline.png)

# 8. References.

[1] Claire E. Murray, et, al, 2020, Extracting the cold neutral medium from HI emission with deep learning: Implications for Galactic.\
[2] E. Saury, et, al, 2014, The structure of the thermally bistable and turbulent atomic gas in the local interstellar medium. arXiv:1301.3446v1.\
[3] Heiles, C., & Troland, T. H. 2003a, ApJS, 145, 329—. 2003b, ApJ, 586, 1067.\
[4] Murray, C. E., Stanimirovi ́c, S., Goss, W. M., et al. 2015, ApJ, 804, 89.


