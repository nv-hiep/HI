# 1. Ideas
- Extract the compact representation features from datasets.
- Then feed the extracted representation features into machine learning models to predict the cold HI gas fraction and HI opacity correction factor.

As far as I'm concerned, the time scale for the project is ~4 months, it would be sufficient to focus on the cold gas fraction (F_CNM).



# 2. Astronomy background
Neutral atomic hydrogen (HI), the most abundant gas in the interstellar medium (ISM), plays a crucial role in the evolution of the ISM in galaxies. It provides the initial material for the formation of stars, serves as the building block of molecular clouds, influences the dynamics of the ISM, acts as a cooling agent and as a source of radiation shielding, and participates in the feedback mechanism that regulates star formation.

The HI gas exists in multiple phases basically distinguished by temperatures: two thermally stable phases the cold neutral medium (CNM) and the warm neutral medium (WNM), with kinetic temperature and density of (Tk, n)=(60–260 K, 7–70 cm−3) and (Tk, n)=(5000–8300 K, 0.2–0.9 cm−3) respectively (Field et al. 1969; McKee & Ostriker 1977; Wolfire et al. 2003) and a thermally unstable phase (UNM) with intermediate temperature and density (Heiles & Troland 2003a; Murray et al. 2018b). The WNM contributes roughly ∼50% of the mass of HI in the ISM, the CNM consists of 30%, and the remaining ∼20% is the contribution of the UNM.

On the journey from the atomic gas to stars, the HI from the warm phase needs to cool down and settles itself into the cold phase. Gravity then causes the cold gas to collapse and form dense regions, known as molecular clouds (T ~ 10 K), where the density becomes high and the temperature is low enough for hydrogen atoms to form molecules. As the density increases further, these molecular clouds collapse and form protostars, which eventually evolve into fully-formed stars. Since the CNM is considered to be a bridge between warm HI gas and molecular clouds, its mass fraction (FCNM) is a key parameter for understanding the transitions from atomic gas to molecular clouds.


In practice, the combination of 21 cm emission and absorption along a line of sight provides the most direct method of estimating the physical properties of the atomic ISM. Such observational pairs allow us to infer optical depths, temperatures (either spin temperatures or upper limits to kinetic temperature), and column densities – the important parameters for observationally distinguishing the different phases of HI gas including CNM, UNM and WNM, as well as constraining their fractions. The column density under the optically-thin assumption (N_HI_thin) is proportional to the brightness HI temperature, hence, can be readily obtained from observed emission profiles. However, this assumption may miss a significant amount of gas mass because the emission includes not only contributions from warm, optically-thin gas, but also from cold, optically-thick gas. In the case when emission/absorption pairs are not available, we alternatively have to apply some kind of opacity correction to the available emission data.

Please note that here (in radio astronomy in general) we consider the WNM as the HI gas with T > 300 K (i.e, WNM includes both UNM and WNM itself) when estimating the total HI column density.


The CNM fraction (F_CNM) and opacity correction factor (R_HI) can only be directly measured when having both emission and absorption information. However, while the HI emission is ubiquitous, observing emission and absorption simultaneously requires a good signal-to-noise for absorption and a limited number of strong radio continuum background sources. This means the HI absorption is too sparse to resolve the spatial distribution of HI phases.

In order to derive the maps of FCNM/R (over large sky areas where spectral emission data are available) without the HI opacity information, we use synthetic emission data produced by (hydrodynamic and magnetohydrodynamic) simulations where the ground-truth values of FCNM/R_HI are available to train machine learning models.

Here we will make use of representation learning for extracting compact and meaningful information from the datasets, then train supervised ML models using the representation features. We will then apply the trained models to the observed HI emission data to estimate the cold HI gas fraction and opacity correction factor along any direction in the sky.


### **In short words:**

HI gas has a wide range of temperatures and densities, meaning that it has several phases distinguished by temperatures (basically).\
WNM : T ~ 8000 K\
UNM : T ~ 500 - 5000 K\
CNM : T ~ 50-250 K


To form stars, HI needs to cool down from WNM -> CNM, where molecular H2 clouds can be formed, then stars can be formed within the H2 clouds (when talking about a cloud, just imagine it's like a giant clump of gas).

The F_CNM is an important parameter, it would tell you about at which conditions (where/when) the CNM HI gas can be converted into a molecular H2 gas cloud.

To accurately measure the column densities (N_CNM, N_WNM, and total N_HI) we need the information on HI emission and HI absorption (I have examples of HI emission spectrum and HI absorption spectrum on Github). Combining the emission-absorption pairs will give you the HI temperature (we use "Spin temperature" in radio astronomy) and HI optical depth (or HI opacity). Then use simple formulae (I showed on Github) to calculate column densities.

The problem is that:
To measure HI absorption, you need a background continuum source behind the HI cloud (I will add an image to illustrate this on Github).

Absorption measurement: Telescope  ------->  [HI cloud]  -----> background continuum source.

But the background continuum sources are NOT always available (they are sparse).

Whereas, the HI emission is always available to observe.

So, if you use the absorption measurements, it will be too sparse and not enough to obtain the spatial distribution of CNM, WNM phases.


Here, with the synthetic data (over large sky areas) from simulations where we know the F_CNM and R_HI, we can train ML models to predict these parameters.



# 4. Why can we recently do that using supervised machine learning techniques?
Up to the present, large, high-resolution observational surveys and new, realistic simulations of the ISM for training are recently available. We can thus compare the observed and simulated data.


# 5. Drawback of CNNs
In my opinion, the drawback of CNNs is that they require large training sets and lack interpretability. The CNNs are somewhat efficient at extracting certain types of information from pixelized data, however, their properties (the trained weights/layers, as well as their relationship with predicted parameters) are not yet fully understood.

Compared to the mathematical tools (e.g, scattering transform - a mathematical alternative of CNNs), neural networks lack transparency, stable mathematical properties or interpretability, which are crucial to scientific research [See here](https://arxiv.org/abs/2112.01288).


# 6. Other thoughts so far
- "deep convolutional neural networks is a solution for the data deficiency as CNN contains a huge number of trainable parameters which can extract complex features"... this is not entirely true as-written, but the insight behind it is correct. We aren't so much solving the data deficiency as side-stepping it, so we can predict parameters we care about without needing so many background sources.

- "CNN structures are shallow and simple ... probability of overfitting." I think the probability of overfitting is more likely to increase with the number of parameters, not decrease. While often deep networks do better, it's not a rule. Maybe we can "explore other architectures" as a means to improving performance, but I suggest dropping the suggestion that a bigger model is a better model.

- With regard to the aim, it seems that there are two key "sub-projects" here:

 - 1. As you have said, "Further exploring method of transforming spectra data into 2D data and compare 1D vs 2D CNN model performance." Learn a representation based on 1D HI spectra, and learn a representation based on 2D representations of that spectrum (e.g. spectrogram or line plot). This is an interesting ML problem.
 - 2. Perform regression (using these features) to determine R_hi and F_cnm. The novel idea here is that we can use representation learning to make use of all the data that are missing background sources for training, not just the sources with labels. This is an interesting astronomy problem.