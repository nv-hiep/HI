# Atomic Hydrogen Gas (HI)

### Estimate cold HI gas fraction ( $F_{CNM}$) and opacity correction factor ( $R_{HI}$) using supervised machine learning techniques.

$HI$: Atomic hydrogen gas\
$H_2$: Molecular hydrogen gas\
Velocity <-> Wavelength/Frequency: In radio astronomy, wavelength[m]/frequency[Hz] is often converted into velocity[VLSR: km/s or m/s].

## 1. What is $HI$ gas?
$HI$ gas is the atomic hydrogen gas, the most abundant element in the interstellar medium (ISM), making up about 75% of the total gas mass. This gas is responsible for emitting the 21-cm line radiation, which is used to study the distribution and kinematics of the interstellar medium. The hydrogen gas is the raw fuel for stars to form. On the journey from the warm atomic phase (WNM: T > 500 K) to stars, the HI gas needs to cool down and settles itself into a cold phase (CNM: T ~ 50-100 K). Gravity then causes the cold gas to collapse and form dense regions, known as molecular clouds ( $H_2$, T ~ 10 K), where the density becomes high and temperature is low enough for hydrogen atoms to form molecules. As the density increases further, these molecular clouds collapse and form protostars, which eventually evolve into fully-formed stars.

## 2. The 21-cm HI emission line and observation
The atomic gas in the ISM has traditionally been traced by the 21-cm hyperfine structure line of neutral hydrogen (with a rest frequency of 1420.40575 MHz). The line arises from the transition between the two hyperfine levels of the HI ground state, which are slightly split by the interaction between the spins of the electron and proton. The state with parallel spins has higher energy. The 21 cm line of atomic hydrogen was detected for the first time in emission by Ewen & Purcell (1951) and Muller & Oort (1951). The observation of this hyperfine transition can be marked as the dawn of spectral-line radio astronomy.

<img width="501" alt="" src="https://user-images.githubusercontent.com/13595525/213603417-c079548a-b89c-4235-b2b3-029f4cc1be01.png">

Since the 21-cm HI line radiation locates in the radio range of the electromagnetic spectrum, it can penetrate through dust clouds and give us a relatively complete map of the atomic hydrogen in the ISM. Observationally, the HI line is detected primarily in emission, but can also be observed in absorption against a sufficiently bright radio continuum sources (e.g. H II regions, radio galaxies, quasars or active galactic nuclei), or in self-absorption when the arrangement of gas along the line of sight permits it. While 21 cm emission is ubiquitous along any direction in the sky, 21 cm absorption is not.

Full-sky map at 21-cm radio emission line (HI4PI collaboration), center of the Milky Way is in the direction towards the image's center.
<img width="800" alt="" src="https://user-images.githubusercontent.com/13595525/213612072-c046a67a-e5ce-4996-8a49-c8fbfca2d6e6.jpeg">

An example of emission profiles:\
<img width="580" alt="" src="https://user-images.githubusercontent.com/13595525/213611479-f492dfe0-0a50-4c15-8745-646b988e6abc.png">


An example of absorption profile:\
![abs_spec_2](https://user-images.githubusercontent.com/13595525/213610230-bd79bb5d-d236-4b4c-b774-55138242886c.png)

## 3. Properties of HI gas clouds from emission and absorption measurements
Combined emission and absorption measurements provide a direct way to estimate the $HI$ optical depth ( $\tau$) and $HI$ spin temperature ( $T_s$), the two crucial parameters for determining the column densities of HI gas ( $N_{HI}$) in both warm and cold phases ( $N_{CNM}, N_{WNM}$, total $N_{HI} = N_{CNM} + N_{WNM}$), and thus the CNM fraction: $F_{CNM} = N_{CNM}/N_{HI}$). The column density under the optically-thin assumption ( $N^{thin}_{HI}$) is proportional to the brightness HI temperature, hence, can be readily obtained from emission profiles. However, this assumption may miss a significant amount of gas mass because the emission includes not only contributions from warm, optically-thin gas, but also from cold, optically-thick gas. In the case when emission/absorption pairs are not available, we alternatively have to apply some kind of opacity correction to the available emission data.

Please note here that when calculating column density we consider the WNM as the HI gas with T > 300 K (i.e, WNM includes both UNM and WNM itself) when estimating the total HI column density.

On-/off-source measurements:\
On-source measurement gives absorption Tb_on.\
Off-source measurement gives emission Tb_off.\
![on_off_src_measurements](https://user-images.githubusercontent.com/13595525/214567508-61d8036d-4010-4d4f-b685-0fcefb67c9bd.png)


An example of combined emission and absorption spectra:
<img width="865" alt="" src="https://user-images.githubusercontent.com/13595525/213611901-b662981b-0d43-4604-a27a-1746fe542823.png">

Optically-thin HI column density along a direction (or line-of-sight):
$$\frac{N_{HI-thin}}{[\mathrm{cm^{-2}}]} = 1.8224\times 10^{18}\ \int \frac{T_\mathrm{B}}{\mathrm{[K]}}\ \frac{d\mathrm{v}}{\mathrm{[km\ s^{-1}]}}$$
where $T_B (v)$ : emission profile observed by a radio telescope.


Total HI column density along a direction (or line-of-sight): $$\frac{N_\mathrm{HI}}{[\mathrm{cm^{-2}}]} = 1.8224\times 10^{18}\ \frac{T_\mathrm{s}}{\mathrm{[K]}} \int\tau_\mathrm{v}\ \frac{d\mathrm{v}}{\mathrm{[km\ s^{-1}]}}$$
where $\tau (v)$ : absoprtion profile observed by a radio telescope when poingting the telecope towards a background continuum source behind the HI gas cloud.\
$T_s$ : temperature of HI gas cloud (referred to as the excitation temperature in general, and spin temperature in radio astronomy), it is often derived from the combination of emission profile and absorption profile.


## 4. Cold HI gas and opacity correction factor 
The cold HI gas fraction (FCNM) is a crutial parameter for understanding the transitions of atomic hydrogen gas from warm (WNM) to cold (CNM) phases and then to molecular clouds (H2). The cold HI gas fraction is closely related to the process of star formation. The CNM is more likely to form H2 molecules, which are the building blocks of molecular clouds, and the primary ingredient for star formation. Cold HI gas is denser and more gravitationally unstable than warm HI gas, making it more likely to collapse and form dense regions where new stars can form. Additionally, the CNM is thought to be confined to thin "sheets" or "filaments" within the WNM, which can affect the overall distribution and kinematics of the interstellar medium. Understanding the cold HI gas fraction can also provide insights into the overall dynamics, temperature and density structure of the interstellar medium.

The opacity correction factor ( fraction $R_{HI} = N^{HI} / N^{thin}_{HI}$) indicates how much the optically-thin assumption underestimates the true column density.

The CNM fraction (F_CNM) and opacity correction factor (R_HI) can only be directly measured when having both emission and absorption information. However, while the HI emission is ubiquitous, observing emission and absorption simultaneously requires a good signal-to-noise for absorption and a limited number of strong radio continuum background sources. This implies that the HI absorption is too sparse to resolve the spatial distribution of the HI phases.

In other words, we DO have full-sky maps of HI emission, but we DO NOT have absorption measurements at any directions (or any pixels) in the sky, we thus do not have a full map of directly-measured FCNM/R_HI.

## 5. Use machine learning to estimate FCNM/R
In order to derive the maps of FCNM/R (over large sky areas where spectral emission data are available) without the HI opacity information, we use synthetic emission data produced by (hydrodynamic and magnetohydrodynamic) simulations where the ground-truth values of FCNM/R_HI are available to train machine learning models.

Here we will make use of representation learning for extracting compact and meaningful information from the datasets, then train supervised ML models using the representation features. We will then apply the trained models to the observed HI emission data to estimate the cold HI gas fraction and opacity correction factor along any direction in the sky.
