# HI emission

$HI$: Atomic hydrogen gas\
$H_2$: Molecular hydrogen gas

*Estimate the cold HI gas fraction ( $F_{CNM}$) and opacity correction factor ( $R_{HI}$) using supervised machine learning techniques.*

Atomic hydrogen gas (HI) is the most abundant element in the interstellar medium (ISM), making up about 75% of the total gas mass. This gas is responsible for emitting the 21-cm line radiation, which is used to study the distribution and kinematics of the interstellar medium. The hydrogen gas is the raw fuel for stars to form. On the journey from the warm atomic phase (WNM: T > 300 K) to stars, the HI gas needs to cool down and settles itself into a cold phase (CNM: T ~ 50-100 K). Gravity then causes the cold gas to collapse and form dense regions, known as molecular clouds ( $H_2$, T ~ 10 K), where the density becomes high and temperature is low enough for hydrogen atoms to form molecules. As the density increases further, these molecular clouds collapse and form protostars, which eventually evolve into fully-formed stars.

Since the 21-cm HI line radiation (1.420 GHz) locates in the radio range of the electromagnetic spectrum, it can penetrate through dust clouds and give us a relatively complete map of the atomic hydrogen in the ISM. Observationally, the HI line is detected primarily in emission, but can also be observed in absorption against a sufficiently bright radio continuum sources (e.g. H II regions, radio galaxies, quasars or active galactic nuclei), or in self-absorption when the arrangement of gas along the line of sight permits it. While 21 cm emission is thus ubiquitous along any direction in the sky, 21 cm absorption is not.


Combined emission and absorption measurements provide a direct way to estimate the optical depth and spin temperature, the two crucial parameters for determining the column densities of HI gas ( $N_{HI}$) in both warm and cold phases ( $N_{CNM}, N_{WNM}$, total $N_{HI} = N_{CNM} + N_{WNM}$), and thus the CNM fraction: $F_{CNM} = N_{CNM}/N_{HI}$). The column density under the optically-thin assumption ( $N^{thin}_{HI}$) is proportional to the brightness HI
temperature, hence, can be readily obtained from emission profiles. However, this assumption may miss a significant amount of gas mass because the emission includes not only contributions from warm, optically-thin gas, but also from cold, optically-thick gas.

The cold HI gas fraction (FCNM) is a crutial parameter to understand the transitions of atomic hydrogen gas from warm (WNM) to cold (CNM) phases and then to molecular clouds (H2). The opacity correction factor ( fraction $R_{HI} = N^{HI} / N^{thin}_{HI}$) indicates how much the optically thin
assumption underestimates the true column density.
