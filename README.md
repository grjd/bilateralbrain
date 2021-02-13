# Code and Dataset supplement "The brain is globally symmetric: an analysis of intra and interhemispheric symmetry of the limbic system in the aging human brain"

**Citation**

Jaime Gómez-Ramírez et al. "The brain is globally symmetric: an analysis of intra and interhemispheric symmetry of the limbic system in the aging human brain", 2021 (pre-print on BioRxiv: https://doi.org/)

**Abstract**
Here we address the hemispheric interdependency of subcortical structures in the aging human brain, in particular, we investigate whether volume variation can be explained with the adjacency of structures in the same hemisphere or is due to interhemispheric development of mirror subcortical structures in the brain.
Seven subcortical structures in both hemispheres were automatically segmented in a large sample of over three thousand magnetic resonance imaging (MRI) studies. We perform Eigenvalue analysis to find that anatomic volumes in the limbic system and basal ganglia show similar statistical dependency when considered in the same hemisphere (intrahemispheric) or different hemispheres (interhemispheric).
Our results indicate that anatomic bilaterality is preserved in the aging human brain, supporting recent findings that postulate increased communication between distant brain areas as a mechanism to compensate for the deleterious effects of aging. 

**Dataset description**

The dataset contains a csv file, it can be opened as a Pandas dataframe containing the results of the automated segmentation performed with FreeSurfer. The dataset includes the columns:

**MRI Data collection**

The imaging data were acquired in the sagittal plane on a 3T General Electric scanner (GE Milwaukee, WI) utilizing T1-weighted inversion recovery, supine position, flip angle $12\circ$, 3-D pulse sequence: echo time \textit{Min. full}, time inversion 600 ms., Receiver Bandwidth $19.23$ kHz, field of view $= 24.0$ cm, slice thickness $1$ mm and Freq $\times$ Phase $288 \times 288$. The brain volume loss at the moment of having the MRI compared to the maximum brain volume is computed as the Brain Segmentation Volume to estimated Total Intracranial Volume (eTIV) \cite{eTIV} ratio (ICV and eTIV the FreeSurfer term for intracranial volume are used equivalently). The postprocessing was performed with FreeSurfer \cite{fischl2012freesurfer}, version freesurfer-darwin-OSX-ElCapitan-dev-20190328-6241d26 running under a Mac OS X, product version 10.14.5. 

_[FreeSurfer, 2017] FreeSurfer cortical reconstruction and parcellation process. (2017).Anatomical processing script:recon-all.
https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all._
