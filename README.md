# Explorable CT-scans Reconstruction using Dictionary Learning
This repository is the implementation of the Bachelor Thesis of "Explorable CT-scans Reconstruction using Dictionary Learning"(Sep. 2023).

# Abstract (English):

The most significant challenge in the field of medicine is providing accurate images of human inner organs to aid in the diagnosis of a diverse range of diseases and health conditions, and potentially to assist in the development of more effective therapy strategies for patients. Despite the fact that Computed Tomography is one of the most well-known tools for recognizing and analyzing a range of medical illnesses and disorders, there are reasons that prevent Computed Tomography from being extremely accurate. The reason lies in the health risks involved in the dangers posed by the use of ionizing X-rays.

**Sparse-view CT** is supposed to deal with these risks, targeting reduced scan times by capturing fewer X-ray projections, which corresponds to a lower set of projection angles and consequent costs. The objective of this thesis is dedicated to improving the results of the paper **"Explorable Data Consistent CT Reconstruction"** Droege et al. [2022](https://bmvc2022.mpi-inf.mpg.de/0746.pdf), in which the technique is demonstrated in the context of detecting malignant lung nodules in chest CT scans. This employs a pre-trained malignancy classifier to generate a range of potential reconstructions, each corresponding to different levels of malignancy and this is in contrast to a single image aligned with an arbitrary medical interpretation.

Among the results, there are a number of artifacts that lead to CT scans that differ from reality. The main goal of this thesis is to improve these images and make them more realistic by applying Dictionary Learning. For this purpose, two different and fast algorithms from the context of sparse coding or more specifically dictionary learning are introduced to achieve an improvement of the final result.

After explaining the basic concepts relevant to the topic, this work focuses on the introduction and implementation of **Bi-level optimization** into dictionary learning and the **KSVD model and Orthogonal Matching Pursuit**. These algorithms are tightly integrated with the primary model so that the images are effectively denoised during reconstruction. This integration results in CT scans that are more realistic and authentic. The algorithms were applied to different samples, and the results of these evaluations showed that the algorithms largely achieved the intended goals, with particular reference to the samples where the main application was not able to provide realistic images in the part of the scans where the nodules are located or have an unrealistic appearance.

# Abstract (German):

Die größte Herausforderung im Bereich der Medizin besteht darin, genaue Bilder von den inneren Organen des Menschen zu erstellen, um die Diagnose einer Vielzahl von Erkrankungen und Gesundheitszuständen zu unterstützen und möglicherweise effizientere Behandlungsmethoden für die Betroffenen zu entwickeln. Trotz der Tatsache, dass die Computertomografie eines der bekanntesten Tools zur Erkennung und Analyse einer Reihe von medizinischen Krankheiten und Beschwerden ist, gibt es Ursachen, die eine hohe Genauigkeit der Computertomografie verhindern. Der Grund liegt in den gesundheitlichen Risiken, die mit der Anwendung von ionisierenden Röntgenstrahlen in Zusammenhang stehen.

**"Sparse-view CT"** soll diesen Risiken mit dem Ziel entgegenwirken, die Dauer des Scans zu minimieren, indem weniger Röntgenprojektionen aufgenommen werden, was einer geringeren Anzahl von Projektionswinkeln und damit geringeren Kosten gleichkommt. Das Hauptziel dieser Arbeit ist es, die Ergebnisse der Arbeit **"Explorable Data Consistent CT Reconstruction"** von Droege et al. [2022](https://bmvc2022.mpi-inf.mpg.de/0746.pdf) zu verbessern, in der die Technik im Zusammenhang mit der Erkennung bösartiger Lungenknötchen in Thorax-CT-Scans demonstriert wird. Dabei wird ein vortrainierter Malignitätsklassifikator eingesetzt, um eine Reihe von potenziellen Rekonstruktionen zu gewinnen, die jeweils unterschiedlichen Malignitätsgraden entsprechen, im Gegensatz zu einem einzigen Bild, das mit einer beliebigen medizinischen Interpretation ausgerichtet ist.


Die Ergebnisse zeigen eine Reihe von Artefakten, die zu CT-Scans führen, die von der Realität deutlich abweichen. Das wesentliche Ziel dieser Arbeit ist es, diese Bilder zu verbessern und sie durch die Anwendung von Dictionary Learning realistischer zu machen. Zu diesem Zweck werden zwei verschiedene und schnelle Algorithmen aus dem Kontext der Sparse Coding oder genauer gesagt Dictionary Learning eingeführt.

Nach der Erläuterung der grundlegenden Konzepte, die für das Thema relevant sind, befasst sich diese Arbeit mit der Einführung und Umsetzung der **Bi-Level-Optimierung** in das Dictionary-Learning sowie mit **KSVD-Model und orthogonal Matching Pursuit**. Die Algorithmen wurden auf verschiedene Beispiele angewandt, und die Ergebnisse dieser Auswertungen ergaben, dass die Algorithmen die angestrebten Ziele weitgehend erreichten, insbesondere bei den Beispielen, bei denen die Hauptanwendung nicht in der Lage war, realistische Bilder in dem Teil der Scans zu liefern.


## Get Started

- **Dependencies** 
  - install PyTorch (e.g. version 1.12.1+cu116) and Torchvision (e.g. version0.13.1+cu116)
  - install NumPy
  - install SciPy 
  - pip install -U scikit-learn
  - install requirements by: `pip install -r requirements`
  - install https://github.com/drgHannah/Radon-Transformation
- **Data** 
  - please download the data from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- **Training and Evaluation** 
	-   for some tests on pre-trained networks, save [this network](https://drive.google.com/drive/folders/16pwCuat4tby_O3k2q2JDf79aYd6-cTGb?usp=sharing)  as *network_100a.pt* into the project folder
  - for the optimization, please have a look at `explarable_reconstruction_Bi_level.ipynb` to use the Bi-level optimization solution or `explorable_reconstruction_KSVD.ipynb` to use the KSVD algorithm solution.


## Specifications used for the thesis
- GeForce GTX 1080 Ti
- CUDA Version: 11.6
