# GreenVine: Early Plant Disease Detection with Machine Learning, Deep Learning, and Transfer Learning with Model Deployment on Web

## Understanding ESCA in Grapevines: Causes, Occurrence, and Challenges

### What is ESCA?
Esca, often referred to as "grapevine wood disease," is a complex and multifaceted ailment that affects grapevines. It encompasses a range of interconnected fungal infections that collectively compromise the health and vitality of grapevine plants. The primary culprits behind ESCA are Phaeomoniella chlamydospora, Phaeoacremonium spp., and Fomitiporia mediterranea fungi.

### How Does ESCA Occur?
ESCA's onset is primarily linked to wounds and injuries sustained by grapevines during various cultivation practices, such as pruning, grafting, or other forms of vineyard management. The fungi responsible for ESCA inhabit the environment, including the soil and pruning tools, ready to exploit any opportunity to infiltrate the plant's system. Once within the vine, these fungi proceed to colonize the vascular tissues, leading to a cascade of symptoms and consequences.

### When Does ESCA Occur?
ESCA's appearance is not restricted to a specific season; it can manifest throughout the year. However, it becomes more noticeable during specific growth stages or environmental conditions. The disease can impact young grapevines and mature plants alike, underscoring its indiscriminate nature.

### To Whom Does ESCA Occur?
ESCA isn't selective when it comes to its victims. It has been observed affecting a diverse array of grapevine varieties across the globe. From traditional wine regions to emerging viticulture areas, no vineyard is entirely immune to the potential threat of ESCA. The impact can be extensive, causing not only a reduction in yield but also a deterioration in the quality of grapes produced.

### Issues and Challenges:

1. **Delayed and Variable Symptoms:** ESCA's symptoms can emerge years after infection, making early detection challenging. Additionally, the presentation of symptoms can vary, complicating diagnosis.
2. **Multifaceted Nature:** ESCA's complexity arises from the involvement of multiple fungal species, leading to a range of symptoms that include leaf discoloration, wood necrosis, and grape yield decline.
3. **Lack of Clear Cure:** Presently, there's no silver bullet for curing ESCA. Management strategies focus on minimizing its impact and slowing its progression.
4. **Economic Implications:** ESCA-related losses can be substantial, affecting both the quantity and quality of grape production. This economic strain extends to winemakers and vineyard operators.

### Managing ESCA:
Effective ESCA management demands a multi-pronged approach that combines preventative and curative measures:

1. **Sanitation and Pruning:** Timely and meticulous pruning, coupled with proper vineyard hygiene, reduces the potential entry points for fungal infections.
2. **Resistant Varieties:** Opting for grapevine varieties with natural resistance to ESCA can offer a level of protection.
3. **Fungicide Application:** Some fungicides show promise in suppressing the development of ESCA, although their efficacy can vary.
4. **Canopy Management:** Strategic canopy management techniques and balanced vine growth can contribute to inhibiting the disease's progression.

In conclusion, ESCA represents a complex challenge for the viticulture industry. Its intricate nature, delayed symptoms, and economic implications necessitate a proactive and adaptable approach. By integrating comprehensive management strategies and fostering ongoing research, the wine industry can work towards minimizing ESCA's impact and safeguarding the longevity of grapevine health and wine production.

---

## Abstract:

Grapevine health is of utmost importance in viticulture. Esca disease, a common threat to grapevines, can inflict severe damage if not identified and managed promptly. Left untreated, it leads to vegetative stress or plant demise, resulting in production losses and an increased risk of spreading to nearby vines.

Currently, Esca detection relies on manual visual surveys, a time-consuming process conducted by agronomists. The advent of image processing, computer vision, and machine learning techniques presents a transformative solution for disease classification. These advanced methods streamline anomaly detection, enabling early identification of Esca disease in grapevine plants. This intervention is pivotal in curbing its spread within vineyards and mitigating financial losses for wine producers.

In this paper, authors present an image dataset comprising grapevine leaf images categorized into two classes: unhealthy leaves from plants afflicted by Esca disease and healthy leaves. This dataset originates from a collaborative research initiative between the Department of Information Engineering, Polytechnic University of Marche, and industry partners. It serves as a valuable resource for researchers employing machine learning and computer vision algorithms to develop applications that aid agronomists in the early detection of grapevine diseases.

---

## Dataset Description:

### Grapevine Leaf Images for Disease Detection

The dataset at hand offers a valuable collection of grapevine leaf images, meticulously curated to facilitate research in the realm of early disease detection, specifically focusing on Esca disease. This dataset serves as a crucial resource for researchers and practitioners alike, aiming to harness the potential of machine learning techniques for the prompt identification of grapevine diseases.

### Dataset Applications:

1. **Classification Algorithms:** The dataset is well-suited for training, validating, and testing classification algorithms. Researchers can leverage images of both Esca-affected and healthy leaves to fine-tune their models for accurate disease identification.

2. **Application Development:** The dataset's versatility extends to the development of practical applications. Whether on computers, smartphones, or embedded devices, the dataset can fuel the creation of tools designed for the early detection of plant diseases.

3. **Diverse Machine Learning Tasks:** The dataset caters to various machine learning tasks beyond mere classification. It can be used for image detection, image segmentation, and even image synthesis, opening avenues for broader research explorations.

### Data Details:

Grapevine trunk diseases, a group of fungal-driven pathologies affecting grapevines, constitute the context for this dataset. These diseases induce wood necrosis, vascular infections, wood discoloration, and white decays. Notably, the well-documented Esca disease, one of the earliest identified diseases, leads to reduced transmission of organic components within plants, culminating in foliage desiccation and eventual plant death.

### Esca's Impact:

Esca initiates a cascade of consequences, including vegetative stress and stunted grape clusters. In severe cases, the disease progresses to vine death, along with the ominous potential for spreading to neighboring grapevines. As a result, grape production faces considerable losses due to the presence of such diseases and the absence of effective control strategies.

### Dataset Content:

The dataset encompasses images captured from vineyard crops, featuring both healthy and unhealthy grapevine leaves. The latter represents the state of Esca disease. These images span diverse scenarios with varying backgrounds, mirroring real-world conditions. Notably, the manifestation of Esca disease occurs exclusively during the July-September period, coinciding with the optimal climate for its growth.

### Data Collection:

A grapevine disease expert manually captured images during the disease-prone period, utilizing three distinct devices: two smartphones and a tablet. The images exhibit resolutions of 1920×1080 pixels and 1280×720 pixels, characterized by both portrait and landscape orientations.

### Dataset Structure:

The dataset, comprising a total of 1770 images, is structured within the "esca_dataset" main folder. This folder contains two sub-folders:

- "esca

": Contains image files related to the Esca class, denoted as "esca_<n>_<camSource>.jpg."
- "healthy": Contains image files representing the healthy class, named "healthy_<n>_<camSource>.jpg."

An annotation file (.csv) accompanies the dataset, featuring filenames alongside corresponding class IDs. Furthermore, a Jupyter notebook (.ipynb) is provided to facilitate data augmentation, enhancing the dataset's potential for robust model training.

### Reproducibility and Augmentation:

To ensure result reproducibility, a Jupyter notebook specifically designed for Convolutional Neural Network (CNN) training is included.

### Dataset Statistics:

For a comprehensive overview of the dataset, Table 1 furnishes details on class descriptions and the dataset's dimensions pre- and post-augmentation. This dataset stands as a testament to the convergence of technology and agricultural expertise, offering researchers a tangible tool to combat the challenges posed by grapevine diseases, ultimately fortifying the future of viticulture.

---

## 2.1. Processing:

The authors embarked on a comprehensive journey to acquire grapevine leaf images of both unhealthy and healthy states. Employing three distinct devices, each equipped with specific cameras, enabled a diverse and representative image collection:

- Cam1: A 12 MP camera, f/1.8 aperture, optical image stabilization, and autofocus tablet camera.
- Cam2: A 13 MP camera, f/1.7 aperture, autofocus smartphone camera.
- Cam3: A 12 MP camera, f/1.8 aperture, optical image stabilization, and autofocus smartphone camera.

Images from Cam2 and Cam3 boasted a resolution of 1920×1080 pixels, while Cam1 yielded 1280×720 pixels. This variety was coupled with portrait and landscape orientations. These images were captured at a working distance of 30 cm, accounting for potential zoom variations.

## Classification Avenues:

For machine learning purposes, the authors focused on classification. They recognized that classification involves categorizing new observations into classes, and a slew of standard and contemporary techniques are available, including support vector machines, decision trees, k-nearest neighbor classification, and convolutional neural networks (CNNs). These techniques, particularly CNNs for image data, were deemed apt for addressing the dataset's classification problem.

## Labeling and Classes:

The dataset catered to a binary classification problem, with classes labeled as "unhealthy" (class 1) for leaves affected by Esca disease and "healthy" (class 2) for unaffected leaves. An expert's visual inspection ensured accurate labeling, resulting in 888 images for the unhealthy class and 882 images for the healthy class.

## Dataset Augmentation:

To enhance the diversity of the training dataset and foster more accurate deep learning models, data augmentation was harnessed. Data augmentation entails applying realistic yet random transformations to training images. The augmentation process employed the ImageDataGenerator class from the Keras library. The augmentation procedure included:

- Downloading the dataset from the Mendeley repository and storing it in the current path.
- Applying a suite of transformations, ranging from flips, rotations, shifts, and zoom, to color space manipulations like brightness, contrast, and saturation adjustments. The augmentation configuration was tunable, allowing users to tailor transformations to their needs.
- Visualizing original images and their augmented counterparts in a grid for comparative analysis.
- Creating an augmented dataset archive, aptly named "augmented_esca_dataset," facilitating user access and future use.

## Illustrating Augmentation:

Figures 3 and 4 showcase the augmentation process, demonstrating the transformation of starting images into augmented variations. The augmentations include geometric transformations (flips, shifts, rotations) and color transformations (brightness, contrast, saturation, hue, gamma). These transformations simulate diverse conditions, essential for robust model training. Geometric transformations capture the randomness of leaf angles in vineyards, while color transformations emulate varying luminosity and exposure conditions, crucial for realism without distorting Esca spots.

## 2.3. Classification with CNN:

The authors exemplified a machine learning approach by training a CNN using the augmented dataset. They showcased how different pixel sizes (1280×720, 320×180, 80×45) cater to distinct applications, such as web applications and embedded systems. The CNN architecture consisted of convolutional 2D layers with ReLu activation, max pooling layers, a dropout layer, and a final softmax layer for classification.

For training, validation, and testing, the authors divided the augmented dataset into a 60-15-25% split. Their CNN underwent 50 epochs of training, and the results were evaluated in terms of loss and accuracy across these phases.

These meticulously executed steps collectively exemplify the authors' approach to tackling the grapevine disease detection problem, showcasing the integration of data acquisition, classification, augmentation, and CNN training to yield valuable insights into the domain of early Esca detection.
