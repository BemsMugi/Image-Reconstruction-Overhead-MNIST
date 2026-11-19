# Image Reconstruction with Autoencoders

This project compares two convolutional autoencoder architectures for reconstructing 28 × 28 grayscale images. The experiments use two classes from the OverheadMNIST dataset: **Storage Tanks** (label 9) and **Parking Lot** (label 4).

Each model compresses an image from 784 pixels into a 128-dimensional latent representation, then reconstructs the image from that representation. Reconstruction quality is evaluated using the **Structural Similarity Index (SSIM)**.

## Objectives

- Apply autoencoders to image compression and reconstruction.
- Compare a baseline model with a modified model.
- Evaluate reconstruction quality on two classes with different visual characteristics.

## Dataset and Preprocessing

The notebook uses **OverheadMNIST version 2**, provided as `train.csv` and `test.csv`. The files are combined, then filtered to include only Storage Tanks and Parking Lot images.

After removing rows with NaN values in the pixel columns, the dataset contains **16,000 images**, with 8,000 samples per class. Pixel values are normalized from the range 0–255 to 0–1.

The data is split separately for each class:

| Split | Samples per class | Proportion |
|---|---:|---:|
| Train | 6,400 | 80% |
| Validation | 800 | 10% |
| Test | 800 | 10% |

The split uses `random_state=42` for reproducibility.

## Models

The project compares two convolutional autoencoders:

- **Baseline Autoencoder**, used as the reference model.
- **Modified Autoencoder**, with a deeper convolutional architecture, Batch Normalization, and Dropout.

A separate model is trained for each class. The Storage Tanks model is trained only on Storage Tanks images, and the Parking Lot model only on Parking Lot images.

> See the notebook for the layer configurations and training details.

## Results

The test set SSIM results reported in the notebook are:

| Class | Baseline SSIM | Modified SSIM | Change |
|---|---:|---:|---:|
| Storage Tanks | 0.7818 ± 0.0892 | 0.7245 ± 0.1174 | -0.0574 |
| Parking Lot | 0.6235 ± 0.1339 | 0.6299 ± 0.1576 | +0.0064 |

The baseline model achieves a higher SSIM on Storage Tanks. For Parking Lot, the modified model performs slightly better, although the improvement is small. Storage Tanks images are generally easier to reconstruct because their geometric features are more consistent, while Parking Lot images contain more varied textures and details.

Both models compress each image from 784 to 128 dimensions, a compression ratio of approximately **6.125×**. These results show that a more complex architecture does not necessarily improve reconstruction quality, and its impact can vary by class.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy and Pandas
- Matplotlib
- scikit-learn
- scikit-image

## Running the Notebook

The notebook is configured for an environment where the OverheadMNIST version 2 dataset is available at the Kaggle paths specified in the code.

1. Open `image-reconstruction.ipynb` in Jupyter Notebook or Google Colab.
2. Make sure the dataset is available, and update `TRAIN_CSV` and `TEST_CSV` if needed.
3. Run the notebook cells in order.
4. The notebook processes the data, trains and evaluates the models, displays visualizations, and saves the models in `.keras` format.

The saved model files are:

- `baseline_storage_tanks.keras`
- `baseline_parking_lot.keras`
- `modified_storage_tanks.keras`
- `modified_parking_lot.keras`

## Limitations

This experiment uses only two classes and 28 × 28 pixel images. The 128-dimensional latent representation limits how much detail can be preserved, especially for the more complex Parking Lot images. Results also depend on the model configuration and data split used in the notebook.
