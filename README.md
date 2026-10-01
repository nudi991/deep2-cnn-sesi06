# CNN Cats vs Dogs

Implementasi Convolutional Neural Network (CNN) untuk klasifikasi gambar **Cat** dan **Dog** menggunakan Cats vs Dogs Dataset.

## Dataset

Dataset yang digunakan:

**Cats vs Dogs Dataset**

Struktur dataset:

```
data/
└── PetImages/
    ├── Cat/
    └── Dog/
```

Dataset berisi 25.000 gambar:

* Cat: 12.500
* Dog: 12.500

Pada proses pemeriksaan ditemukan 2 file gambar yang corrupt:

* `Cat/666.jpg`
* `Dog/11702.jpg`

Kedua file tersebut tidak digunakan dalam proses training, validation, maupun testing.

Setelah mengeluarkan file corrupt, terdapat 24.998 gambar valid.

## Dataset Split

Dataset dibagi secara stratified dengan random seed `42`:

| Dataset    | Jumlah |
| ---------- | -----: |
| Train      | 17.498 |
| Validation |  3.750 |
| Test       |  3.750 |

Test set hanya digunakan untuk evaluasi akhir dan tidak digunakan untuk pemilihan model atau hyperparameter.

## Preprocessing

Setiap gambar diproses dengan:

* Resize menjadi `128 × 128`
* Convert ke RGB
* Pixel value diubah menjadi range `0–1` menggunakan `ToTensor()`

Batch size yang digunakan adalah `32`.

## Model

Model menggunakan CNN sederhana dengan 3 convolution block:

```
Input: 128 × 128 × 3

Conv2D: 3 → 32, kernel 3×3, padding 1
ReLU
MaxPool 2×2

Conv2D: 32 → 64, kernel 3×3, padding 1
ReLU
MaxPool 2×2

Conv2D: 64 → 128, kernel 3×3, padding 1
ReLU
MaxPool 2×2

AdaptiveAvgPool2D(1)

Linear: 128 → 1
```

Output model berupa satu logit untuk binary classification.

Loss function yang digunakan adalah `BCEWithLogitsLoss`.

Optimizer yang digunakan adalah `Adam`.

## Training Configuration

| Parameter     | Value              |
| ------------- | ------------------ |
| Random seed   | 42                 |
| Input size    | 128 × 128          |
| Batch size    | 32                 |
| Optimizer     | Adam               |
| Learning rate | 0.001              |
| Loss          | BCEWithLogitsLoss  |
| Epochs        | 10                 |
| Device        | CUDA jika tersedia |

Model terbaik dipilih berdasarkan validation loss.

## Cara Menjalankan

### 1. Local

Pastikan dataset berada pada:

```
data/PetImages/Cat/
data/PetImages/Dog/
```

Pada `train.py`, gunakan:

```
DATASET_PATH = Path("data")
```

Install dependency:

```
pip install -r requirements.txt
```

Jalankan:

```
python3 train.py
```

### 2. Google Colab

Upload atau clone project ke Google Colab, kemudian gunakan dataset dari Kaggle.

Pada `train.py`, ubah:

```python
DATASET_PATH = Path("data")
```

menjadi:

```python
DATASET_PATH = Path("/kaggle/input/microsoft-catsvsdogs-dataset")
```

Setelah itu, jalankan seluruh kode dengan memilih **Runtime → Run all**.

Dataset akan dibaca dari:

```text
/kaggle/input/microsoft-catsvsdogs-dataset/PetImages/
├── Cat/
└── Dog/
```


## Output

Training menghasilkan beberapa informasi dan visualisasi yang digunakan untuk evaluasi model:

* Training dan validation loss
* Training dan validation accuracy
* Test loss dan test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Contoh gambar yang salah diklasifikasikan

Hasil visualisasi disimpan di folder `outputs/`.

## Reproducibility

Training menggunakan random seed `42` untuk proses yang menggunakan randomness, termasuk pembagian dataset dan inisialisasi model.

Pembagian dataset menggunakan stratified split sehingga proporsi Cat dan Dog tetap terjaga pada train, validation, dan test set.
