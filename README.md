# KBJ-Final Project

## Laporan
### Safril:
1) Dataset diambil dari website CoMoFoD dengan folder bernama "aug_included"
2) Semua file .ipynb ada pada folder program.
3) Dalam folder "Program" tersebut terdapat beberapa eksperimen uji coba metode CNN dengan menggunakan metode Transfer Learning MobileNet, VGG-16, dan VGG-19 sesuai dengan referensi review paper yang berjudul "Development of precise forgery detection algorithms in digital radiography images using convolution neural network".
4) Dalam file percobaan 1,2,3 belum menggunakan Ekstraksi Fitur Wavelete namun sudah dilakukan proses Augmentasi .

*Catatan saat ini :*
- Program masih dalam tahap perancangan komposisi Arsitektur, yaitu melakukan train model dengan membandingkan antara 3 metode.

### Rudy:
1) folder data_preparation berisi program python untuk memanipulasi dataset, baik untuk pre-processing atau train-val-test split.
2) dataset berisi folder-folder dataset yang mana 1_0 adalah CoMoFod asli, 1_1 hanya citra nya saja, dan 1_2 termasuk augmentasi
3) **Perhatian**, untuk menjalankan data_preparation nomor 2, pastikan ada folder 1_1 atau 1_2 di folder dataset.
4) Jika ingin mengganti train-val-test split ke folder 1_2, ubah notasi is_aug pada data_preparation nomor 2 ke True
