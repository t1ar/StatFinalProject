import pandas as pd


def hitung_prior(df: pd.DataFrame, nama_kolom_target):
    """Menghitung probabilitas prior untuk masing-masing kelas target"""
    # value_counts(normalize=True) otomatis menghitung proporsi (jumlah kelas / total data)
    prior = df[nama_kolom_target].value_counts(normalize=True).to_dict()
    return prior


def hitung_likelihood(df, nama_kolom_fitur, nilai_fitur, nama_kolom_target, nilai_target, smooth_version = False) -> float:
    """Menghitung probabilitas Likelihood: P(Fitur | Target)"""
    # 1. Filter data hanya untuk kelas target tertentu (misal: hanya yang 'Disetujui')
    df_target = df[df[nama_kolom_target] == nilai_target]

    # 2. Hitung berapa banyak nilai fitur tersebut muncul di kelas target
    jumlah_fitur_di_target = len(df_target[df_target[nama_kolom_fitur] == nilai_fitur])
    total_data_target = len(df_target)

    # Mencegah pembagian dengan nol jika data tidak ditemukan
    if total_data_target == 0:
        return 0.0

    if smooth_version:
        V = df[nama_kolom_fitur].nunique()
        smooth_likelihood = (jumlah_fitur_di_target + 1) / (total_data_target + V)
        return smooth_likelihood

    likelihood = jumlah_fitur_di_target / total_data_target
    return likelihood


def prediksi_naive_bayes(df: pd.DataFrame, data_uji, nama_kolom_target, smooth_version = False):
    """Menghitung Posterior dan menentukan hasil prediksi"""
    prior = hitung_prior(df, nama_kolom_target)
    daftar_kelas = df[nama_kolom_target].unique()

    hasil_posterior = {}
    hasil_likelihood = {}

    # Menghitung posterior untuk setiap kelas (Disetujui / Ditolak)
    for kelas in daftar_kelas:
        # Rumus Posterior dimulai dari nilai Prior
        prob_posterior = prior[kelas]
        hasil_likelihood[kelas] = {}

        # Kalikan dengan Likelihood dari setiap atribut yang ada di data uji
        for fitur, nilai in data_uji.items():
            likelihood = hitung_likelihood(df, fitur, nilai, nama_kolom_target, kelas, smooth_version)
            prob_posterior *= likelihood  # Mengalikan nilai berantai
            hasil_likelihood[kelas][f"{fitur}->{nilai}"] = likelihood

        hasil_posterior[kelas] = prob_posterior

    # Mencari nilai posterior tertinggi untuk menentukan prediksi akhir
    prediksi_akhir = max(hasil_posterior, key=hasil_posterior.get)

    return prior, hasil_posterior, prediksi_akhir, hasil_likelihood