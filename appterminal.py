import os.path
import sys

import pandas as pd
from calculation import *

if __name__ == "__main__":
    csv_data = 'raw_kasus5.csv'
    if not os.path.exists(csv_data):
        print ("CSV doesnt exist")
        sys.exit(1)
    else:
        try:
            df = pd.read_csv(csv_data)
            print("Loaded CSV!")
        except Exception as e:
            print(e)
            sys.exit(1)

    data_uji = {
        'Penghasilan': 'Sedang',
        'Pekerjaan': 'Tetap',
        'RiwayatKredit': 'Baik'
    }

    kolom_target = 'Kredit'

    prior, posterior, hasil_prediksi, hasil_likelihood = prediksi_naive_bayes(df, data_uji, kolom_target)

    print("=== PROGRAM NAIVE BAYES ===")
    print(f"\nData Uji yang dimasukkan:")
    for key, value in data_uji.items():
        print(f"   {key}: {value}")

    print(f"\nData Kasus:\n{df.to_string(index=False, col_space=15)}\n")

    print("1. PROBABILITAS PRIOR:")
    for kelas, prob in prior.items():
        print(f"   P({kelas}) = {prob:.4f}")

    print("\n2. LIKELIHOOD ATRIBUT:")
    daftar_kelas = list(hasil_likelihood.keys())
    daftar_atribut = list(hasil_likelihood[daftar_kelas[0]].keys())

    for atribut in daftar_atribut:
        for kelas in daftar_kelas:
            print(f"   P({atribut} | {kelas}) = {hasil_likelihood[kelas][atribut]}")
        print("")
        

    print("3. PROBABILITAS POSTERIOR:")
    for kelas, prob in posterior.items():
        print(f"   P({kelas} | Data Uji) \u221d {prob:.4f}")

    print("\n4. KESIMPULAN:")
    print(f"   Berdasarkan nilai posterior tertinggi, data tersebut diprediksi: >>> {hasil_prediksi.upper()} <<<")
