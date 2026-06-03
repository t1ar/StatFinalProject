import os
import sys
import pandas as pd
import streamlit as st
from calculation import *
 
st.set_page_config(page_title="Prediksi Kredit - Naive Bayes", page_icon="💳", layout="centered")
 
st.title("💳 Prediksi Kelayakan Kredit")
st.markdown("Menggunakan algoritma **Naive Bayes** untuk memprediksi apakah kredit akan disetujui atau ditolak.")
 
# Load CSV
csv_data = 'raw_kasus5.csv'
if not os.path.exists(csv_data):
    st.error("File CSV tidak ditemukan! Pastikan `raw_kasus5.csv` ada di folder yang sama.")
    sys.exit(1)
 
df = pd.read_csv(csv_data)
 
# Show dataset
with st.expander("📊 Lihat Data Kasus"):
    df_display = df.copy()
    df_display.index = range(1, len(df) + 1)
    st.dataframe(df_display, use_container_width=True)
 
st.divider()
 
# Input form
st.subheader("🔍 Masukkan Data Uji")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    penghasilan = st.selectbox("Penghasilan", df['Penghasilan'].unique())
 
with col2:
    pekerjaan = st.selectbox("Pekerjaan", df['Pekerjaan'].unique())
 
with col3:
    riwayat_kredit = st.selectbox("Riwayat Kredit", df['RiwayatKredit'].unique())
 
data_uji = {
    'Penghasilan': penghasilan,
    'Pekerjaan': pekerjaan,
    'RiwayatKredit': riwayat_kredit,
}
 
kolom_target = 'Kredit'
 
smooth = st.toggle("🧮 Gunakan Laplace Smoothing")

# if st.button("🚀 Prediksi Sekarang", use_container_width=True, type="primary"):
prior, posterior, hasil_prediksi, hasil_likelihood = prediksi_naive_bayes(df, data_uji, kolom_target, smooth)

st.divider()
st.subheader("📋 Hasil Analisis")

# Result banner
if hasil_prediksi == "Disetujui":
    st.success(f"✅ Hasil Prediksi: **{hasil_prediksi.upper()}**")
else:
    st.error(f"❌ Hasil Prediksi: **{hasil_prediksi.upper()}**")

# Prior probabilities
st.markdown("**1. Probabilitas Prior**")
prior_df = pd.DataFrame(list(prior.items()), columns=["Kelas", "Probabilitas"])
prior_df["Probabilitas"] = prior_df["Probabilitas"].map("{:.4f}".format)
st.table(prior_df)

# Likelihood
st.markdown("**2. Likelihood Atribut**")
daftar_kelas = list(hasil_likelihood.keys())
daftar_atribut = list(hasil_likelihood[daftar_kelas[0]].keys())

likelihood_rows = []
for atribut in daftar_atribut:
    row = {"Atribut": atribut}
    for kelas in daftar_kelas:
        row[f"P(Atribut | {kelas})"] = hasil_likelihood[kelas][atribut]
    likelihood_rows.append(row)

likelihood_df = pd.DataFrame(likelihood_rows)
st.table(likelihood_df)

# Posterior probabilities
st.markdown("**3. Probabilitas Posterior**")
posterior_df = pd.DataFrame(list(posterior.items()), columns=["Kelas", "Nilai Posterior"])
posterior_df["Nilai Posterior"] = posterior_df["Nilai Posterior"].map("{:.6f}".format)
st.table(posterior_df)

st.markdown("**4. Kesimpulan**")
st.info(f"Berdasarkan nilai posterior tertinggi, data tersebut diprediksi: **{hasil_prediksi.upper()}**")

    