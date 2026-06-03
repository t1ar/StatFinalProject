import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
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

smooth = st.toggle("🧮 Gunakan Laplace Smoothing", value=True)

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
prior_df = pd.DataFrame(list(prior.items()), columns=["Kelas", "Probabilitas"])
st.markdown("**1. Probabilitas Prior**")
# Membuat Pie Chart dengan Plotly Express
fig_prior = px.pie(
    prior_df,
    values='Probabilitas',
    names='Kelas',
    color_discrete_sequence=px.colors.qualitative.Pastel, # Mengubah tema warna agar estetik
    hole=0.4 # Opsional: Membuatnya jadi Donut Chart agar lebih modern
)

# Mengatur agar label persentase dan teks muncul dengan rapi
fig_prior.update_traces(textposition='inside', textinfo='percent+label')

# Menampilkan di Streamlit
st.plotly_chart(fig_prior, use_container_width=True)

# Likelihood
st.markdown("**2. Likelihood Atribut**")

# 1. Melt DataFrame
df_melted = df.melt(
    id_vars=['Kredit'],
    value_vars=['Penghasilan', 'Pekerjaan', 'RiwayatKredit'],
    var_name='Jenis_Atribut',
    value_name='Nilai_Kategori'
)

# 2. Gabungkan nama atribut dan nilainya agar sumbu X lebih jelas
# Hasilnya misal: "Penghasilan: Tinggi", "Pekerjaan: Tetap", dll.
df_melted['Label_X'] = df_melted['Jenis_Atribut'] + ": " + df_melted['Nilai_Kategori']

# Set warna kustom
warna_keputusan = {'Disetujui': '#2ecc71', 'Ditolak': '#e74c3c'}

# 3. Buat Histogram dalam 1 Grafik Utuh (tanpa facet_col)
fig_combined = px.histogram(
    df_melted,
    x="Label_X", # Gunakan label yang sudah digabung
    color="Kredit",
    barmode="group",
    title="Distribusi Keseluruhan Kategori terhadap Keputusan Kredit",
    text_auto=True,
    color_discrete_map=warna_keputusan
)

# Rapikan label sumbu X agar lebih mudah dibaca
fig_combined.update_layout(
    xaxis_title="Kategori Atribut",
    yaxis_title="Jumlah Pelamar",
    xaxis={'categoryorder':'category ascending'} # Mengurutkan secara alfabetis (opsional)
)

st.plotly_chart(fig_combined, use_container_width=True)

# daftar_kelas = list(hasil_likelihood.keys())
# daftar_atribut = list(hasil_likelihood[daftar_kelas[0]].keys())
#
# likelihood_rows = []
# for atribut in daftar_atribut:
#     row = {"Atribut": atribut}
#     for kelas in daftar_kelas:
#         row[f"P(Atribut | {kelas})"] = hasil_likelihood[kelas][atribut]
#     likelihood_rows.append(row)
#
# likelihood_df = pd.DataFrame(likelihood_rows)
# st.table(likelihood_df)

# Posterior probabilities
st.markdown("**3. Probabilitas Posterior**")

# Membuat DataFrame baru khusus untuk grafik dari dictionary 'posterior'
# (Agar angkanya tetap float, bukan string yang sudah diformat)
posterior_plot_df = pd.DataFrame({
    "Kelas": list(posterior.keys()),
    "Probabilitas": list(posterior.values())
})

# Set warna kustom agar konsisten dengan grafik sebelumnya
warna_keputusan = {'Disetujui': '#2ecc71', 'Ditolak': '#e74c3c'}

# Membuat Bar Chart Horizontal
fig_posterior = px.bar(
    posterior_plot_df,
    x="Probabilitas",
    y="Kelas",
    orientation='h', # Menjadikan batang horizontal
    color="Kelas",
    color_discrete_map=warna_keputusan,
    title="Kekuatan Keputusan (Posterior)"
)

# Menampilkan angka di ujung batang agar jelas
fig_posterior.update_traces(
    texttemplate='%{x:.6f}',
    textposition='outside' # Teks ditaruh di luar/ujung batang
)

# Merapikan layout grafik
fig_posterior.update_layout(
    xaxis_title="Nilai Probabilitas Posterior",
    yaxis_title="",
    showlegend=False # Legend disembunyikan karena label Y sudah jelas
)

# Menampilkan di Streamlit
st.plotly_chart(fig_posterior, use_container_width=True)
# posterior_df = pd.DataFrame(list(posterior.items()), columns=["Kelas", "Nilai Posterior"])
# posterior_df["Nilai Posterior"] = posterior_df["Nilai Posterior"].map("{:.6f}".format)
# st.table(posterior_df)

st.markdown("**4. Kesimpulan**")
st.info(f"Berdasarkan nilai posterior tertinggi, data tersebut diprediksi: **{hasil_prediksi.upper()}**")

