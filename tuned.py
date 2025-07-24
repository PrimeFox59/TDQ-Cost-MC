import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="TOTAL COST MC - Portofolio App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Memuat Model dari file pkl ---
try:
    with open('tuned_gradient_boosting_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'tuned_gradient_boosting_model.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Mengatur Tabs ---
tab1, tab2 = st.tabs(["Aplikasi Prediksi", "Proses Pengolahan Data & Model"])

# --- TAB 1: Aplikasi Prediksi ---
with tab1:
    st.title('TOTAL COST MC Prediction App')
    st.markdown("""
    Selamat datang! Aplikasi ini memungkinkan Anda memprediksi **Total Cost Machining**
    dan menghitung harga jual produk. Gunakan panel di sebelah kiri untuk memasukkan nilai parameter.
    """)
    st.markdown("---")

    # --- Sidebar untuk Input ---
    st.sidebar.header('Input Parameter Prediksi')
    st.sidebar.markdown("Masukkan nilai untuk memprediksi **Total Cost MC**.")

    depresiasi_mesin = st.sidebar.number_input(
        'Depresiasi Mesin', 
        min_value=0.0, 
        value=1000.0,
        help="Nilai depresiasi mesin per unit."
    )
    casting_weight = st.sidebar.number_input(
        'Casting Weight (kg)', 
        min_value=0.0, 
        value=1.0,
        help="Berat bahan baku casting."
    )
    mct_total = st.sidebar.number_input(
        'MCT TOTAL (menit)', 
        min_value=0.0, 
        value=1.0,
        help="Total waktu machining yang diperlukan."
    )

    st.sidebar.markdown("---")

    st.sidebar.header('Pengaturan Profit')
    margin_percent = st.sidebar.slider(
        'Margin Keuntungan (%)', 
        min_value=0, 
        max_value=100, 
        value=20, 
        step=1,
        help="Geser untuk menentukan persentase profit dari total biaya."
    )

    # --- Tombol Prediksi ---
    if st.sidebar.button('Predict TOTAL COST MC & PROFIT'):
        if casting_weight == 0 or mct_total == 0:
            st.error("Nilai 'Casting Weight' dan 'MCT TOTAL' tidak boleh nol. Silakan perbaiki input Anda.")
        else:
            input_data = pd.DataFrame({
                'Depresiasi Mesin': [depresiasi_mesin],
                'Casting Weight': [casting_weight],
                'MCT TOTAL': [mct_total]
            })

            try:
                prediction = model.predict(input_data)[0]
                profit = prediction * (margin_percent / 100)
                selling_price = prediction + profit

                st.subheader("Hasil Prediksi")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediksi TOTAL COST MC", f"Rp {prediction:,.2f}")
                with col2:
                    st.metric(f"Profit (@{margin_percent}%)", f"Rp {profit:,.2f}")
                with col3:
                    st.metric("Prediksi Harga Jual", f"Rp {selling_price:,.2f}")
                
                st.success("Prediksi berhasil ditampilkan!")
                st.markdown("---")

                st.subheader("Visualisasi Perbandingan Biaya, Keuntungan, dan Harga Jual")
                data = {'Kategori': ['Total Cost MC', 'Profit', 'Harga Jual'],
                        'Nilai': [prediction, profit, selling_price]}
                df_results = pd.DataFrame(data)
                fig = px.bar(df_results, x='Kategori', y='Nilai', color='Kategori',
                             title='Perbandingan Total Biaya, Keuntungan, dan Harga Jual',
                             labels={'Kategori': 'Kategori Biaya', 'Nilai': 'Jumlah (Rp)'},
                             text='Nilai')
                fig.update_traces(texttemplate='Rp %{y:,.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {e}")

    # --- Footer ---
    st.markdown("---")
    st.caption("Versi Demo - Prediksi berdasarkan model Gradient Boosting yang telah dituning.")


# --- TAB 2: Proses Pengolahan Data ---
with tab2:
    st.title('Proses Pengolahan Data dan Alur Kerja Model')
    st.markdown("""
    Bagian ini menjelaskan langkah-langkah yang dilakukan untuk membersihkan data,
    menganalisis fitur, dan melatih model prediktif **Gradient Boosting** yang digunakan dalam aplikasi ini.
    """)
    
    st.markdown("---")

    # --- Simulasi Proses Olah Data ---
    st.header('1. Memuat Data & Eksplorasi Awal')
    st.write("Data dimuat dari file `data mc.csv` dan 5 baris pertama ditampilkan untuk melihat struktur data.")
    st.code("""
        import pandas as pd
        df = pd.read_csv('data mc.csv', encoding='latin1')
        print(df.head())
        print(df.info())
    """)
    st.write("Output `df.info()` menunjukkan banyak kolom dengan tipe data `object` yang perlu dibersihkan.")

    st.header('2. Pembersihan & Pra-proses Data')
    st.write("Langkah-langkah berikut dilakukan untuk mempersiapkan data:")
    st.markdown("- **Menghapus kolom** yang tidak relevan ('No.', 'Nama File', dll.).")
    st.markdown("- **Membersihkan nilai non-numerik** (`'N/A'`, `'#VALUE!'`) dan mengubah tipe data kolom biaya menjadi numerik.")
    st.markdown("- **Menghapus duplikasi** dan baris dengan nilai kosong (`NaN`) pada kolom-kolom kunci.")
    st.code("""
        # Hapus kolom yang tidak relevan
        df = df.drop(['No.', 'Nama File', 'Item', 'Part No', 'Cust', 'Date'], axis=1)

        # Ubah tipe data kolom ke numerik
        cols_to_convert = ['Insert', 'Oil Cost', 'Jig, Tool & Maint', 'Man Power Direct', 'Electricity', 'Depresiasi Mesin', 'TOTAL COST MC', 'Casting Weight', 'MCT TOTAL']
        for col in cols_to_convert:
            df[col] = df[col].replace(['N/A', '#VALUE!', 'Not found'], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Hapus baris dengan nilai kosong di kolom kunci
        df.dropna(subset=['TOTAL COST MC'], inplace=True)
        
        # Hapus data duplikat
        df.drop_duplicates(inplace=True)
    """)

    st.header('3. Analisis Fitur & Seleksi')
    st.write("Analisis korelasi dan VIF (Variance Inflation Factor) dilakukan untuk memilih fitur yang paling relevan.")
    
    st.subheader('Korelasi Fitur')
    st.write("Heatmap korelasi menunjukkan hubungan antara setiap fitur dengan variabel target **`TOTAL COST MC`**.")
    st.code("""
        # Tampilkan 5 fitur dengan korelasi tertinggi terhadap 'TOTAL COST MC'
        corr_matrix = df.corr()
        print(corr_matrix['TOTAL COST MC'].sort_values(ascending=False).head())
    """)

    st.subheader('Analisis Multikolinieritas (VIF)')
    st.write("VIF digunakan untuk mendeteksi multikolinieritas (korelasi antar fitur). Fitur dengan nilai VIF yang sangat tinggi dipertimbangkan untuk dihilangkan.")
    st.code("""
        # Contoh perhitungan VIF
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    """)
    st.write("Berdasarkan analisis ini, fitur-fitur seperti `Insert`, `Oil Cost`, `Man Power`, dll., dihilangkan karena korelasi yang rendah atau multikolinieritas yang tinggi.")

    st.header('4. Pelatihan & Evaluasi Model')
    st.write("Model **Random Forest** dan **Gradient Boosting** dilatih dan dievaluasi untuk membandingkan performanya.")
    st.code("""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score

        # Pisahkan data latih dan uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Latih model Gradient Boosting
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X_train, y_train)

        # Evaluasi model
        y_pred_gb = gb_model.predict(X_test)
        r2_gb = r2_score(y_test, y_pred_gb)
        print(f"R-squared (R2) Gradient Boosting: {r2_gb:.2f}")
    """)

    st.subheader('Hyperparameter Tuning')
    st.write("Untuk mendapatkan performa terbaik, model Gradient Boosting disetel menggunakan **`GridSearchCV`**.")
    st.code("""
        from sklearn.model_selection import GridSearchCV
        param_grid = { ... }
        grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, ...)
        grid_search.fit(X_train, y_train)
        best_gb_model = grid_search.best_estimator_
    """)
    st.write("Model yang telah disetel ini menunjukkan performa terbaik dan digunakan dalam aplikasi prediksi.")

    st.header('5. Penyimpanan Model')
    st.write("Model **Gradient Boosting** terbaik yang telah disetel disimpan dalam file **`tuned_gradient_boosting_model.pkl`** menggunakan library `pickle` agar dapat dimuat kembali dan digunakan untuk prediksi tanpa perlu dilatih ulang.")
    st.code("""
        import pickle
        with open('tuned_gradient_boosting_model.pkl', 'wb') as file:
            pickle.dump(best_gb_model, file)
    """)