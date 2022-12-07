import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA")
st.write("##### Nama  : R.Bella Aprilia Damayanti ")
st.write("##### Nim   : 200411100082 ")
st.write("##### Kelas : Penambangan Data B ")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Klasifikasi Kelayakan Calon Pendonor Darah Menggunakan Metode Naive Bayes")
    st.write("###### Studi Kasus PMI Kabupaten Bangkalan")
    st.write("###### Darah adalah cairan yang terdapat pada semua makhluk hidup (kecuali tumbuhan) tingkat tinggi yang berfungsi mengirimkan zat – zat dan oksigen yang dibutuhkan oleh jaringan tubuh, mengangkut bahan – bahan kimia hasil metabolisme dan juga sebagai pertahanan tubuh terhadap virus atau bakteri [1] (Desmawati, 2013).")
    st.write("###### Dalam tubuh orang dewasa, kira – kira 4 sampai 5 liter darah yang beredar terus - menerus melalui jaringan yang rumit mulai dari pembuluh darah, didorong oleh kontraksi kuat dari detak jantung. Setelah darah bergerak menjauh dari paru – paru dan jantung, melewati arteri besar dan mengalir ke jaringan yang sempit dan lebih kompleks dari pembuluh – pembuluh kecil, darah berinteraksi dengan sel – sel individual dari jaringan. Pada tingkat ini, fungsi utamanya adalah untuk memberi makan sel – sel tersebut, memberi mereka nutrisi, termasuk oksigen yang merupakan unsur paling dasar yang diperlukan untuk keberlangsungan hidup manusia. Dalam pertukaran nutrisi bermanfaat ini, darah menggambil dan membawa pergi limbah seluler seperti karbon dioksida yang pada akhirnya akan dikeluarkan dari tubuh ketika darah mengalir kembali ke paru – paru.")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1.Tempat lahir :Dimana seseorang dilahirkan""")
    st.write("""2.Tanggal lahir :Identitas kapan dilahirkan ke dunia setelah berada di kandungan. """)
    st.write("""3.Umur : Umur atau usia pada manusia adalah waktu yang terlewat sejak kelahiran""")
    st.write("""4.Golongan Darah :Golongan darah adalah ilmu pengklasifikasian darah dari suatu kelompok berdasarkan ada atau tidak adanya zat antigen warisan pada permukaan membran sel darah merah.""")
    st.write("""5.Jenis Kelamin :jenis kelamin adalah perbedaan bentuk, sifat, dan fungsi biologis antara laki-laki dan perempuan yang menentukan perbedaan peran mereka dalam menyelenggarakan upaya eneruskan garis keturunan.""")
    st.write("""6.HB :Hb adalah protein yang ada di dalam sel darah merah. Protein inilah yang membuat darah berwarna merah.""")
    st.write("""7.BB (kg) :berat badan tubuh yang memiliki proporsi seimbang dengan tinggi badan""")
    st.write("""8.Tensi :Tensi normal atau tekanan darah normal adalah ukuran ideal dari kekuatan yang digunakan jantung untuk melakukan fungsinya yaitu memompa darah ke seluruh tubuh.""")
    st.write("""9.Status : status kelayakan donor Darah yang diklasifikasikan ke dalam kelas “BOLEH DONOR” dan “TIDAK BOLEH DONOR”""")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link :  https://github.com/RBellaApriliaDamayanti22/Datasets")

with upload_data:
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    df = pd.read_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/Datasets/main/kelompok8.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=["NO"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['STATUS'])
    y = df['STATUS'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Grade).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '0' : [dumies[0]],
        '1' : [dumies[1]],
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))


        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi],
                'Model' : ['Gaussian Naive Bayes'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        UMUR = st.number_input('Masukkan Umur : ')
        GOLONGAN_DARAH = st.number_input('Masukkan golongan darah : ')
        JENIS_KELAMIN = st.number_input('Masukkan jenis kelamin : ')
        HEMOGLOBIN = st.number_input('Masukkan hemoglobin : ')
        BERAT_BADAN = st.number_input('Masukkan berat badan : ')
        TENSI = st.number_input('Masukkan tensi : ')
        model = st.selectbox('model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                UMUR,
                GOLONGAN_DARAH,
                JENIS_KELAMIN,
                HEMOGLOBIN,
                BERAT_BADAN,
                TENSI
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
           
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
  