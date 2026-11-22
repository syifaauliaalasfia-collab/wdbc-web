import streamlit as st
from core import model_terlatih

st.title("✅ Kesimpulan & Saran")
_, tabel, *_ = model_terlatih()
b = tabel.index[0]; r = tabel.loc[b]
st.success(f"Model terbaik pada data uji: **{b}** (Recall {r['Recall']:.3f}, F1 {r['F1-Score']:.3f}, "
           f"ROC-AUC {r['ROC-AUC']:.3f}, ganas terlewat {int(r['FN'])})")
st.markdown("""
### Kesimpulan
- Data WDBC bersih (569 × 30, tanpa missing value dan duplikat), kelas sedikit tidak seimbang (B 62,7% : M 37,3%), skala dan korelasi antarfitur bermasalah.
- **RFE dan PCA** mengurangi 30 fitur menjadi sekitar 10 tanpa menurunkan akurasi secara berarti.
- **EDA:** gunakan Pipeline agar tidak ada kebocoran data, jangan asal menghapus outlier, dan buang kolom ID.
- **Evaluasi data medis:** lihat Recall, F1, dan ROC-AUC, tidak hanya akurasi, karena kasus ganas yang terlewat paling berbahaya.
- Logistic Regression dan SVM berkinerja sangat dekat, model ensemble tidak otomatis lebih baik pada data ini.

### Saran
- Lakukan *hyperparameter tuning* (GridSearchCV) untuk SVM dan Logistic Regression.
- Validasi pada data dari rumah sakit lain untuk menguji generalisasi.
- Atur ambang probabilitas (threshold) lebih rendah dari 0,5 jika ingin menekan False Negative.
- Hasil aplikasi adalah alat bantu belajar, **bukan pengganti diagnosis dokter**.
""")
