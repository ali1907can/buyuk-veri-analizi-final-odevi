# Büyük Veri Analizi Final Projesi
**Heart Disease Risk Factors and Patient Health Survey Dataset**

**Öğrenci:** Ali Can  
**Üniversite:** Mersin Üniversitesi  
**Tarih:** Haziran 2026  

---

## Dosya Yapısı

```
├── heart_analysis.py               # Tüm analiz kodu (Bölüm A–F)
├── Heart_Dataset_Cleaned.csv       # Veri seti
├── Rapor_Heart_Disease_Ali_Can.docx   # Tam analiz raporu (Word)
├── Ozet_1Sayfa_Ali_Can.docx           # 1 sayfalık kısa özet
└── figures/                           # Otomatik oluşturulan grafikler
    ├── fig1_heart_patient_dagılımı.png
    ├── fig2_yas_kalp.png
    ├── fig3_cinsiyet_kalp.png
    ├── fig4_bmi.png
    ├── fig5_kan_basinci.png
    ├── fig6_fiziksel_aktivite.png
    ├── fig7_sigara.png
    ├── fig8_diyabet.png
    ├── fig9_stres.png
    ├── fig10_bmi_kilo_scatter.png
    ├── fig11_ilac.png
    ├── fig_ml_confusion_matrix.png
    ├── fig_ml_roc.png
    ├── fig_ml_feature_importance.png
    └── fig_ml_performans_karsilastirma.png
```

## Çalıştırma

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python heart_analysis.py
```

`figures/` klasörü otomatik oluşturulur ve tüm grafikler kaydedilir.

## Kullanılan Yöntemler
- **Veri Analizi:** pandas, numpy
- **Görselleştirme:** matplotlib, seaborn
- **Makine Öğrenmesi:** scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- **Metrikler:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
