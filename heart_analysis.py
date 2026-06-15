"""
Büyük Veri Analizi Final Projesi
Heart Disease Risk Factors and Patient Health Survey Dataset

Öğrenci: Ali Can
Üniversite: Mersin Üniversitesi
Tarih: Haziran 2026

Gereksinimler: pip install pandas numpy matplotlib seaborn scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # Sunucularda / headless ortamda
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# Çıktı klasörü
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {"Yes": "#C0392B", "No": "#2E86AB"}

# =============================================================================
# BÖLÜM A — VERİ SETİNİ TANIMA
# =============================================================================
print("=" * 65)
print("BÖLÜM A — VERİ SETİNİ TANIMA")
print("=" * 65)

df = pd.read_csv("Heart_Dataset_Cleaned.csv")

print(f"\nSatır sayısı   : {df.shape[0]}")
print(f"Sütun sayısı   : {df.shape[1]}")
print(f"\nDeğişken isimleri:\n{list(df.columns)}")

kategorik = df.select_dtypes(exclude="number").columns.tolist()
sayisal   = df.select_dtypes(include="number").columns.tolist()
print(f"\nKategorik değişkenler ({len(kategorik)}): {kategorik}")
print(f"Sayısal değişkenler  ({len(sayisal)}): {sayisal}")

print(f"\nHedef değişken: Heart patient")
print(f"\nEksik değer sayısı (sütun bazlı):\n{df.isnull().sum()}")
print(f"\nToplam eksik değer: {df.isnull().sum().sum()}")
print(f"Yinelenen (duplicate) kayıt sayısı: {df.duplicated().sum()}")

# =============================================================================
# BÖLÜM B — TANIMLEYICI İSTATİSTİKLER
# =============================================================================
print("\n" + "=" * 65)
print("BÖLÜM B — TANIMLEYICI İSTATİSTİKLER")
print("=" * 65)

print("\n--- Sayısal Değişkenler ---")
print(df[sayisal].describe().round(3))

print("\n--- Kategorik Değişken Frekans Tabloları ---")
for col in kategorik:
    freq = df[col].value_counts()
    pct  = df[col].value_counts(normalize=True).mul(100).round(2)
    tbl  = pd.DataFrame({"Frekans": freq, "Yüzde (%)": pct})
    print(f"\n{col}:\n{tbl}")

hp = df["Heart patient"].value_counts()
hp_pct = df["Heart patient"].value_counts(normalize=True).mul(100).round(2)
print(f"\nKalp hastası sayısı: {hp['Yes']}  ({hp_pct['Yes']}%)")
print(f"Kalp hastası olmayanlar: {hp['No']}  ({hp_pct['No']}%)")

gender = df["Gender"].value_counts()
print(f"\nCinsiyet dağılımı:\n{gender}")

print(f"\nEn yaygın yaş grubu: {df['Age'].mode()[0]}")
print(f"En yaygın kan basıncı: {df['Blood Pressure'].mode()[0]}")

# =============================================================================
# BÖLÜM C — VERİ GÖRSELLEŞTİRME
# =============================================================================
print("\n" + "=" * 65)
print("BÖLÜM C — VERİ GÖRSELLEŞTİRME")
print("=" * 65)

# ── Grafik 1: Kalp Hastası Dağılımı (Bar Chart) ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Heart patient"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[k] for k in counts.index], edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_title("Kalp Hastası Dağılımı", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Kalp Hastası")
ax.set_ylabel("Kişi Sayısı")
ax.set_ylim(0, 270)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_heart_patient_dagılımı.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 1 kaydedildi.")

# ── Grafik 2: Yaş Grubuna Göre Kalp Hastalığı ────────────────────────────
age_order = ["< 35", "35–50", "> 50"]
fig, ax = plt.subplots(figsize=(7, 4.5))
age_hp = df.groupby(["Age", "Heart patient"]).size().unstack(fill_value=0).reindex(age_order)
age_hp.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
            edgecolor="white", linewidth=1, width=0.65)
ax.set_title("Yaş Grubuna Göre Kalp Hastalığı Dağılımı", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Yaş Grubu")
ax.set_ylabel("Kişi Sayısı")
ax.set_xticklabels(age_order, rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_yas_kalp.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 2 kaydedildi.")

# ── Grafik 3: Cinsiyete Göre Kalp Hastalığı ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
gen_hp = df.groupby(["Gender", "Heart patient"]).size().unstack(fill_value=0)
gen_hp.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
            edgecolor="white", linewidth=1, width=0.55)
ax.set_title("Cinsiyete Göre Kalp Hastalığı Dağılımı", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Cinsiyet")
ax.set_ylabel("Kişi Sayısı")
ax.set_xticklabels(["Kadın", "Erkek"], rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_cinsiyet_kalp.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 3 kaydedildi.")

# ── Grafik 4: BMI Dağılımı (Histogram + Boxplot) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].hist(df["BMI"], bins=20, color="#2980B9", edgecolor="white", linewidth=0.8)
axes[0].axvline(df["BMI"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Ort: {df['BMI'].mean():.2f}")
axes[0].axvline(df["BMI"].median(), color="orange", linestyle="--", linewidth=1.5,
                label=f"Med: {df['BMI'].median():.2f}")
axes[0].set_title("BMI Dağılımı (Histogram)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Frekans")
axes[0].legend(fontsize=9)
axes[0].spines[["top", "right"]].set_visible(False)

yes_bmi = df[df["Heart patient"] == "Yes"]["BMI"]
no_bmi  = df[df["Heart patient"] == "No"]["BMI"]
bp_plot = axes[1].boxplot([no_bmi, yes_bmi], patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
bp_plot["boxes"][0].set_facecolor(COLORS["No"])
bp_plot["boxes"][1].set_facecolor(COLORS["Yes"])
axes[1].set_title("BMI: Kalp Hastası vs Değil", fontsize=13, fontweight="bold")
axes[1].set_xticklabels(["Kalp Hastası Değil", "Kalp Hastası"])
axes[1].set_ylabel("BMI")
axes[1].spines[["top", "right"]].set_visible(False)
plt.suptitle("BMI Dağılımı Analizi", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_bmi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 4 kaydedildi.")

# ── Grafik 5: Kan Basıncı ve Kalp Hastalığı ──────────────────────────────
bp_order = ["Hypotension", "Normal", "Pre-hypertension", "Hypertension"]
fig, ax = plt.subplots(figsize=(8, 4.5))
bp_df = (df.groupby(["Blood Pressure", "Heart patient"]).size()
           .unstack(fill_value=0).reindex(bp_order))
bp_df.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
           edgecolor="white", linewidth=1, width=0.65)
ax.set_title("Kan Basıncı Kategorisi ve Kalp Hastalığı İlişkisi",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Kan Basıncı")
ax.set_ylabel("Kişi Sayısı")
ax.set_xticklabels(["Hipotansiyon", "Normal", "Pre-Hipertansiyon", "Hipertansiyon"],
                   rotation=15, ha="right")
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_kan_basinci.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 5 kaydedildi.")

# ── Grafik 6: Fiziksel Aktivite ve Kalp Hastalığı ─────────────────────────
pa_order = ["Rarely / Never", "Less than 2 days per week",
            "2–4 days per week", "5 or more days per week"]
fig, ax = plt.subplots(figsize=(9, 4.5))
pa_df = (df.groupby(["Physical activity", "Heart patient"]).size()
           .unstack(fill_value=0).reindex(pa_order))
pa_pct = pa_df.div(pa_df.sum(axis=1), axis=0) * 100
pa_pct.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
            edgecolor="white", linewidth=1, width=0.65)
ax.set_title("Fiziksel Aktivite Düzeyine Göre Kalp Hastalığı Oranı",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Fiziksel Aktivite")
ax.set_ylabel("Yüzde (%)")
ax.set_xticklabels(["Hiç Yok", "Haftada <2 Gün", "Haftada 2–4 Gün", "Haftada ≥5 Gün"],
                   rotation=15, ha="right")
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_fiziksel_aktivite.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 6 kaydedildi.")

# ── Grafik 7: Sigara / Tütün Kullanımı ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
smoke_df = df.groupby(["Smoke or Tobacco", "Heart patient"]).size().unstack(fill_value=0)
smoke_pct = smoke_df.div(smoke_df.sum(axis=1), axis=0) * 100
smoke_pct.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
               edgecolor="white", linewidth=1, width=0.6)
ax.set_title("Sigara / Tütün Kullanımı ve Kalp Hastalığı İlişkisi",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Sigara Kullanımı")
ax.set_ylabel("Yüzde (%)")
ax.set_xticklabels(["Hiç", "Ara Sıra", "Düzenli"], rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_sigara.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 7 kaydedildi.")

# ── Grafik 8: Diyabet ve Kalp Hastalığı ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
dia_df = df.groupby(["Diabetes", "Heart patient"]).size().unstack(fill_value=0)
dia_pct = dia_df.div(dia_df.sum(axis=1), axis=0) * 100
dia_pct.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
             edgecolor="white", linewidth=1, width=0.6)
ax.set_title("Diyabet Durumu ve Kalp Hastalığı İlişkisi",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Diyabet Durumu")
ax.set_ylabel("Yüzde (%)")
ax.set_xticklabels(["Diyabet", "Normal", "Prediyabet"], rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_diyabet.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 8 kaydedildi.")

# ── Grafik 9: Stres Düzeyi ve Kalp Hastalığı ─────────────────────────────
stress_order = ["Never", "Low", "Moderate", "High"]
fig, ax = plt.subplots(figsize=(7, 4.5))
str_df = (df.groupby(["Feel stressed", "Heart patient"]).size()
            .unstack(fill_value=0).reindex(stress_order))
str_pct = str_df.div(str_df.sum(axis=1), axis=0) * 100
str_pct.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
             edgecolor="white", linewidth=1, width=0.6)
ax.set_title("Stres Düzeyi ve Kalp Hastalığı İlişkisi",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Stres Düzeyi")
ax.set_ylabel("Yüzde (%)")
ax.set_xticklabels(["Hiç", "Düşük", "Orta", "Yüksek"], rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_stres.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 9 kaydedildi.")

# ── Grafik 10: BMI — Kilo Scatter ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for hp_val, grp in df.groupby("Heart patient"):
    ax.scatter(grp["Weight"], grp["BMI"], alpha=0.55, s=30,
               color=COLORS[hp_val], label=f"{'Evet' if hp_val == 'Yes' else 'Hayır'}",
               edgecolors="white", linewidths=0.3)
ax.set_title("Kilo ve BMI İlişkisi (Kalp Hastası Durumuna Göre)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Kilo (kg)")
ax.set_ylabel("BMI")
ax.legend(title="Kalp Hastası", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_bmi_kilo_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 10 kaydedildi.")

# ── Grafik 11: İlaç Kullanımı ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
med_df = df.groupby(["Medicines", "Heart patient"]).size().unstack(fill_value=0)
med_pct = med_df.div(med_df.sum(axis=1), axis=0) * 100
med_pct.plot(kind="bar", ax=ax, color=[COLORS["No"], COLORS["Yes"]],
             edgecolor="white", linewidth=1, width=0.6)
ax.set_title("İlaç Kullanımı ve Kalp Hastalığı İlişkisi",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("İlaç Kullanımı")
ax.set_ylabel("Yüzde (%)")
ax.set_xticklabels(["Kullanıyor", "Hiç\nKullanmadı", "Geçmişte\nKullandı"], rotation=0)
ax.legend(title="Kalp Hastası", labels=["Hayır", "Evet"])
ax.spines[["top", "right"]].set_visible(False)
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f%%", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig11_ilac.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafik 11 kaydedildi.")

# =============================================================================
# BÖLÜM D — RİSK FAKTÖRLERİ ANALİZİ
# =============================================================================
print("\n" + "=" * 65)
print("BÖLÜM D — RİSK FAKTÖRLERİ ANALİZİ")
print("=" * 65)

risk_factors = [
    "Age", "Gender", "BMI", "Blood Pressure", "Diabetes",
    "Family Heart problems History", "Physical activity",
    "Food habits", "Sleep at night", "Depression",
    "Smoke or Tobacco", "Alcohol", "Feel stressed", "Medicines"
]

for factor in risk_factors:
    if factor in sayisal:
        g = df.groupby("Heart patient")[factor]
        print(f"\n[{factor}] — Kalp Hastası Ortalama: {g.get_group('Yes').mean():.2f}"
              f" | Değil Ortalama: {g.get_group('No').mean():.2f}")
    else:
        ct = pd.crosstab(df[factor], df["Heart patient"], normalize="index") * 100
        print(f"\n[{factor}]:\n{ct.round(1)}")

# =============================================================================
# BÖLÜM E — MAKİNE ÖĞRENMESİ MODELLERİ
# =============================================================================
print("\n" + "=" * 65)
print("BÖLÜM E — MAKİNE ÖĞRENMESİ MODELLERİ")
print("=" * 65)

df_ml = df.copy()
le_enc = LabelEncoder()
for col in df_ml.select_dtypes(exclude="number").columns:
    df_ml[col] = le_enc.fit_transform(df_ml[col].astype(str))

X = df_ml.drop("Heart patient", axis=1)
y = df_ml["Heart patient"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Eğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti  : {X_test.shape[0]} örnek")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "Karar Agaci":        DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Lojistik Regresyon":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred),
        "y_prob":    y_prob,
        "model":     model,
    }

    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(f"  Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"  Precision : {results[name]['precision']:.4f}")
    print(f"  Recall    : {results[name]['recall']:.4f}")
    print(f"  F1-Score  : {results[name]['f1']:.4f}")
    print(f"  ROC-AUC   : {results[name]['roc_auc']:.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['Hayır','Evet'])}")

# ── Confusion Matrix Grafikleri ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Hayır", "Evet"], yticklabels=["Hayır", "Evet"],
                linewidths=0.5, cbar=False, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek Değer")
plt.suptitle("Confusion Matrix — Model Karşılaştırması", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_ml_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nConfusion matrix grafiği kaydedildi.")

# ── ROC Eğrisi ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cols = ["#2E86AB", "#C0392B", "#27AE60"]
for (name, r), col in zip(results.items(), cols):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax.plot(fpr, tpr, color=col, linewidth=2.2,
            label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
ax.set_title("ROC Eğrisi — Model Karşılaştırması", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_ml_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("ROC eğrisi grafiği kaydedildi.")

# ── Feature Importance (Random Forest) ───────────────────────────────────
rf_model = results["Random Forest"]["model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
top15 = feat_imp.tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(top15.index, top15.values, color="#2E86AB", edgecolor="white", height=0.7)
ax.set_title("Random Forest — Değişken Önemi (Feature Importance)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Önem Skoru")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, top15.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_ml_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance grafiği kaydedildi.")

# ── Performans Karşılaştırma ──────────────────────────────────────────────
metrics   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
m_labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.25
colors_m = ["#2E86AB", "#C0392B", "#27AE60"]
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, col) in enumerate(zip(results.keys(), colors_m)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=col, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(m_labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Skor")
ax.set_title("Model Performans Karşılaştırması", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig_ml_performans_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Performans karşılaştırma grafiği kaydedildi.")

# =============================================================================
# BÖLÜM F — MODEL YORUMLAMA (Konsola özet)
# =============================================================================
print("\n" + "=" * 65)
print("BÖLÜM F — MODEL YORUMLAMA")
print("=" * 65)

best_auc_model = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\nEn yüksek ROC-AUC'ye sahip model: {best_auc_model}"
      f"  (AUC={results[best_auc_model]['roc_auc']:.4f})")
print("Recall degerlerinin düsüklügü, sinif dengesizligini ve veri kücüklügünü"
      " yansitmaktadır.")

print("\n" + "=" * 65)
print("TÜM ANALİZ TAMAMLANDI — figures/ klasöründe grafikler kayıtlı.")
print("=" * 65)
