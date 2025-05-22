# analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def run_analysis():
    # Crear carpeta static si no existe
    os.makedirs("static", exist_ok=True)

    # Cargar y preparar datos
    df = pd.read_csv("Datos_enriquecido_final(1).csv", index_col=0, parse_dates=True).dropna()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.upper())

    target = "PM25"
    targets = ["CO", "PM10"]

    # Correlación
    correlation_matrix = df[[target] + targets].corr()
    print("\n📊 MATRIZ DE CORRELACIÓN:")
    print(correlation_matrix)

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación entre PM25, CO y PM10")
    plt.tight_layout()
    plt.savefig("static/heatmap_correlacion.png")
    plt.close()

    # Dispersión PM25 vs CO/PM10
    for t in targets:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[target], y=df[t], alpha=0.6)
        sns.regplot(x=df[target], y=df[t], scatter=False, color="red")
        plt.xlabel("PM25")
        plt.ylabel(t)
        plt.title(f"Relación entre PM25 y {t}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"static/dispersion_pm25_vs_{t.lower()}.png")
        plt.close()

    # REGRESIÓN MULTIVARIADA
    X = df[[target]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    r2_scores = {}  # Para almacenar los R²

    for t in targets:
        y = df[t]
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores[t] = round(r2, 4)

        # Gráfico real vs predicho
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        sns.lineplot(x=y_test, y=y_test, color="red", label="Ideal")
        plt.xlabel(f"{t} Real")
        plt.ylabel(f"{t} Predicho")
        plt.title(f"Regresión de {t} usando PM25 (R² = {r2:.2f})")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"static/regresion_{t.lower()}.png")
        plt.close()

    # Promedio PM25 por año
    yearly_avg = df.resample("YE").mean()["PM25"]
    plt.figure(figsize=(10, 5))
    yearly_avg.plot(kind="bar", color="orange")
    plt.title("Promedio Anual de PM25")
    plt.xlabel("Año")
    plt.ylabel("PM25")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("static/pm25_anual.png")
    plt.close()

    # Devuelve datos útiles para mostrar en HTML
    return {
        "correlation_matrix": correlation_matrix.round(2).to_dict(),
        "r2_scores": r2_scores,
        "yearly_avg": yearly_avg.round(2).to_dict()
    }
