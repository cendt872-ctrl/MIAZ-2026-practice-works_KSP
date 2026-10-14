"""
Практична робота № 8 «Класифікація». Варіант 7.

Номер за журналом: 7
  n_samples = 1000 * 7 = 7000
  n_classes = 7 + 2    = 9

Етапи:
  1. Генерація даних (sklearn.datasets.make_classification).
  2. Підготовка даних: стратифікований поділ train/val/test, StandardScaler.
  3. Проста повнозв'язна нейронна мережа Keras із Softmax-виходом.
  4. Навчання з анімованим відображенням метрик після кожної епохи.
  5. Оцінювання на тестовій вибірці: accuracy, classification_report,
     матриця плутанини.

Усі результати зберігаються в папку results поруч із main.py.
"""

import os

# Фіксуємо детермінованість операцій TensorFlow ДО імпорту tensorflow.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # приховати службові INFO-повідомлення

import random
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Параметри варіанта та технічні параметри
# ---------------------------------------------------------------------------
JOURNAL_NUMBER = 7
N_SAMPLES = 1000 * JOURNAL_NUMBER  # 7000
N_CLASSES = JOURNAL_NUMBER + 2     # 9

# Параметри генерації (технічне рішення, умова їх не визначає)
N_FEATURES = 20
N_INFORMATIVE = 10
N_REDUNDANT = 5
N_REPEATED = 0
N_CLUSTERS_PER_CLASS = 2

SEED = 42

# Частки вибірок: 70 % / 15 % / 15 %
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Параметри навчання
EPOCHS = 40  # після ~35 епох val_loss перестає знижуватися (визначено за валідаційною вибіркою)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

CLASS_NAMES = [f"Клас {i}" for i in range(N_CLASSES)]


def set_seed(seed: int) -> None:
    """Фіксація seed для Python, NumPy і TensorFlow/Keras."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)  # встановлює seed для random, numpy і tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def is_interactive_backend() -> bool:
    """Чи може Matplotlib показувати вікна (у терміналі VS Code на Windows — TkAgg)."""
    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}
    return backend not in non_interactive and "inline" not in backend


# ---------------------------------------------------------------------------
# 1. Генерація даних
# ---------------------------------------------------------------------------
def generate_data():
    # Обмеження make_classification
    assert N_CLASSES * N_CLUSTERS_PER_CLASS <= 2 ** N_INFORMATIVE, (
        "Порушено обмеження n_classes * n_clusters_per_class <= 2**n_informative"
    )
    assert N_INFORMATIVE + N_REDUNDANT + N_REPEATED <= N_FEATURES

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=N_REPEATED,
        n_classes=N_CLASSES,
        n_clusters_per_class=N_CLUSTERS_PER_CLASS,
        random_state=SEED,
    )

    print("=" * 70)
    print("1. ГЕНЕРАЦІЯ ДАНИХ")
    print("=" * 70)
    print(f"n_samples={N_SAMPLES}, n_classes={N_CLASSES}, n_features={N_FEATURES}, "
          f"n_informative={N_INFORMATIVE}, n_redundant={N_REDUNDANT}, "
          f"n_clusters_per_class={N_CLUSTERS_PER_CLASS}")
    print(f"Перевірка обмеження: {N_CLASSES} × {N_CLUSTERS_PER_CLASS} = "
          f"{N_CLASSES * N_CLUSTERS_PER_CLASS} <= 2**{N_INFORMATIVE} = {2 ** N_INFORMATIVE}")
    print(f"Розмір X: {X.shape}, розмір y: {y.shape}")

    classes, counts = np.unique(y, return_counts=True)
    print(f"Кількість унікальних класів: {len(classes)}")
    print("Кількість зразків кожного класу:")
    for c, n in zip(classes, counts):
        print(f"  клас {c}: {n}")
    return X, y


def plot_class_distribution(y):
    classes, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([str(c) for c in classes], counts, color="#4C72B0")
    ax.bar_label(bars, padding=2)
    ax.set_title(f"Розподіл зразків за класами (усього {len(y)})")
    ax.set_xlabel("Клас")
    ax.set_ylabel("Кількість зразків")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = RESULTS_DIR / "class_distribution.png"
    fig.savefig(path, dpi=120)
    print(f"Збережено: {path}")
    return fig


# ---------------------------------------------------------------------------
# 2. Підготовка даних
# ---------------------------------------------------------------------------
def prepare_data(X, y):
    # Спочатку відокремлюємо тестову вибірку, потім із решти — валідаційну.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    val_fraction_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp, random_state=SEED
    )

    # StandardScaler навчається ЛИШЕ на train; val і test — тільки transform.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("2. ПІДГОТОВКА ДАНИХ")
    print("=" * 70)
    for name, Xs, ys in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
        counts = np.bincount(ys, minlength=N_CLASSES)
        print(f"{name:>5}: X={Xs.shape}, y={ys.shape}, зразків за класами: {counts.tolist()}")
    print("StandardScaler: fit лише на train; середні ознак train після масштабування "
          f"≈ {np.abs(X_train.mean(axis=0)).max():.2e} (макс. |mean|), "
          f"стд ≈ {X_train.std(axis=0).mean():.3f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 3. Модель
# ---------------------------------------------------------------------------
def build_model(n_features: int, n_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="simple_softmax_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Анімація навчання (Keras callback)
# ---------------------------------------------------------------------------
class LiveTrainingPlot(keras.callbacks.Callback):
    """Після кожної епохи оновлює графіки accuracy і loss фактичними
    значеннями з logs та запам'ятовує їх для збереження GIF-анімації."""

    def __init__(self, epochs: int, show: bool):
        super().__init__()
        self.epochs = epochs
        self.show = show
        self.history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        if self.show:
            plt.ion()
        self.fig, (self.ax_acc, self.ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Навчання моделі (оновлення після кожної епохи)")
        (self.l_acc,) = self.ax_acc.plot([], [], "o-", ms=3, label="train accuracy")
        (self.l_vacc,) = self.ax_acc.plot([], [], "s-", ms=3, label="val accuracy")
        (self.l_loss,) = self.ax_loss.plot([], [], "o-", ms=3, label="train loss")
        (self.l_vloss,) = self.ax_loss.plot([], [], "s-", ms=3, label="val loss")
        for ax, title, ylabel in (
            (self.ax_acc, "Accuracy", "accuracy"),
            (self.ax_loss, "Loss (sparse categorical crossentropy)", "loss"),
        ):
            ax.set_title(title)
            ax.set_xlabel("Епоха")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.5, epochs + 0.5)
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
        self.fig.tight_layout()

    def _redraw(self, n: int):
        """Оновлює лінії графіка до епохи n (1..n)."""
        ep = np.arange(1, n + 1)
        h = self.history
        self.l_acc.set_data(ep, h["accuracy"][:n])
        self.l_vacc.set_data(ep, h["val_accuracy"][:n])
        self.l_loss.set_data(ep, h["loss"][:n])
        self.l_vloss.set_data(ep, h["val_loss"][:n])

        acc_vals = h["accuracy"][:n] + h["val_accuracy"][:n]
        loss_vals = h["loss"][:n] + h["val_loss"][:n]
        self.ax_acc.set_ylim(max(0.0, min(acc_vals) - 0.05), min(1.0, max(acc_vals) + 0.05))
        self.ax_loss.set_ylim(max(0.0, min(loss_vals) - 0.1), max(loss_vals) + 0.1)
        self.fig.suptitle(
            f"Епоха {n}/{self.epochs}: "
            f"acc={h['accuracy'][n - 1]:.4f}, val_acc={h['val_accuracy'][n - 1]:.4f}, "
            f"loss={h['loss'][n - 1]:.4f}, val_loss={h['val_loss'][n - 1]:.4f}"
        )
        return self.l_acc, self.l_vacc, self.l_loss, self.l_vloss

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in self.history:
            self.history[key].append(float(logs[key]))
        self._redraw(epoch + 1)
        if self.show:
            self.fig.canvas.draw_idle()
            plt.pause(0.01)  # дає вікну перемалюватися під час навчання

    def save_gif(self, path: Path, fps: int = 6):
        """Зберігає анімацію з фактичних метрик (PillowWriter — без FFmpeg)."""
        n_frames = len(self.history["loss"])
        anim = FuncAnimation(
            self.fig, lambda i: self._redraw(i + 1), frames=n_frames, blit=False, repeat=False
        )
        anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
        self._redraw(n_frames)  # повертаємо фінальний стан у вікно
        print(f"Збережено: {path}")


# ---------------------------------------------------------------------------
# 5. Підсумкові графіки
# ---------------------------------------------------------------------------
def plot_final_history(history: dict):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["accuracy"], label="train accuracy")
    ax1.plot(epochs, history["val_accuracy"], label="val accuracy")
    ax1.set_title("Підсумкова accuracy")
    ax1.set_xlabel("Епоха")
    ax1.set_ylabel("accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(epochs, history["loss"], label="train loss")
    ax2.plot(epochs, history["val_loss"], label="val loss")
    ax2.set_title("Підсумкова loss")
    ax2.set_xlabel("Епоха")
    ax2.set_ylabel("loss")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    path = RESULTS_DIR / "training_history.png"
    fig.savefig(path, dpi=120)
    print(f"Збережено: {path}")
    return fig


def plot_confusion_matrix(cm: np.ndarray, test_acc: float):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(N_CLASSES)
    ax.set_xticks(ticks, labels=[str(i) for i in ticks])
    ax.set_yticks(ticks, labels=[str(i) for i in ticks])
    ax.set_xlabel("Прогнозований клас")
    ax.set_ylabel("Істинний клас")
    ax.set_title(f"Матриця плутанини (тестова вибірка), accuracy = {test_acc:.4f}")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=10)
    fig.tight_layout()
    path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=120)
    print(f"Збережено: {path}")
    return fig


# ---------------------------------------------------------------------------
# Головна функція
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    show = is_interactive_backend()
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}, "
          f"backend Matplotlib: {matplotlib.get_backend()}")

    # 1. Дані
    X, y = generate_data()
    fig_dist = plot_class_distribution(y)
    if show:
        plt.show(block=False)
        plt.pause(0.5)

    # 2. Підготовка
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)

    # 3. Модель
    print("\n" + "=" * 70)
    print("3. МОДЕЛЬ")
    print("=" * 70)
    model = build_model(X_train.shape[1], N_CLASSES)
    model.summary()

    # 4. Навчання: використовуються лише train і val; test не бере участі.
    print("\n" + "=" * 70)
    print("4. НАВЧАННЯ")
    print("=" * 70)
    live = LiveTrainingPlot(EPOCHS, show)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[live],
        verbose=2,
    )
    live.save_gif(RESULTS_DIR / "training_animation.gif")
    fig_hist = plot_final_history(live.history)

    # 5. Оцінювання на тестовій вибірці
    print("\n" + "=" * 70)
    print("5. ОЦІНЮВАННЯ НА ТЕСТОВІЙ ВИБІРЦІ")
    print("=" * 70)
    probs = model.predict(X_test, verbose=0)          # ймовірності Softmax, форма (n, 9)
    y_pred = np.argmax(probs, axis=1)                  # клас = argmax ймовірностей
    print(f"Форма прогнозів Softmax: {probs.shape}; сума ймовірностей у рядку ≈ "
          f"{probs.sum(axis=1).mean():.4f}")

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}")

    report = classification_report(
        y_test, y_pred, labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))
    print("Матриця плутанини (рядки — істинні класи, стовпці — прогнозовані):")
    print(cm)
    fig_cm = plot_confusion_matrix(cm, test_acc)

    # Текстовий звіт у файл
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Варіант {JOURNAL_NUMBER}: n_samples={N_SAMPLES}, n_classes={N_CLASSES}\n")
        f.write(f"Розміри: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nМатриця плутанини:\n")
        f.write(np.array2string(cm))
        f.write("\n")
    print(f"Збережено: {report_path}")

    print(f"\nУсі результати збережено в папці: {RESULTS_DIR}")
    if show:
        plt.ioff()
        print("Закрийте вікна графіків, щоб завершити програму.")
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
