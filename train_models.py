"""Trains and compares the models, saves reports and the model used by the app."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from config import *

np.random.seed(RANDOM_STATE)
CAT, NUM = CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def make_pipe(model, scale=False):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler() if scale else "passthrough", NUM),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_tr, X_te, y_tr, y_te, _, rule_te = train_test_split(
        X, y, df[RULE_TARGET], test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "majority_baseline": make_pipe(DummyClassifier(strategy="most_frequent")),
        "decision_tree": GridSearchCV(
            make_pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            {"clf__max_depth": [4, 6, 8, 10, 12, 15, None]}, cv=5, scoring="f1_macro"),
        "random_forest": make_pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        "logistic_regression": make_pipe(LogisticRegression(max_iter=2000), scale=True),
    }

    rows, fitted, preds = [], {}, {}
    for name, m in models.items():
        if name == "decision_tree":
            m.fit(X_tr, y_tr)
            cv = m.best_score_
            print("Tuned decision tree max_depth:", m.best_params_["clf__max_depth"])
        else:
            cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="f1_macro").mean()
            m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        fitted[name], preds[name] = m, p
        rows.append({"model": name, "cv_macro_f1": cv,
                     "test_accuracy": accuracy_score(y_te, p),
                     "test_macro_f1": f1_score(y_te, p, average="macro"),
                     "agreement_with_rules": float((p == rule_te.values).mean())})
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    # Confusion matrix for the model with the best cv_macro_f1
    best = comp.loc[comp.cv_macro_f1.idxmax(), "model"]
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_te, preds[best], labels=CLASS_LABELS,
        display_labels=[CLASSES[c][0] for c in CLASS_LABELS],
        xticks_rotation=45, ax=ax, colorbar=False, cmap="Blues")
    ax.set_xlabel("Predicted process"); ax.set_ylabel("True (noisy-label) process")
    ax.set_title(f"Confusion matrix on test set: {best}")
    fig.tight_layout(); fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    # Random forest importances, one-hot columns summed back per field
    rf = fitted["random_forest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    per_field = {}
    for f in FEATURE_COLUMNS:
        per_field[f] = imp[[n for n in names if n.split("__", 1)[1] == f
                            or n.split("__", 1)[1].startswith(f + "_")]].sum()
    per_field = pd.Series(per_field).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(per_field.index, per_field.values, color="#0d6efd")
    ax.set_xlabel("Random forest feature importance"); ax.set_ylabel("Input field")
    ax.set_title("Which inputs matter most")
    fig.tight_layout(); fig.savefig(ARTIFACT_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    # Readable tree rules
    tree = fitted["decision_tree"].best_estimator_
    tnames = list(tree.named_steps["pre"].get_feature_names_out())
    (ARTIFACT_DIR / "decision_tree_rules.txt").write_text(
        export_text(tree.named_steps["clf"], feature_names=tnames, max_depth=50))

    # Logistic coefficients
    lr = fitted["logistic_regression"]
    pd.DataFrame(lr.named_steps["clf"].coef_, index=lr.named_steps["clf"].classes_,
                 columns=lr.named_steps["pre"].get_feature_names_out()
                 ).rename_axis("class").to_csv(ARTIFACT_DIR / "logistic_coefficients.csv")

    # Refit the app model on all rows
    final = models[MODEL_FOR_APP]
    final = final.best_estimator_ if hasattr(final, "best_estimator_") else final
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)

    print("\n" + comp.to_string(index=False))
    print(f"\nBest cv_macro_f1 model: {best}. Saved app model: {MODEL_FOR_APP} -> {MODEL_PATH}")
    acc = comp.set_index("model").test_accuracy
    if not 0.84 <= acc["random_forest"] <= 0.95:
        print("WARNING: random forest accuracy outside 0.84-0.95; check data and split.")


if __name__ == "__main__":
    main()
