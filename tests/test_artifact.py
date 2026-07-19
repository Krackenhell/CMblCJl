from vivatrace.artifact import inspect_ml_artifact


def test_detects_scaler_leakage_and_missing_seed() -> None:
    code = """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=.2)
print(accuracy_score(y_test, pred))
"""
    skills = {finding.skill_id for finding in inspect_ml_artifact(code)}
    assert {"data_leakage", "metrics", "reproducibility"}.issubset(skills)


def test_clean_pipeline_does_not_trigger_scaler_leakage() -> None:
    code = """
X_train, X_test = train_test_split(X, test_size=.2, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
"""
    skills = {finding.skill_id for finding in inspect_ml_artifact(code)}
    assert "data_leakage" not in skills
