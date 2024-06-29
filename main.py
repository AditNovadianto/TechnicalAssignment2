import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Membaca dataset
df = pd.read_csv('ai4i2020.csv')

print(df.head())

# Mengecek nilai yang hilang
print(df.isnull().sum())

# Menghapus baris dengan nilai yang hilang (jika ada)
df = df.dropna()

# Mengecek statistik dasar untuk outlier
print(df.describe())

# Print column names to check for discrepancies
print("\nColumns in the dataset:", df.columns.tolist())

# Mengidentifikasi fitur dan label
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
label = 'Machine failure'

# Print the features to check for discrepancies
print("\nSpecified features:", features)

# Check if specified features exist in the DataFrame
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print("There is missing features in the DataFrame:", missing_features)
else:
    X = df[features]
    y = df[label]

    # Membagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pra-pemrosesan data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']),
            ('cat', OneHotEncoder(), ['Type'])
        ])

    # Membuat pipeline dengan RandomForest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Melatih model
    pipeline.fit(X_train, y_train)

    # Memprediksi dan evaluasi model
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Calculate and print R² score
    r2 = r2_score(y_test, y_pred)
    print("R² score:", r2)

    # Calculate and print RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)

    # Visualisasi confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot()
    plt.show()

    # Visualisasi fitur penting
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = preprocessor.transformers_[0][2] + list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['Type']))
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Plot fitur penting
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.show()
