"""
Synthetic Credit Score Estimator (GAN + Regression)

Pipeline:
1. Load real credit-like data.
2. Train a GAN to generate synthetic tabular samples.
3. Train a regression model on synthetic samples to predict a custom credit score.
4. Use the regressor to score new users with limited / no traditional credit history.

NOTE: This is a demo skeleton. You MUST:
- Replace column names with your actual dataset schema.
- Add proper preprocessing / scaling.
- Carefully validate model fairness and regulatory compliance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------
# 1. Load and preprocess real credit data
# ---------------------------------------------------------------------

REAL_DATA_PATH = "real_credit_data.csv"   # adjust as needed
TARGET_COL = "credit_score"               # custom score you want to model

# Expected example schema (you must align with your real CSV):
# ['annual_income', 'product_price', 'population_demographics',
#  'age_of_phone_number', 'credit_score']

real_data = pd.read_csv(REAL_DATA_PATH)

# Drop rows with missing target
real_data = real_data.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# Split features/target
X_real = real_data.drop(columns=[TARGET_COL])
y_real = real_data[TARGET_COL].astype(float)

# Optional: if categorical columns exist, encode them here.
# For demo, assume all numeric already:
feature_cols = X_real.columns.tolist()

# Scale features (GAN trains better on scaled data)
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real.values)

# ---------------------------------------------------------------------
# 2. Define GAN (generator + discriminator) for tabular data
# ---------------------------------------------------------------------

latent_dim = 32  # noise dimension

def build_generator(latent_dim: int, out_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(out_dim, activation="linear"),  # output = scaled feature vector
        ],
        name="generator",
    )
    return model

def build_discriminator(in_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(in_dim,)),
            layers.Dense(64),
            layers.LeakyReLU(0.2),
            layers.Dense(64),
            layers.LeakyReLU(0.2),
            layers.Dense(1),  # logits: real vs fake
        ],
        name="discriminator",
    )
    return model

n_features = X_real_scaled.shape[1]
generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator(n_features)

# Compile discriminator
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator.compile(
    optimizer=discriminator_optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Build combined GAN (for training generator only)
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
fake_samples = generator(gan_input)
gan_output = discriminator(fake_samples)
gan = keras.Model(gan_input, gan_output, name="gan")

gan_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gan.compile(
    optimizer=gan_optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

# ---------------------------------------------------------------------
# 3. Train GAN
# ---------------------------------------------------------------------

batch_size = 128
epochs = 2000  # adjust based on convergence
real_labels = np.ones((batch_size, 1), dtype=np.float32)
fake_labels = np.zeros((batch_size, 1), dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices(X_real_scaled.astype("float32"))
dataset = dataset.shuffle(buffer_size=len(X_real_scaled)).batch(batch_size, drop_remainder=True)

for epoch in range(epochs):
    for step, real_batch in enumerate(dataset):
        current_bs = real_batch.shape[0]
        # 1. Train discriminator
        noise = tf.random.normal(shape=(current_bs, latent_dim))
        fake_batch = generator(noise, training=True)

        # Combine real + fake
        x_combined = tf.concat([real_batch, fake_batch], axis=0)
        y_combined = tf.concat(
            [tf.ones((current_bs, 1)), tf.zeros((current_bs, 1))],
            axis=0,
        )

        # Train discriminator
        d_loss, d_acc = discriminator.train_on_batch(x_combined, y_combined)

        # 2. Train generator (via GAN)
        noise = tf.random.normal(shape=(current_bs, latent_dim))
        # Generator wants discriminator to output 1 (real) for fake samples
        g_loss = gan.train_on_batch(noise, tf.ones((current_bs, 1)))

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs} | D_loss={d_loss:.4f} | D_acc={d_acc:.3f} | G_loss={g_loss:.4f}")

print("GAN training complete.")

# ---------------------------------------------------------------------
# 4. Generate synthetic credit data
# ---------------------------------------------------------------------

n_synthetic = 10000  # how many synthetic rows you want
noise = tf.random.normal(shape=(n_synthetic, latent_dim))
synthetic_features_scaled = generator.predict(noise, verbose=0)
synthetic_features = scaler.transform(X_real)  # NOTE:
# ^ Above line is WRONG if used as-is; we want inverse_transform of fake data.
# Correct:
synthetic_features = scaler.inverse_transform(synthetic_features_scaled)

synthetic_df = pd.DataFrame(synthetic_features, columns=feature_cols)

# For demo: attach a synthetic credit_score by training a regressor on real data
# and using it to label the synthetic features.
base_reg = LinearRegression().fit(X_real.values, y_real.values)
synthetic_scores = base_reg.predict(synthetic_df.values)
synthetic_df[TARGET_COL] = synthetic_scores

# ---------------------------------------------------------------------
# 5. Train regression model on synthetic dataset
# ---------------------------------------------------------------------

X_syn = synthetic_df.drop(columns=[TARGET_COL])
y_syn = synthetic_df[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(
    X_syn, y_syn, test_size=0.2, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Synthetic-data regressor R^2 on held-out synthetic val set:",
      regressor.score(X_val, y_val))

# ---------------------------------------------------------------------
# 6. Predict custom credit score for a new individual
# ---------------------------------------------------------------------

def predict_custom_score(
    annual_income: float,
    product_price: float,
    population_demographics: float,
    age_of_phone_number: float,
) -> float:
    """
    Predicts custom credit score based on synthetic-data-trained regressor.
    All inputs must match the scale / meaning of your original dataset.
    """
    new_row = pd.DataFrame(
        [[annual_income, product_price, population_demographics, age_of_phone_number]],
        columns=feature_cols,
    )
    return float(regressor.predict(new_row.values)[0])

# Example call (replace with real values)
if __name__ == "__main__":
    example_score = predict_custom_score(
        annual_income=65000,
        product_price=1200,
        population_demographics=0.6,  # e.g., encoded index / factor
        age_of_phone_number=36,       # months
    )
    print("Predicted custom credit score:", example_score)
