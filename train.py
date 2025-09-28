import os
import sys
import argparse
import pickle
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras import layers, Model, losses, optimizers
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import activations, initializers

def cosine_adj_dense(X: np.ndarray, topk: int) -> np.ndarray:
    normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = np.dot(normed, normed.T)
    N = sim.shape[0]
    A = np.zeros_like(sim, dtype=np.float32)
    if topk <= 0:
        A = sim.copy()
    else:
        k = min(topk + 1, sim.shape[1])
        idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        for i in range(N):
            neighbors = [j for j in idx[i] if j != i]
            if len(neighbors) == 0:
                continue
            A[i, neighbors] = sim[i, neighbors]
    A = A + A.T
    A = A + np.eye(N, dtype=np.float32)
    rowsum = A.sum(axis=1, keepdims=True) + 1e-12
    A = A / rowsum
    return A.astype(np.float32)



@tf.keras.utils.register_keras_serializable(package="Custom", name="DenseGCNLayer")
class DenseGCNLayer(layers.Layer):
    def __init__(self,
                 out_features,
                 use_bias=True,
                 activation=None,
                 dropout=0.0,
                 kernel_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self._activation_arg = activations.serialize(activation) if activation is not None else None
        self.activation = activations.get(self._activation_arg) if self._activation_arg is not None else None
        self.dropout_rate = float(dropout)
        self._kernel_init_arg = (initializers.serialize(kernel_initializer)
                                 if kernel_initializer is not None else initializers.serialize(GlorotUniform()))
        self.kernel_initializer = initializers.get(self._kernel_init_arg)

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        self.W = self.add_weight(
            name="W",
            shape=(in_features, self.out_features),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.out_features,),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.b = None
        super().build(input_shape)

    def call(self, X, A=None, training=False):
        H = tf.matmul(X, self.W)
        if A is not None:
            H = tf.matmul(A, H)
        if self.b is not None:
            H = H + self.b
        if training and self.dropout_rate and self.dropout_rate > 0.0:
            H = tf.nn.dropout(H, rate=self.dropout_rate)
        if self.activation is not None:
            return self.activation(H)
        return H

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "use_bias": self.use_bias,
            "activation": self._activation_arg,
            "dropout": self.dropout_rate,
            "kernel_initializer": self._kernel_init_arg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = activations.deserialize(config.pop("activation", None))
        kernel_init = initializers.deserialize(config.pop("kernel_initializer", None))
        return cls(activation=activation, kernel_initializer=kernel_init, **config)


def build_igcn_model(in_features_list: List[int],
                     hid_size: int = 64,
                     n_classes: int = 2,
                     dropout_rate: float = 0.5) -> tf.keras.Model:
    AX_inputs = [layers.Input(shape=(in_features_list[i],), dtype=tf.float32, name=f"AX_{i}")
                 for i in range(len(in_features_list))]

    per_embs = []
    att_scores = []
    for i, AX_in in enumerate(AX_inputs):
        x = DenseGCNLayer(hid_size, activation=relu, dropout=dropout_rate, name=f"gcn_{i}")(AX_in)
        per_embs.append(x)
        s = layers.Dense(1, use_bias=False, name=f"att_fc_{i}")(x)
        att_scores.append(s)

    stack_scores = layers.Concatenate(axis=1, name="stack_scores")(att_scores)
    att_weights = layers.Activation('softmax', name='att_weights')(stack_scores)

    weighted_embs = []
    for i, emb in enumerate(per_embs):
        def slice_weight(t, idx=i):
            return tf.expand_dims(t[:, idx], axis=1)
        w_i = layers.Lambda(slice_weight, name=f"extract_w_{i}")(att_weights)  # (N,1)
        weighted_embs.append(layers.Multiply(name=f"weighted_emb_{i}")([emb, w_i]))

    fused = layers.Add(name="fused")(weighted_embs)
    out = Dense(n_classes, name="out_dense")(fused)

    model = Model(inputs=AX_inputs, outputs=[out, att_weights], name="iGCN_like")
    return model

def load_features_and_edges(dataset_dir: str = "dataset", edge_dir: str = "data/sample_data"):
    features = []
    for i in range(3):
        csv_file = os.path.join(dataset_dir, f"{i+1}_.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"{csv_file} not found")
        df = pd.read_csv(csv_file, header=None).values.astype(np.float32)
        features.append(df)
    labels_file = os.path.join(dataset_dir, "labels_.csv")
    labels = pd.read_csv(labels_file, header=None).iloc[:, 0].values.astype(np.int32)
    return features, labels


def prepare_AXs(features: List[np.ndarray], topk_frac=0.05):
    N = features[0].shape[0]
    AXs = []
    As = []
    for X in features:
        topk = max(1, int(X.shape[0] * topk_frac))
        A = cosine_adj_dense(X, topk=topk)
        AX = A @ X
        As.append(A.astype(np.float32))
        AXs.append(AX.astype(np.float32))
    return AXs, As


def train_and_evaluate(dataset_dir="dataset",
                       edge_dir="data/sample_data",
                       hid_size=64,
                       lr=0.005,
                       epochs=600,
                       folds=5,
                       batch_size=None):
    features, labels = load_features_and_edges(dataset_dir, edge_dir)
    num_classes = len(np.unique(labels))
    in_feats = [f.shape[1] for f in features]
    N = features[0].shape[0]

    model = build_igcn_model(in_features_list=in_feats, hid_size=hid_size, n_classes=num_classes, dropout_rate=0.5)
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = {'acc': [], 'wf1': [], 'mf1': [], 'mcc': []}
    all_weights = []
    run_times = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features[0], labels)):
        start_time = time.time()
        AXs, As = prepare_AXs(features, topk_frac=0.05)

        model_inputs = {f"AX_{i}": AXs[i] for i in range(len(AXs))}
        stacked_placeholder = np.zeros((N, sum(in_feats)), dtype=np.float32)
        model_inputs["stacked_X"] = stacked_placeholder

        AX_tensors = [tf.convert_to_tensor(AXs[i]) for i in range(len(AXs))]

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                logits, att_w = model(AX_tensors, training=True)
                train_logits = tf.gather(logits, train_idx, axis=0)
                train_labels = tf.gather(labels, train_idx, axis=0)
                loss = loss_fn(train_labels, train_logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        logits, att_w = model(AX_tensors, training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        gt = labels[test_idx]
        pr = preds[test_idx]

        metrics['acc'].append(accuracy_score(gt, pr))
        metrics['wf1'].append(f1_score(gt, pr, average='weighted'))
        metrics['mf1'].append(f1_score(gt, pr, average='macro'))
        metrics['mcc'].append(matthews_corrcoef(gt, pr))

        if fold == folds - 1:
            all_weights = att_w.numpy()
            final_preds = preds
            final_test_idx = test_idx

        end_time = time.time()
        run_times.append(end_time - start_time)
        print(f"Fold {fold} finished in {end_time-start_time:.2f}s")

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    print(f"Avg runtime per fold: {np.mean(run_times):.3f}s")
    return model, all_weights, labels, final_preds, final_test_idx, run_times


def export_to_tflite(keras_model, out_tflite_path="igcn_dynamic.tflite", quantize_type="dynamic"):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Wrote", out_tflite_path)
    return True


def main_cli():

    model, all_weights, labels, preds, test_idx, run_times = train_and_evaluate(
        dataset_dir="dataset",
        edge_dir="data/sample_data",
        hid_size=64,
        epochs=600,
        folds=5
    )

    model.save("igcn_saved_model.h5", include_optimizer=False)


    ok = export_to_tflite(model, out_tflite_path="igcn_dynamic.tflite")


if __name__ == "__main__":
    main_cli()
