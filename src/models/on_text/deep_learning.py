import pandas as pd
from pathlib import Path
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Embedding


def define_model(text_vectorizer, num_classes=27):
    '''
    Args:
        text_vectorizer: Une couche TextVectorization déjà adaptée.
        num_classes: Nombre de classes de la variable cible.
    '''
    # --- Architecture de la branche Texte ---
    text_input = keras.Input(shape=(1,), dtype=tf.string, name='text_input')
    vectorized_text = text_vectorizer(text_input)
    text_embedding = layers.Embedding(
        input_dim=len(text_vectorizer.get_vocabulary()),
        output_dim=128,
        name='text_embedding'
    )(vectorized_text)
    x = layers.Conv1D(128, 5, activation='relu')(text_embedding)
    x = layers.GlobalMaxPooling1D()(x) # Prend l'information la plus importante de la séquence
    text_features = layers.Dense(64, activation='relu')(x)

    # Classification

    x = layers.Dropout(0.5)(text_features)

    output = layers.Dense(
        num_classes, activation='softmax', name='output',
        kernel_regularizer=regularizers.l2(0.001),
    )(x)

    # Model

    model = keras.Model(
        inputs=[text_input],
        outputs=output
    )

    return model
