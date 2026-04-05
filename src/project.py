import os
from preprocessing import load_data, split_data, batch_generator
from model import build_model
from evaluation import plot_history

def main():

    data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    batch_size = 32
    epochs = 10

    # Load and split data
    image_paths, steering_angles = load_data(data_path)
    X_train, X_valid, y_train, y_valid = split_data(image_paths, steering_angles)

    print('Total samples:', len(image_paths))
    print('Training samples:', len(X_train))
    print('Validation samples:', len(X_valid))

    # Create generators
    train_generator = batch_generator(data_path, X_train, y_train, batch_size, True)
    valid_generator = batch_generator(data_path, X_valid, y_valid, batch_size, False)

    # Build model
    model = build_model()

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=valid_generator,
        validation_steps=len(X_valid) // batch_size,
        epochs=epochs,
        verbose=1
    )

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.h5')
    model.save(model_path)
    print('Model saved as model.h5')

    plot_history(history)

    return history


if __name__ == '__main__':
    main()