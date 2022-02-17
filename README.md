# ml-localisation
A simple attempt at learning how to use tensorflow for localisation.

It is recommended to have CUDA enabled, and significant system memory available for training.

# Instructions
```bash
mkdir validation
mkdir training
cargo run --release #generate the validation and training data
python model.py #train the model, may take some time
```