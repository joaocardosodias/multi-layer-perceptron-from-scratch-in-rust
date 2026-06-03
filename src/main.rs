use crate::data::load_images;

mod activations;
mod data;
mod losses;
mod network;
mod optimizers;
fn main() {
    let path = "/home/cardoso/GitHub/multi-layer-perceptron-from-scratch-in-rust/src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    let images = load_images(path);
    println!("{:?}", images[1]);
}

