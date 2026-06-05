use std::fs::File;
use std::io::Read;


pub fn load_images(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).expect("Failed to open images");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("Falha ao ler o arquivo");
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "Invalid images magic number");
    let num_imagens = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;

    let offset=16;
    let mut images=Vec::with_capacity(num_imagens);
    for i in 0..num_imagens{
        let start=offset +i *rows*cols;
        let mut image=Vec::with_capacity(rows*cols);
        for j in 0..(rows*cols){
            image.push(buf[start+j] as f32/255.0);
            
        }
        images.push(image);
    }
    images
}
pub fn load_labels(path:&str)->Vec<usize>{
    let mut file=File::open(path).expect("Failed to open labels");
    let mut buf=Vec::new();
    file.read_to_end(&mut buf).expect("Falha ao ler o arquivo");
    let magic=u32::from_be_bytes([buf[0],buf[1],buf[2],buf[3]]) as usize;
    assert_eq!(magic,2049,"Invalid labels magic number");
    let num_labels=u32::from_be_bytes([buf[4],buf[5],buf[6],buf[7]]) as usize;
    let mut labels=Vec::with_capacity(num_labels);
    for i in 0..num_labels{
        labels.push(buf[8+i] as usize);
        
    }
    labels
}
