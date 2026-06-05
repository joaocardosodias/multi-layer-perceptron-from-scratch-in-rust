fn main() {
    println!("cargo:rustc-link-lib=mkl_rt");
    println!("cargo:rustc-link-search=native=/opt/intel/oneapi/mkl/latest/lib/intel64/");
}
