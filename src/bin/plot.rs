use plotters::prelude::*;
use std::error::Error;

struct TrainingData {
    epoch: Vec<u32>,
    test_acc: Vec<f32>,
    test_loss: Vec<f32>,
    label: String,
}

impl TrainingData {
    fn from_csv(path: &str, label: &str) -> Result<Self, Box<dyn Error>> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut data = Self {
            epoch: vec![],
            test_acc: vec![],
            test_loss: vec![],
            label: label.to_string(),
        };
        
        for result in reader.records() {
            let record = result?;
            data.epoch.push(record[0].parse()?);
            // Formato do CSV: epoch, train_loss, train_acc, test_acc, test_loss
            data.test_acc.push(record[3].parse()?);
            data.test_loss.push(record[4].parse()?);
        }
        
        Ok(data)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Uso: cargo run --bin plot -- <arquivo1.csv> <nome1> <arquivo2.csv> <nome2>");
        println!("Exemplo: cargo run --bin plot -- run1.csv \"Rede Grande\" run2.csv \"Rede Pequena\"");
        return Ok(());
    }
    
    let mut datasets = vec![];
    let mut i = 1;
    while i + 1 < args.len() {
        datasets.push(TrainingData::from_csv(&args[i], &args[i+1])?);
        i += 2;
    }
    
    // Criar pasta experiments/output se não existir
    let output_dir = "experiments/output";
    std::fs::create_dir_all(output_dir)?;
    
    let output_path = format!("{}/comparison_plot.png", output_dir);
    let root_area = BitMapBackend::new(&output_path, (1200, 500)).into_drawing_area();
    let white_color = WHITE;
    root_area.fill(&white_color)?;
    
    let (left_area, right_area) = root_area.split_horizontally(600);
    
    // Gráfico de Acurácia
    let max_epoch = datasets.iter().flat_map(|d| d.epoch.iter()).max().copied().unwrap_or(300) + 10;
    let mut acc_chart = ChartBuilder::on(&left_area)
        .caption("Acurácia ao longo do Treinamento", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0u32..max_epoch, 90.0f32..100.0f32)?;
        
    acc_chart.configure_mesh().draw()?;
    
    let colors: Vec<RGBColor> = vec![BLUE, RED, GREEN, MAGENTA];
    for (idx, ds) in datasets.iter().enumerate() {
        let color = colors[idx % colors.len()];
        acc_chart.draw_series(LineSeries::new(
            ds.epoch.iter().zip(ds.test_acc.iter()).map(|(&x, &y)| (x, y)),
            color,
        ))?.label(&ds.label)
          .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    acc_chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft).draw()?;
    
    // Gráfico de Loss
    let mut loss_chart = ChartBuilder::on(&right_area)
        .caption("Loss ao longo do Treinamento", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0u32..max_epoch, 0.0f32..2.0f32)?;
        
    loss_chart.configure_mesh().draw()?;
    
    for (idx, ds) in datasets.iter().enumerate() {
        let color = colors[idx % colors.len()];
        loss_chart.draw_series(LineSeries::new(
            ds.epoch.iter().zip(ds.test_loss.iter()).map(|(&x, &y)| (x, y)),
            color,
        ))?.label(&ds.label)
          .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    loss_chart.configure_series_labels().position(SeriesLabelPosition::UpperRight).draw()?;
    
    root_area.present()?;
    println!("✅ Gráfico salvo em '{}/comparison_plot.png'", output_dir);
    
    Ok(())
}
