use anyhow::Result;
use brush_train::eval::EvalSample;
use burn::prelude::Backend;
use std::path::Path;

#[allow(unused)]
pub async fn eval_save_to_disk<B: Backend>(sample: &EvalSample<B>, path: &Path) -> Result<()> {
    // TODO: Maybe figure out how to do this on WASM.
    #[cfg(not(target_family = "wasm"))]
    {
        use image::{Rgb32FImage, RgbaImage};
        log::info!("Saving eval image to disk.");
        let img = sample.rendered.clone();
        let [h, w, c] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = sample
            .rendered
            .clone()
            .into_data_async()
            .await?
            .into_vec::<f32>()?;

        let img: image::DynamicImage = if sample.mode == brush_render::RenderMode::Depth {
            let data_u8: Vec<u8> = data.iter().map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect();
            RgbaImage::from_raw(w as u32, h as u32, data_u8)
                .expect("Failed to create image from tensor")
                .into()
        } else {
            // For standard and index rendering, we only care about the RGB channels.
            // But the tensor might have 4 channels because inference sets 4 channels for `bwd_info: true`.
            let mut rgb_data = Vec::with_capacity(w * h * 3);
            for i in 0..(w * h) {
                rgb_data.push(data[i * c + 0]);
                rgb_data.push(data[i * c + 1]);
                rgb_data.push(data[i * c + 2]);
            }
            let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, rgb_data)
                .expect("Failed to create image from tensor")
                .into();
            img.into_rgb8().into()
        };

        let parent = path.parent().expect("Eval must have a filename");
        tokio::fs::create_dir_all(parent).await?;
        log::info!("Saving eval view to {path:?}");
        img.save(path)?;
    }
    Ok(())
}
