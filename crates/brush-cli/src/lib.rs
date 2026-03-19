#![recursion_limit = "256"]

use brush_process::{config::ProcessArgs, message::ProcessMessage};
use brush_vfs::DataSource;
use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use std::time::Duration;
use tokio_stream::{Stream, StreamExt};
use tracing::trace_span;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub process: ProcessArgs,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

pub async fn process_ui(
    stream: impl Stream<Item = anyhow::Result<ProcessMessage>>,
    process_args: ProcessArgs,
) -> Result<(), anyhow::Error> {
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        log::info!("ℹ️  running in debug mode, compile with --release for best performance");
    }

    let mut duration = Duration::from_secs(0);

    let mut stream = std::pin::pin!(stream);
    while let Some(msg) = stream.next().await {
        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                log::error!("❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewSource => {
                log::info!("Starting process...");
            }
            ProcessMessage::StartLoading { training } => {
                if !training {
                    log::error!("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                log::info!("Loading data...");
            }
            ProcessMessage::ViewSplats { .. } => {
            }
            ProcessMessage::Dataset { dataset } => {
                let train_views = dataset.train.views.len();
                let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                log::info!("Loaded dataset with {train_views} training, {eval_views} eval views",);
                if let Some(val) = dataset.eval.as_ref() {
                    log::info!(
                        "evaluating {} views every {} steps",
                        val.views.len(),
                        process_args.process_config.eval_every,
                    );
                }
            }
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
            }
            ProcessMessage::TrainStep {
                iter,
                total_elapsed,
                ..
            } => {
                if iter % 100 == 0 {
                    log::info!("Training step {iter}/{}", process_args.train_config.total_steps);
                }
                duration = total_elapsed;
            }
            ProcessMessage::RefineStep {
                cur_splat_count,
                iter,
                ..
            } => {
                log::info!("Refine iter {iter}, {cur_splat_count} splats.");
            }
            ProcessMessage::EvalResult {
                iter,
                avg_psnr,
                avg_ssim,
            } => {
                log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
            }
            ProcessMessage::DoneTraining => {
                log::info!("Done training.");
            }
        }
    }

    let duration_secs = Duration::from_secs(duration.as_secs());
    log::info!(
        "Training took {}",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}
