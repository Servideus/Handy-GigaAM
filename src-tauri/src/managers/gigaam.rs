use anyhow::{Context, Result};
use ndarray::{s, Array1, Array3, ArrayView3, Ix3};
use once_cell::sync::Lazy;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use regex::Regex;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::cmp::Ordering;
use std::f32::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MODEL_FILENAME: &str = "v3_e2e_ctc.int8.onnx";
const VOCAB_FILENAME: &str = "v3_e2e_ctc_vocab.txt";
const CONFIG_FILENAME: &str = "v3_e2e_ctc.yaml";

const MEL_MIN_CLAMP: f32 = 1e-9;
const MEL_MAX_CLAMP: f32 = 1e9;

static DECODE_SPACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b").expect("valid decode spacing regex"));

#[derive(Debug, Clone)]
struct GigaamConfig {
    sample_rate: usize,
    n_mels: usize,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
    center: bool,
    mel_scale: String,
    subsampling_factor: usize,
    model_name: String,
}

impl Default for GigaamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 64,
            win_length: 320,
            hop_length: 160,
            n_fft: 320,
            center: false,
            mel_scale: "htk".to_string(),
            subsampling_factor: 4,
            model_name: "v3_e2e_ctc".to_string(),
        }
    }
}

impl GigaamConfig {
    fn from_yaml(content: &str) -> Self {
        let mut config = Self::default();

        for raw_line in content.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            if let Some(value) = line.strip_prefix("sample_rate:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.sample_rate = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("features:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.n_mels = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("win_length:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.win_length = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("hop_length:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.hop_length = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("n_fft:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.n_fft = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("center:") {
                if let Ok(parsed) = value.trim().parse::<bool>() {
                    config.center = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("mel_scale:") {
                config.mel_scale = value.trim().to_string();
                continue;
            }

            if let Some(value) = line.strip_prefix("subsampling_factor:") {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    config.subsampling_factor = parsed;
                }
                continue;
            }

            if let Some(value) = line.strip_prefix("model_name:") {
                config.model_name = value.trim().to_string();
            }
        }

        config
    }
}

struct GigaamFrontend {
    n_mels: usize,
    win_length: usize,
    hop_length: usize,
    n_fft: usize,
    center: bool,
    hann_window: Vec<f32>,
    mel_filterbank: Vec<f32>, // [n_freq_bins, n_mels]
    fft: Arc<dyn Fft<f32>>,
}

impl GigaamFrontend {
    fn from_config(config: &GigaamConfig) -> Result<Self> {
        if config.hop_length == 0 {
            return Err(anyhow::anyhow!("Invalid GigaAM config: hop_length must be > 0"));
        }
        if config.win_length == 0 {
            return Err(anyhow::anyhow!("Invalid GigaAM config: win_length must be > 0"));
        }
        if config.n_fft == 0 {
            return Err(anyhow::anyhow!("Invalid GigaAM config: n_fft must be > 0"));
        }
        if config.n_fft < config.win_length {
            return Err(anyhow::anyhow!(
                "Invalid GigaAM config: n_fft ({}) < win_length ({})",
                config.n_fft,
                config.win_length
            ));
        }
        if !config.mel_scale.eq_ignore_ascii_case("htk") {
            return Err(anyhow::anyhow!(
                "Unsupported GigaAM mel_scale '{}'; expected 'htk'",
                config.mel_scale
            ));
        }
        if config.center {
            return Err(anyhow::anyhow!(
                "Unsupported GigaAM config: center=true is not supported for this model"
            ));
        }

        let quantize_bf16 = config.model_name.contains("v3") || (!config.center && config.n_fft == 320);
        let hann_window = build_hann_window(config.win_length, quantize_bf16);
        let mel_filterbank = build_mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            quantize_bf16,
        )?;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.n_fft);

        Ok(Self {
            n_mels: config.n_mels,
            win_length: config.win_length,
            hop_length: config.hop_length,
            n_fft: config.n_fft,
            center: config.center,
            hann_window,
            mel_filterbank,
            fft,
        })
    }

    fn extract_features(&self, samples: &[f32]) -> Result<(Array3<f32>, i64)> {
        if samples.is_empty() {
            return Ok((Array3::zeros((1, self.n_mels, 0)), 0));
        }

        if self.center {
            return Err(anyhow::anyhow!(
                "Unsupported GigaAM preprocessor configuration: center=true"
            ));
        }

        if samples.len() < self.win_length {
            return Ok((Array3::zeros((1, self.n_mels, 0)), 0));
        }

        let frame_count = ((samples.len() - self.win_length) / self.hop_length) + 1;
        let n_freq_bins = (self.n_fft / 2) + 1;

        let mut features = vec![0.0_f32; self.n_mels * frame_count];
        let mut fft_buffer = vec![Complex32::new(0.0, 0.0); self.n_fft];
        let mut power_spectrum = vec![0.0_f32; n_freq_bins];

        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_length;

            for i in 0..self.n_fft {
                let sample = if i < self.win_length {
                    samples[start + i] * self.hann_window[i]
                } else {
                    0.0
                };
                fft_buffer[i] = Complex32::new(sample, 0.0);
            }

            self.fft.process(&mut fft_buffer);

            for (bin_idx, power) in power_spectrum.iter_mut().enumerate() {
                let complex = fft_buffer[bin_idx];
                *power = complex.re.mul_add(complex.re, complex.im * complex.im);
            }

            for mel_idx in 0..self.n_mels {
                let mut mel_energy = 0.0_f32;
                for (bin_idx, &power) in power_spectrum.iter().enumerate() {
                    mel_energy += power * self.mel_filterbank[bin_idx * self.n_mels + mel_idx];
                }
                let clamped = mel_energy.clamp(MEL_MIN_CLAMP, MEL_MAX_CLAMP);
                features[mel_idx * frame_count + frame_idx] = clamped.ln();
            }
        }

        let features = Array3::from_shape_vec((1, self.n_mels, frame_count), features)?;
        Ok((features, frame_count as i64))
    }
}

struct GigaamModel {
    session: Session,
    frontend: GigaamFrontend,
    vocab: Vec<String>,
    blank_idx: usize,
    subsampling_factor: usize,
    features_input_name: String,
    feature_lengths_input_name: String,
    logits_output_name: String,
}

impl GigaamModel {
    fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join(MODEL_FILENAME);
        let vocab_path = model_dir.join(VOCAB_FILENAME);
        let config_path = model_dir.join(CONFIG_FILENAME);

        if !model_path.exists() {
            return Err(anyhow::anyhow!(
                "Missing GigaAM model file: {}",
                model_path.display()
            ));
        }
        if !vocab_path.exists() {
            return Err(anyhow::anyhow!(
                "Missing GigaAM vocab file: {}",
                vocab_path.display()
            ));
        }
        if !config_path.exists() {
            return Err(anyhow::anyhow!(
                "Missing GigaAM config file: {}",
                config_path.display()
            ));
        }

        let vocab_content = fs::read_to_string(&vocab_path).with_context(|| {
            format!("Failed to read GigaAM vocab file: {}", vocab_path.display())
        })?;
        let (vocab, blank_idx) = parse_vocab_content(&vocab_content)?;

        let config_content = fs::read_to_string(&config_path).with_context(|| {
            format!("Failed to read GigaAM config file: {}", config_path.display())
        })?;
        let config = GigaamConfig::from_yaml(&config_content);
        if config.sample_rate != 16_000 {
            return Err(anyhow::anyhow!(
                "Unsupported GigaAM sample rate {} Hz; Handy currently provides 16000 Hz PCM input",
                config.sample_rate
            ));
        }
        let frontend = GigaamFrontend::from_config(&config)?;

        let providers = vec![CPUExecutionProvider::default().build()];
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(&model_path)
            .with_context(|| {
                format!("Failed to initialize ONNX Runtime session: {}", model_path.display())
            })?;

        for input in &session.inputs {
            log::info!(
                "GigaAM input: name={}, type={:?}",
                input.name,
                input.input_type
            );
        }
        for output in &session.outputs {
            log::info!("GigaAM output: name={}", output.name);
        }

        let features_input_name = session
            .inputs
            .iter()
            .find(|input| input.name == "features")
            .or_else(|| {
                session.inputs.iter().find(|input| {
                    input
                        .input_type
                        .tensor_shape()
                        .map(|shape| shape.len() == 3)
                        .unwrap_or(false)
                })
            })
            .map(|input| input.name.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to determine GigaAM features input"))?;

        let feature_lengths_input_name = session
            .inputs
            .iter()
            .find(|input| input.name == "feature_lengths")
            .or_else(|| {
                session.inputs.iter().find(|input| {
                    input
                        .input_type
                        .tensor_shape()
                        .map(|shape| shape.len() == 1)
                        .unwrap_or(false)
                })
            })
            .map(|input| input.name.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to determine GigaAM feature lengths input"))?;

        let logits_output_name = session
            .outputs
            .iter()
            .find(|output| output.name == "log_probs" || output.name == "logits")
            .or_else(|| session.outputs.first())
            .map(|output| output.name.clone())
            .ok_or_else(|| anyhow::anyhow!("Failed to determine GigaAM logits output"))?;

        Ok(Self {
            session,
            frontend,
            vocab,
            blank_idx,
            subsampling_factor: config.subsampling_factor.max(1),
            features_input_name,
            feature_lengths_input_name,
            logits_output_name,
        })
    }

    fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        let (features, feature_length) = self.frontend.extract_features(samples)?;
        if feature_length == 0 {
            return Ok(String::new());
        }

        let feature_lengths = Array1::from_vec(vec![feature_length]);
        let inputs = inputs![
            self.features_input_name.as_str() => TensorRef::from_array_view(features.view())?,
            self.feature_lengths_input_name.as_str() => TensorRef::from_array_view(feature_lengths.view())?,
        ];
        let outputs = self.session.run(inputs)?;

        let logits = outputs
            .get(self.logits_output_name.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GigaAM output '{}' not found in inference outputs",
                    self.logits_output_name
                )
            })?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<Ix3>()?;

        let encoded_len = ((feature_length - 1) / self.subsampling_factor as i64 + 1).max(0) as usize;
        let token_ids = ctc_greedy_decode_ids(logits.view(), encoded_len, self.blank_idx);
        Ok(decode_token_ids_to_text(&token_ids, &self.vocab))
    }
}

#[derive(Default)]
pub struct GigaamEngine {
    loaded_model_path: Option<PathBuf>,
    model: Option<GigaamModel>,
}

impl GigaamEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_model(&mut self, model_path: &Path) -> Result<()> {
        let model = GigaamModel::new(model_path)?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.model = None;
    }

    pub fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<String> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GigaAM model is not loaded"))?;
        model.transcribe_samples(&samples)
    }
}

fn parse_vocab_content(content: &str) -> Result<(Vec<String>, usize)> {
    let mut entries = Vec::<(usize, String)>::new();
    let mut max_id = 0_usize;
    let mut blank_idx = None;

    for raw_line in content.lines() {
        let line = raw_line.trim_end();
        if line.is_empty() {
            continue;
        }
        let (token, id_str) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow::anyhow!("Invalid vocab line: '{}'", line))?;
        let id = id_str
            .trim()
            .parse::<usize>()
            .with_context(|| format!("Invalid vocab token id in line '{}'", line))?;

        if token == "<blk>" {
            blank_idx = Some(id);
        }

        max_id = max_id.max(id);
        entries.push((id, token.replace('\u{2581}', " ")));
    }

    let blank_idx =
        blank_idx.ok_or_else(|| anyhow::anyhow!("Missing <blk> token in vocabulary file"))?;

    let mut vocab = vec![String::new(); max_id + 1];
    for (id, token) in entries {
        vocab[id] = token;
    }

    Ok((vocab, blank_idx))
}

fn ctc_greedy_decode_ids(
    logits: ArrayView3<'_, f32>,
    encoded_len: usize,
    blank_idx: usize,
) -> Vec<usize> {
    let time_steps = logits.shape()[1];
    let usable_steps = encoded_len.min(time_steps);

    let mut token_ids = Vec::with_capacity(usable_steps);
    let mut prev_token = blank_idx;

    for frame_idx in 0..usable_steps {
        let frame = logits.slice(s![0, frame_idx, ..]);
        let best_idx = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(blank_idx);

        if best_idx != blank_idx && best_idx != prev_token {
            token_ids.push(best_idx);
        }
        prev_token = best_idx;
    }

    token_ids
}

fn decode_token_ids_to_text(token_ids: &[usize], vocab: &[String]) -> String {
    let concatenated = token_ids
        .iter()
        .filter_map(|&id| vocab.get(id))
        .fold(String::new(), |mut text, token| {
            text.push_str(token);
            text
        });

    DECODE_SPACE_RE
        .replace_all(&concatenated, |captures: &regex::Captures<'_>| {
            if captures.get(1).is_some() {
                " "
            } else {
                ""
            }
        })
        .to_string()
}

fn build_hann_window(win_length: usize, quantize_bf16: bool) -> Vec<f32> {
    if win_length == 1 {
        return vec![1.0];
    }

    (0..win_length)
        .map(|n| {
            let value = 0.5 - 0.5 * (2.0 * PI * n as f32 / (win_length as f32 - 1.0)).cos();
            if quantize_bf16 {
                quantize_to_bf16(value)
            } else {
                value
            }
        })
        .collect()
}

fn build_mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    quantize_bf16: bool,
) -> Result<Vec<f32>> {
    let n_freq_bins = n_fft / 2 + 1;
    let f_min = 0.0_f32;
    let f_max = (sample_rate as f32) / 2.0;

    let mel_min = hz_to_mel_htk(f_min);
    let mel_max = hz_to_mel_htk(f_max);

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32 / (n_mels + 1) as f32))
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_htk).collect();
    let fft_freqs: Vec<f32> = (0..n_freq_bins)
        .map(|bin| bin as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filterbank = vec![0.0_f32; n_freq_bins * n_mels];

    for mel_idx in 0..n_mels {
        let left = hz_points[mel_idx];
        let center = hz_points[mel_idx + 1];
        let right = hz_points[mel_idx + 2];

        if center <= left || right <= center {
            return Err(anyhow::anyhow!(
                "Invalid mel filter points for index {}",
                mel_idx
            ));
        }

        for (bin_idx, &freq) in fft_freqs.iter().enumerate() {
            let weight = if freq >= left && freq <= center {
                (freq - left) / (center - left)
            } else if freq > center && freq <= right {
                (right - freq) / (right - center)
            } else {
                0.0
            };

            if weight > 0.0 {
                filterbank[bin_idx * n_mels + mel_idx] = if quantize_bf16 {
                    quantize_to_bf16(weight)
                } else {
                    weight
                };
            }
        }
    }

    Ok(filterbank)
}

#[inline]
fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[inline]
fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}

#[inline]
fn quantize_to_bf16(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xFFFF_0000)
}
