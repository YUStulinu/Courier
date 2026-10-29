//! WAV files in and out (recordings for `sim`, the evaluations' audio).

use anyhow::{Context, Result};
use std::path::Path;

/// Mono samples (channels averaged) and the sample rate.
pub fn read(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("reading {}", path.display()))?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    Ok((all.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect(), spec.sample_rate))
}

pub fn write_i16(path: &Path, pcm: &[i16], rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("writing {}", path.display()))?;
    for &s in pcm {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}

pub fn write_f32(path: &Path, x: &[f32], rate: u32) -> Result<()> {
    let pcm: Vec<i16> = x.iter().map(|&v| (v.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
    write_i16(path, &pcm, rate)
}
