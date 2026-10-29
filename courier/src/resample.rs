//! Streaming sample-rate conversion: a windowed-sinc filter evaluated at each output position. The
//! microphone comes at the device's rate (often 48 kHz) and the models want 16 kHz; the voice comes at
//! 22.05 kHz and the speakers want the device's rate.

use std::f64::consts::PI;

pub struct Resampler {
    step: f64,   // input samples per output sample
    cutoff: f64, // as a fraction of the input Nyquist frequency
    half: usize, // filter half-width, in input samples
    buf: Vec<f32>,
    pos: f64, // position of the next output sample in `buf`
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Resampler {
        let step = from as f64 / to as f64;
        // Below the lower of the two Nyquist frequencies, with a little room for the transition band.
        let cutoff = (1.0 / step).min(1.0) * 0.92;
        let half = (16.0 / cutoff).ceil() as usize;
        // Start with `half` zeros of history, so the first output is centred on the first input.
        Resampler { step, cutoff, half, buf: vec![0.0; half], pos: half as f64 }
    }

    pub fn is_identity(&self) -> bool {
        self.step == 1.0
    }

    fn kernel(&self, x: f64) -> f64 {
        // sinc lowpass, Blackman window over [-half, half].
        let t = x * self.cutoff;
        let sinc = if t.abs() < 1e-9 { 1.0 } else { (PI * t).sin() / (PI * t) };
        let w = x / self.half as f64;
        if w.abs() >= 1.0 {
            return 0.0;
        }
        let blackman = 0.42 + 0.5 * (PI * w).cos() + 0.08 * (2.0 * PI * w).cos();
        self.cutoff * sinc * blackman
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.is_identity() {
            return input.to_vec();
        }
        self.buf.extend_from_slice(input);
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 2);
        let h = self.half as f64;
        while self.pos + h < self.buf.len() as f64 {
            let centre = self.pos.floor() as isize;
            let frac = self.pos - centre as f64;
            let mut acc = 0.0;
            let lo = centre - self.half as isize + 1;
            let hi = centre + self.half as isize;
            for k in lo..=hi {
                if k < 0 || k as usize >= self.buf.len() {
                    continue;
                }
                acc += self.buf[k as usize] as f64 * self.kernel((k - centre) as f64 - frac);
            }
            out.push(acc as f32);
            self.pos += self.step;
        }
        // Keep what the next outputs still need.
        let keep_from = (self.pos.floor() as usize).saturating_sub(self.half);
        if keep_from > 0 {
            self.buf.drain(..keep_from);
            self.pos -= keep_from as f64;
        }
        out
    }
}

/// Whole-buffer conversion.
pub fn convert(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    let mut r = Resampler::new(from, to);
    let mut out = r.process(samples);
    // Flush the filter's delay with silence.
    if !r.is_identity() {
        out.extend(r.process(&vec![0.0; r.half + 1]));
        out.truncate((samples.len() as f64 * to as f64 / from as f64).round() as usize);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * PI * hz * i as f64 / rate as f64).sin() as f32 * 0.5).collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn keeps_a_tone_in_band() {
        let out = convert(&tone(1000.0, 48000, 48000), 48000, 16000);
        assert_eq!(out.len(), 16000);
        let mid = &out[2000..14000];
        assert!((rms(mid) - 0.5 / 2f64.sqrt()).abs() < 0.01, "rms {}", rms(mid));
        // Same phase as a tone made at 16 kHz.
        let want = tone(1000.0, 16000, 16000);
        let err: f64 = mid.iter().zip(&want[2000..14000]).map(|(a, b)| ((a - b) as f64).abs()).fold(0.0, f64::max);
        assert!(err < 0.02, "max error {err}");
    }

    #[test]
    fn removes_what_would_alias() {
        // 12 kHz cannot exist at 16 kHz: it must be filtered out, not folded down to 4 kHz.
        let out = convert(&tone(12000.0, 48000, 48000), 48000, 16000);
        assert!(rms(&out[2000..14000]) < 0.005, "rms {}", rms(&out[2000..14000]));
    }

    #[test]
    fn streaming_equals_whole() {
        let x = tone(440.0, 22050, 22050);
        let whole = convert(&x, 22050, 48000);
        let mut r = Resampler::new(22050, 48000);
        let mut parts = Vec::new();
        for c in x.chunks(317) {
            parts.extend(r.process(c));
        }
        let n = parts.len().min(whole.len());
        assert!(n > 47000);
        let err = parts[..n].iter().zip(&whole[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-6, "max difference {err}");
    }

    #[test]
    fn identity_passes_through() {
        let x = tone(440.0, 16000, 1000);
        assert_eq!(convert(&x, 16000, 16000), x);
    }
}
