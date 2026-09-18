/* A forward pass of this project's nanoGPT, in plain JavaScript.
 *
 * The explorer is a static page, so there is no Python process to ask. Rather than show
 * canned answers, it runs the model: 122,368 float32 parameters exported by
 * viz/export_model.py, and the same arithmetic nanoGPT does in PyTorch —
 *   token + position embedding
 *   2 blocks of { LayerNorm -> 4-head causal self-attention -> residual
 *                 LayerNorm -> Linear 64->256 -> GELU -> Linear 256->64 -> residual }
 *   final LayerNorm, then the tied embedding table as the output projection.
 *
 * `verify()` checks the result against logits captured from PyTorch at export time, so a
 * mistake here shows up as a loud failure instead of a plausible-looking wrong answer.
 */
"use strict";

const erf = x => {                     // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const sign = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                  - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
};
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));   // nn.GELU(), exact, not the tanh form

class TinyGPT {
  constructor(manifest, buffer) {
    this.manifest = manifest;
    this.vocabulary = manifest.vocabulary;
    this.config = manifest.config;
    this.stoi = new Map(manifest.vocabulary.map((token, index) => [token, index]));
    const floats = new Float32Array(buffer);
    this.t = {};
    for (const [name, {offset, shape}] of Object.entries(manifest.tensors)) {
      const size = shape.reduce((a, b) => a * b, 1);
      this.t[name] = floats.subarray(offset, offset + size);
    }
  }

  static async load(name, base = "data") {
    const manifest = await (await fetch(`${base}/${name}.json`)).json();
    const buffer = await (await fetch(`${base}/${manifest.weights_file}`)).arrayBuffer();
    const model = new TinyGPT(manifest, buffer);
    model.verification = model.verify();
    return model;
  }

  tokenize(text) {
    return (text.toLowerCase().match(/\w+(?:['’]\w+)*|[^\w\s]/gu) || []);
  }
  encode(text) {
    const tokens = this.tokenize(text);
    return {tokens, ids: tokens.map(t => this.stoi.has(t) ? this.stoi.get(t) : 0),
            unknown: [...new Set(tokens.filter(t => !this.stoi.has(t)))]};
  }

  /* --- small dense helpers; shapes follow PyTorch (weight is [out, in]) ------------- */
  linear(x, weight, bias, inDim, outDim) {
    const out = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = bias ? bias[o] : 0;
      const row = o * inDim;
      for (let i = 0; i < inDim; i++) sum += x[i] * weight[row + i];
      out[o] = sum;
    }
    return out;
  }
  layerNorm(x, weight, bias) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
    variance /= n;
    const scale = 1 / Math.sqrt(variance + 1e-5);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    return out;
  }

  /* --- the forward pass; returns everything the visualisation wants to show --------- */
  forward(ids) {
    const {n_embd: C, n_head: H, n_layer: L, block_size: B} = this.config;
    if (ids.length > B) ids = ids.slice(-B);
    const T = ids.length, head = C / H;
    const wte = this.t["transformer.wte.weight"], wpe = this.t["transformer.wpe.weight"];

    let x = ids.map((id, position) => {
      const row = new Float32Array(C);
      for (let i = 0; i < C; i++) row[i] = wte[id * C + i] + wpe[position * C + i];
      return row;
    });

    const attention = [];
    for (let layer = 0; layer < L; layer++) {
      const p = name => this.t[`transformer.h.${layer}.${name}`];
      const normed = x.map(row => this.layerNorm(row, p("ln_1.weight"), p("ln_1.bias")));
      const qkv = normed.map(row => this.linear(row, p("attn.c_attn.weight"), p("attn.c_attn.bias"), C, 3 * C));
      const heads = [];
      const mixed = Array.from({length: T}, () => new Float32Array(C));
      for (let h = 0; h < H; h++) {
        const weights = [];
        for (let i = 0; i < T; i++) {
          const scores = new Float32Array(i + 1);
          for (let j = 0; j <= i; j++) {          // causal: never look at a later token
            let dot = 0;
            for (let d = 0; d < head; d++) dot += qkv[i][h * head + d] * qkv[j][C + h * head + d];
            scores[j] = dot / Math.sqrt(head);
          }
          let max = -Infinity;
          for (const s of scores) max = Math.max(max, s);
          let total = 0;
          const row = new Float32Array(T);
          for (let j = 0; j <= i; j++) { row[j] = Math.exp(scores[j] - max); total += row[j]; }
          for (let j = 0; j <= i; j++) row[j] /= total;
          for (let j = 0; j <= i; j++)
            for (let d = 0; d < head; d++) mixed[i][h * head + d] += row[j] * qkv[j][2 * C + h * head + d];
          weights.push(Array.from(row));
        }
        heads.push(weights);
      }
      attention.push(heads);
      x = x.map((row, i) => {
        const projected = this.linear(mixed[i], p("attn.c_proj.weight"), p("attn.c_proj.bias"), C, C);
        const residual = new Float32Array(C);
        for (let d = 0; d < C; d++) residual[d] = row[d] + projected[d];
        const second = this.layerNorm(residual, p("ln_2.weight"), p("ln_2.bias"));
        const hidden = this.linear(second, p("mlp.c_fc.weight"), p("mlp.c_fc.bias"), C, 4 * C);
        for (let d = 0; d < 4 * C; d++) hidden[d] = gelu(hidden[d]);
        const back = this.linear(hidden, p("mlp.c_proj.weight"), p("mlp.c_proj.bias"), 4 * C, C);
        const out = new Float32Array(C);
        for (let d = 0; d < C; d++) out[d] = residual[d] + back[d];
        return out;
      });
    }

    const last = this.layerNorm(x[T - 1], this.t["transformer.ln_f.weight"], this.t["transformer.ln_f.bias"]);
    const V = this.vocabulary.length;
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {           // lm_head is tied to the embedding table
      let dot = 0;
      for (let d = 0; d < C; d++) dot += last[d] * wte[v * C + d];
      logits[v] = dot;
    }
    return {logits, attention, hidden: last, tokens: T};
  }

  softmax(logits, temperature = 1) {
    const out = new Float64Array(logits.length);
    let max = -Infinity;
    for (const value of logits) max = Math.max(max, value);
    let total = 0;
    for (let i = 0; i < logits.length; i++) {
      out[i] = Math.exp((logits[i] - max) / temperature);
      total += out[i];
    }
    for (let i = 0; i < out.length; i++) out[i] /= total;
    return out;
  }

  /** Top-k candidates as {token, id, probability}, highest first. */
  candidates(probabilities, k = 8) {
    return Array.from(probabilities, (probability, id) => ({id, probability,
        token: this.vocabulary[id]}))
      .sort((a, b) => b.probability - a.probability).slice(0, k);
  }

  /** Sample one token id, reproducibly: the same seed always gives the same word.
   *  Returns the dice roll too, so the page can show why that word and not another. */
  sample(probabilities, random) {
    const roll = random();
    let cumulative = 0;
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (roll <= cumulative) return {id: i, roll, reached: cumulative};
    }
    return {id: probabilities.length - 1, roll, reached: 1};
  }

  /** Compare against logits captured from PyTorch when the weights were exported. */
  verify() {
    let worst = 0;
    for (const check of this.manifest.checks) {
      const {logits} = this.forward(check.ids);
      check.top_tokens.forEach((token, i) => {
        const id = this.stoi.get(token);
        worst = Math.max(worst, Math.abs(logits[id] - check.top_logits[i]));
      });
    }
    return {maxLogitDifference: worst, ok: worst < 2e-3,
            prompts: this.manifest.checks.length};
  }
}

window.TinyGPT = TinyGPT;
