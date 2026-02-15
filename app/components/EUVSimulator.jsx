'use client'

import { useState, useEffect, useCallback, useMemo, useRef } from "react";

// ============================================================
// PHYSICS ENGINE: Coupled Stochastic ODE for MOR PR Aging
// ============================================================
const BOLTZMANN_EV = 8.617e-5; // eV/K

function langmuirCoverage(Kads, RH) {
  return (Kads * RH) / (1 + Kads * RH);
}

function arrheniusRate(A, Ea, T) {
  return A * Math.exp(-Ea / (BOLTZMANN_EV * T));
}

function solveAgingODE(params, dt_hours, totalHours) {
  const { T, RH, pO2, Kads, fOH0, rMol0, nSn,
    A_A, Ea_A, A_B, Ea_B, A_C, Ea_C, A_D, Ea_D, A_D2, Ea_D2,
    kE_prime, Ea_Diff, alpha } = params;

  const kA = arrheniusRate(A_A, Ea_A, T);
  const kB = arrheniusRate(A_B, Ea_B, T);
  const kC = arrheniusRate(A_C, Ea_C, T);
  const kD = arrheniusRate(A_D, Ea_D, T);
  const kD2 = arrheniusRate(A_D2, Ea_D2, T);
  const thetaH2O = langmuirCoverage(Kads, RH);
  const Tref = 298;
  const kA_rev = kA * 0.01; // reverse hydrolysis (small)
  
  // Diffusion coeff for aggregation
  const D0 = 1e-14; // cm²/s baseline
  const D = D0 * Math.exp(-Ea_Diff / (BOLTZMANN_EV * T));
  // LSW coarsening rate ~ D * gamma * Vm * c_inf / RT
  const kE_eff = kE_prime * (D / (D0 * Math.exp(-Ea_Diff / (BOLTZMANN_EV * 298))));

  let fSnOSn = 1.0;
  let fSnC = 1.0;
  let fSnOH = fOH0;
  let fSnII = 0.01;
  let rAgg = rMol0;

  const dt_s = dt_hours * 3600;
  const steps = Math.round(totalHours / dt_hours);
  const history = [];

  for (let i = 0; i <= steps; i++) {
    const t_hr = i * dt_hours;

    // Compute lithographic observables from state
    const sigmaSnC = 1.0, sigmaSnO = 0.65; // relative abs cross-sections
    const deltaOx = 0.15;
    const muRatio = (fSnC * sigmaSnC + (1 - fSnC) * sigmaSnO) / sigmaSnC;
    const muAbs = muRatio * (1 + deltaOx * (1 - fSnII / 0.01));
    
    const beta = 0.3;
    const phiEff = fSnC * (1 - beta * (1 - fSnOSn));
    
    const gamma = 2.5, deltaGel = 0.15;
    const DRcontrast = Math.exp(gamma * phiEff) * (1 - deltaGel * (1 - fSnOSn) * (1 - fSnC));
    
    const etaAgg = 2.5, etaOH = 1.5;
    const sigmaDR = Math.sqrt(1 + etaAgg * Math.pow(rAgg / rMol0 - 1, 2)) *
                    (1 + etaOH * Math.abs(fSnOH - fOH0));
    const DRslopeRatio = phiEff * muAbs;
    
    const LERratio = Math.sqrt(1 + Math.pow(sigmaDR, 2) * Math.pow(1 / DRslopeRatio, 2));
    
    // Effective Nmss for defect calc
    const NmssRatio = phiEff * fSnOSn * Math.pow(rMol0 / rAgg, 3);
    
    history.push({
      time: t_hr,
      fSnOSn, fSnC, fSnOH, fSnII, rAgg,
      muAbs, phiEff, DRcontrast, LERratio, NmssRatio, sigmaDR, DRslopeRatio
    });

    if (i === steps) break;

    // RK4 integration
    const derivs = (s) => {
      const [_fSnOSn, _fSnC, _fSnOH, _fSnII, _rAgg] = s;
      const dfSnOSn = -kA * _fSnOSn * thetaH2O + kB * _fSnOH * _fSnOH - kA_rev * (1 - _fSnOSn) * (1 - thetaH2O);
      const dfSnC = -kC * _fSnC * thetaH2O - kD2 * _fSnC * pO2 * Math.pow(T / Tref, 2);
      const dfSnOH = 2 * kA * _fSnOSn * thetaH2O - 2 * kB * _fSnOH * _fSnOH + kC * _fSnC * thetaH2O;
      const dfSnII = alpha * Math.abs(dfSnC) - kD * _fSnII * pO2;
      const drAgg = kE_eff / (3 * _rAgg * _rAgg + 1e-6);
      return [dfSnOSn, dfSnC, dfSnOH, dfSnII, drAgg];
    };

    const state = [fSnOSn, fSnC, fSnOH, fSnII, rAgg];
    const k1 = derivs(state).map(v => v * dt_s);
    const k2 = derivs(state.map((v, j) => v + k1[j] * 0.5)).map(v => v * dt_s);
    const k3 = derivs(state.map((v, j) => v + k2[j] * 0.5)).map(v => v * dt_s);
    const k4 = derivs(state.map((v, j) => v + k3[j])).map(v => v * dt_s);

    const newState = state.map((v, j) => v + (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) / 6);
    
    fSnOSn = Math.max(0, Math.min(1, newState[0]));
    fSnC = Math.max(0, Math.min(1, newState[1]));
    fSnOH = Math.max(0, Math.min(1, newState[2]));
    fSnII = Math.max(0, Math.min(1, newState[3]));
    rAgg = Math.max(rMol0, newState[4]);
  }
  return history;
}

// ============================================================
// STOCHASTIC PATTERNING ENGINE (Fukuda-inspired)
// ============================================================

// Poisson CDF (regularized incomplete gamma)
function poissonCDF(k, lambda) {
  if (lambda <= 0) return k >= 0 ? 1.0 : 0.0;
  let sum = 0;
  let term = Math.exp(-lambda);
  for (let i = 0; i <= k; i++) {
    if (i > 0) term *= lambda / i;
    sum += term;
    if (sum > 1 - 1e-15) return 1.0;
  }
  return Math.min(1.0, sum);
}

// Binomial PMF
function binomPMF(k, n, p) {
  if (p <= 0) return k === 0 ? 1 : 0;
  if (p >= 1) return k === n ? 1 : 0;
  // Use log to avoid overflow
  let logPMF = 0;
  for (let i = 0; i < k; i++) logPMF += Math.log(n - i) - Math.log(i + 1);
  logPMF += k * Math.log(p) + (n - k) * Math.log(1 - p);
  return Math.exp(logPMF);
}

// Binomial CDF
function binomCDF(k, n, p) {
  let sum = 0;
  for (let i = 0; i <= k; i++) sum += binomPMF(i, n, p);
  return Math.min(1.0, sum);
}

// Beta function using Stirling approximation
function lnGamma(z) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
  let x = z, y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function betaBinomPMF(k, n, alpha, beta_p) {
  if (alpha <= 0 || beta_p <= 0) return binomPMF(k, n, alpha / (alpha + beta_p + 1e-10));
  const logB = (a, b) => lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  const lnC = lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
  const val = lnC + logB(k + alpha, n - k + beta_p) - logB(alpha, beta_p);
  return Math.exp(val);
}

function computePatternProfile(params) {
  const { halfPitch, NA, lambda, moleculeSize, filmThickness, dose, blur,
    thRE, thMSS, thSCF, absCoeff, seEnergy, overDispersion,
    agingState } = params;
  
  const hp = halfPitch;
  const nPositions = 41; // positions across half-period
  const positions = [];
  for (let i = 0; i < nPositions; i++) {
    positions.push(-hp + (2 * hp * i) / (nPositions - 1));
  }

  // Image intensity (dipole illumination, 1D periodic)
  const k1 = hp / (lambda / NA);
  const NILS_nominal = Math.PI * k1; // simplified NILS
  const imageIntensity = positions.map(x => {
    const normalized = x / hp;
    return 0.5 * (1 + Math.cos(Math.PI * normalized));
  });

  // Apply aging effects
  const muEff = agingState ? agingState.muAbs : 1.0;
  const phiEff = agingState ? agingState.phiEff : 1.0;
  const sigmaDR_aging = agingState ? agingState.sigmaDR : 1.0;

  // Photon absorption density (nm⁻³)
  const photonEnergy_eV = 92; // EUV
  const photonEnergy_J = photonEnergy_eV * 1.602e-19;
  const dose_J_per_nm2 = dose * 1e-3 * 1e-14; // mJ/cm² -> J/nm²

  // Reaction probability at each position
  const nMolThickness = Math.round(filmThickness / moleculeSize);
  const nSites = Math.max(1, Math.round(Math.pow(moleculeSize, 3) * 2)); // reaction sites per molecule
  
  const blurSigma = blur / moleculeSize;
  
  // Compute event probabilities through the hierarchical network
  const profileData = positions.map((x, idx) => {
    const I = imageIntensity[idx];
    const photonDensity = dose_J_per_nm2 * I * muEff * absCoeff / photonEnergy_J;
    
    // Mean reactions per molecule (Poisson parameter)
    const nReactions = photonDensity * Math.pow(moleculeSize, 3) * phiEff * seEnergy / 15;
    
    // Step 1: Reaction → Molecular solubility switching (Binomial/Poisson)
    const pReaction = Math.min(0.999, 1 - Math.exp(-nReactions / Math.max(1, nSites)));
    const pMSS = 1 - binomCDF(thRE - 1, nSites, pReaction);
    
    // Step 2: MSS → Sub-cluster formation (Beta-binomial with overdispersion)
    const w = overDispersion; // overdispersion parameter
    const alpha_bb = pMSS / (w + 1e-6) * (1 - w);
    const beta_bb = (1 - pMSS) / (w + 1e-6) * (1 - w);
    let pSC1 = 0;
    for (let k = thMSS; k <= 27; k++) {
      pSC1 += betaBinomPMF(k, 27, Math.max(0.01, alpha_bb), Math.max(0.01, beta_bb));
    }
    pSC1 = Math.min(1, Math.max(0, pSC1));
    
    // Step 3: SC1 → Molecular dissolution (Bernoulli convolution approximation)
    // Account for spatial variation across 3x3x3 voxels
    const gradientEffect = Math.abs(Math.sin(Math.PI * x / hp)) * 0.3;
    const pSC2_mean = 1 - binomCDF(thSCF - 1, 27, pSC1);
    const pSC2 = pSC2_mean * (1 - gradientEffect * (moleculeSize / hp));
    
    // Step 4: Dissolution rate (through thickness sum)
    const dissolutionRate = Math.min(nMolThickness, 
      nMolThickness * Math.max(0, Math.min(1, pSC2)));
    
    // PMF width (stochasticity measure) - beta-binomial variance
    const variance = nMolThickness * pSC2 * (1 - pSC2) * 
                     (1 + (nMolThickness - 1) * w * sigmaDR_aging) / (nMolThickness);
    const pmfWidth = Math.sqrt(Math.max(0, variance));
    
    return {
      position: x,
      imageIntensity: I,
      reactionDensity: nReactions,
      pMSS,
      pSC1,
      dissolutionRate: dissolutionRate / nMolThickness,
      pmfWidth: pmfWidth / nMolThickness,
      pDefect: pSC2 < 0.001 ? pSC2 : (pSC2 > 0.999 ? 1 - pSC2 : 0)
    };
  });

  return { positions, profileData, k1, NILS_nominal, nMolThickness };
}

// Compute lithographic metrics from profile
function computeMetrics(profileData, halfPitch, moleculeSize, dose, filmThickness) {
  const n = profileData.length;
  const mid = Math.floor(n / 2);
  
  // Find edge position (dissolution rate = 0.5 threshold)
  let edgeIdx = mid;
  for (let i = 0; i < n - 1; i++) {
    if ((profileData[i].dissolutionRate - 0.5) * (profileData[i + 1].dissolutionRate - 0.5) <= 0) {
      edgeIdx = i;
      break;
    }
  }
  
  // DR slope at edge
  const dx = profileData[1].position - profileData[0].position;
  const drSlope = edgeIdx > 0 && edgeIdx < n - 1 ? 
    Math.abs(profileData[edgeIdx + 1].dissolutionRate - profileData[edgeIdx - 1].dissolutionRate) / (2 * dx) : 0.1;
  
  // PMF width at edge
  const pmfWidthEdge = profileData[edgeIdx] ? profileData[edgeIdx].pmfWidth : 0.1;
  
  // LER (3σ) = ΔDR / (∂DR/∂x) with molecular granularity
  const LER_3sigma = Math.min(halfPitch * 0.5, 3 * pmfWidthEdge / (drSlope + 1e-6));
  
  // LWR ≈ √2 × LER
  const LWR_3sigma = LER_3sigma * Math.SQRT2;
  
  // CD from actual profile
  let cdCount = 0;
  for (let i = 0; i < n; i++) {
    if (profileData[i].dissolutionRate > 0.5) cdCount++;
  }
  const actualCD = cdCount * dx;
  const cdError = actualCD - halfPitch;
  
  // Defect probability (bridge/break in unexposed/exposed centers)
  const pBridge = profileData[0] ? profileData[0].dissolutionRate : 0;
  const pBreak = profileData[n - 1] ? (1 - profileData[n - 1].dissolutionRate) : 0;
  const defectProb = Math.max(pBridge, pBreak);
  
  // Universal DR log-slope
  const drLogSlope = drSlope * (1 / (profileData[edgeIdx]?.dissolutionRate || 0.5 + 1e-6));
  
  // Dose to size
  const doseToSize = dose;
  
  // Process window (simplified)
  const processWindow = Math.max(0, 1 - 2 * LER_3sigma / halfPitch - defectProb * 10);

  return {
    LER_3sigma: Math.max(0.1, LER_3sigma),
    LWR_3sigma: Math.max(0.14, LWR_3sigma),
    actualCD,
    cdError,
    defectProb: Math.max(1e-12, defectProb),
    drLogSlope: Math.max(0.1, drLogSlope * halfPitch),
    doseToSize,
    processWindow: Math.max(0, Math.min(1, processWindow)),
    edgePosition: profileData[edgeIdx]?.position || 0,
    pmfWidthEdge
  };
}

// ============================================================
// 2D PATTERN GENERATOR
// ============================================================
function generatePattern2D(type, gridSize, halfPitch, metrics, agingState, moleculeSize) {
  const grid = [];
  const n = gridSize;
  const pixelSize = halfPitch * 4 / n; // physical size per pixel
  
  const LER = metrics.LER_3sigma;
  const cdShift = metrics.cdError;
  
  // Seeded pseudo-random for reproducibility
  let seed = 42 + Math.round((agingState?.time || 0) * 137);
  const random = () => {
    seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  
  // Box-Muller for Gaussian noise
  const gaussRandom = () => {
    const u1 = random() + 1e-10;
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  
  // Generate correlated noise (simplified spatial correlation)
  const correlationLength = Math.max(2, Math.round(moleculeSize * 3 / pixelSize));
  const noiseField = Array(n).fill(null).map(() => Array(n).fill(0));
  // White noise
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      noiseField[i][j] = gaussRandom();
  
  // Simple box-blur for spatial correlation
  const blurredNoise = Array(n).fill(null).map(() => Array(n).fill(0));
  const bk = correlationLength;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0, cnt = 0;
      for (let di = -bk; di <= bk; di++) {
        for (let dj = -bk; dj <= bk; dj++) {
          const ii = (i + di + n) % n;
          const jj = (j + dj + n) % n;
          const w = Math.exp(-(di*di + dj*dj) / (2*bk*bk));
          sum += noiseField[ii][jj] * w;
          cnt += w;
        }
      }
      blurredNoise[i][j] = sum / cnt;
    }
  }

  for (let i = 0; i < n; i++) {
    grid[i] = [];
    for (let j = 0; j < n; j++) {
      const x = (j - n/2) * pixelSize;
      const y = (i - n/2) * pixelSize;
      
      let isExposed = false;
      const noise = blurredNoise[i][j] * LER * 0.5;
      
      if (type === 'line_space') {
        // Periodic L/S in x direction
        const period = halfPitch * 2;
        const xMod = ((x % period) + period) % period;
        const edgePos = halfPitch / 2 + cdShift / 2 + noise;
        isExposed = xMod > (halfPitch - edgePos) && xMod < (halfPitch + edgePos);
      } else if (type === 'pillar') {
        // Hexagonal array of cylindrical pillars
        const period = halfPitch * 2;
        const row = Math.round(y / (period * Math.sqrt(3) / 2));
        const offset = (row % 2) * period / 2;
        const cx = Math.round((x - offset) / period) * period + offset;
        const cy = row * period * Math.sqrt(3) / 2;
        const dist = Math.sqrt((x - cx)**2 + (y - cy)**2);
        const radius = halfPitch * 0.4 + cdShift / 2 + noise;
        isExposed = dist < radius;
      } else if (type === 'square') {
        // Square array
        const period = halfPitch * 2;
        const cx = Math.round(x / period) * period;
        const cy = Math.round(y / period) * period;
        const halfW = halfPitch * 0.4 + cdShift / 2 + noise;
        isExposed = Math.abs(x - cx) < halfW && Math.abs(y - cy) < halfW;
      }
      
      // Stochastic defects
      const defectChance = random();
      if (defectChance < metrics.defectProb * 100) { // amplified for visibility
        isExposed = !isExposed;
      }
      
      grid[i][j] = isExposed ? 1 : 0;
    }
  }
  return grid;
}

function generateCrossSection(halfPitch, metrics, agingState, filmThickness, moleculeSize) {
  const width = 80;
  const height = 20;
  const grid = [];
  const pixelW = halfPitch * 4 / width;
  const pixelH = filmThickness / height;
  
  let seed = 99 + Math.round((agingState?.time || 0) * 53);
  const random = () => { seed = (seed * 1664525 + 1013904223) & 0x7fffffff; return seed / 0x7fffffff; };
  const gaussRandom = () => { const u1 = random()+1e-10, u2 = random(); return Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2); };
  
  for (let iz = 0; iz < height; iz++) {
    grid[iz] = [];
    const depth = iz / height;
    const depthAttenuation = Math.exp(-0.004 * depth * filmThickness); // absorption through thickness
    for (let ix = 0; ix < width; ix++) {
      const x = (ix - width/2) * pixelW;
      const period = halfPitch * 2;
      const xMod = ((x % period) + period) % period;
      const noise = gaussRandom() * metrics.LER_3sigma * 0.3;
      const edgePos = halfPitch / 2 + metrics.cdError / 2 + noise;
      const isExposed = xMod > (halfPitch - edgePos) && xMod < (halfPitch + edgePos);
      
      // Sidewall angle degradation with aging
      const sidewallNoise = gaussRandom() * metrics.LER_3sigma * 0.2 * depth;
      const sidewallShift = depth * (1 - (agingState?.DRslopeRatio || 1)) * moleculeSize;
      
      let val = isExposed ? 1 : 0;
      // Add footing/undercut effects near edges
      if (Math.abs(xMod - (halfPitch - edgePos)) < moleculeSize * 2 ||
          Math.abs(xMod - (halfPitch + edgePos)) < moleculeSize * 2) {
        val = random() > 0.5 ? 1 : 0;
      }
      
      grid[iz][ix] = val;
    }
  }
  return grid;
}

// ============================================================
// CHART COMPONENTS
// ============================================================
function MiniChart({ data, xKey, yKey, color, label, width = 280, height = 140, logScale = false }) {
  if (!data || data.length === 0) return null;
  const margin = { top: 20, right: 15, bottom: 30, left: 45 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  
  const xVals = data.map(d => d[xKey]);
  const yVals = data.map(d => logScale ? Math.log10(Math.max(1e-12, d[yKey])) : d[yKey]);
  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  
  const points = data.map((d, i) => {
    const px = margin.left + ((xVals[i] - xMin) / xRange) * w;
    const py = margin.top + (1 - (yVals[i] - yMin) / yRange) * h;
    return `${px},${py}`;
  }).join(' ');
  
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <text x={width/2} y={14} textAnchor="middle" fontSize="10" fill="#94a3b8" fontFamily="monospace">{label}</text>
      <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + h} stroke="#334155" strokeWidth="1"/>
      <line x1={margin.left} y1={margin.top + h} x2={margin.left + w} y2={margin.top + h} stroke="#334155" strokeWidth="1"/>
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round"/>
      {/* Axis labels */}
      <text x={margin.left - 5} y={margin.top + 5} textAnchor="end" fontSize="8" fill="#64748b" fontFamily="monospace">
        {logScale ? yMax.toFixed(1) : yMax.toFixed(2)}
      </text>
      <text x={margin.left - 5} y={margin.top + h} textAnchor="end" fontSize="8" fill="#64748b" fontFamily="monospace">
        {logScale ? yMin.toFixed(1) : yMin.toFixed(2)}
      </text>
      <text x={margin.left} y={margin.top + h + 12} fontSize="8" fill="#64748b" fontFamily="monospace">{xMin.toFixed(0)}</text>
      <text x={margin.left + w} y={margin.top + h + 12} textAnchor="end" fontSize="8" fill="#64748b" fontFamily="monospace">{xMax.toFixed(0)}h</text>
    </svg>
  );
}

function PMFHeatmap({ profileData, width = 320, height = 180, nMolThickness }) {
  if (!profileData || profileData.length === 0) return null;
  const margin = { top: 25, right: 10, bottom: 30, left: 40 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const nPos = profileData.length;
  const nDR = nMolThickness || 10;
  const cellW = w / nPos;
  const cellH = h / nDR;
  
  const cells = [];
  for (let ix = 0; ix < nPos; ix++) {
    const d = profileData[ix];
    const mean = d.dissolutionRate * nDR;
    const sigma = d.pmfWidth * nDR + 0.5;
    for (let idr = 0; idr < nDR; idr++) {
      const prob = Math.exp(-0.5 * Math.pow((idr - mean) / (sigma + 0.1), 2)) / (sigma * 2.507 + 0.1);
      const logProb = Math.max(-4, Math.log10(prob + 1e-5));
      const norm = (logProb + 4) / 4;
      cells.push({
        x: margin.left + ix * cellW,
        y: margin.top + (nDR - 1 - idr) * cellH,
        w: cellW + 0.5,
        h: cellH + 0.5,
        color: `hsl(${200 + norm * 40}, ${50 + norm * 50}%, ${10 + norm * 70}%)`
      });
    }
  }
  
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <text x={width/2} y={14} textAnchor="middle" fontSize="10" fill="#94a3b8" fontFamily="monospace">
        Dissolution Rate PMF Distribution
      </text>
      {cells.map((c, i) => (
        <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.color} />
      ))}
      <text x={margin.left - 3} y={margin.top + 5} textAnchor="end" fontSize="8" fill="#64748b">{nDR}</text>
      <text x={margin.left - 3} y={margin.top + h} textAnchor="end" fontSize="8" fill="#64748b">0</text>
      <text x={margin.left + 2} y={margin.top + h + 12} fontSize="8" fill="#64748b">Unexposed</text>
      <text x={margin.left + w - 2} y={margin.top + h + 12} textAnchor="end" fontSize="8" fill="#64748b">Exposed</text>
      <text x={margin.left - 5} y={margin.top + h/2} textAnchor="end" fontSize="8" fill="#64748b" transform={`rotate(-90, ${margin.left - 18}, ${margin.top + h/2})`}>DR</text>
    </svg>
  );
}

function PatternCanvas({ grid, width = 200, height = 200, label, colorScheme = 'resist' }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid || grid.length === 0) return;
    const ctx = canvas.getContext('2d');
    const n = grid.length;
    const m = grid[0].length;
    const cellW = width / m;
    const cellH = height / n;
    
    ctx.clearRect(0, 0, width, height);
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        if (colorScheme === 'resist') {
          ctx.fillStyle = grid[i][j] ? '#10b981' : '#1e293b';
        } else {
          // Cross section: gradient
          const depth = i / n;
          ctx.fillStyle = grid[i][j] ? 
            `hsl(160, ${60 - depth * 20}%, ${45 + depth * 10}%)` : 
            `hsl(220, 20%, ${12 + depth * 5}%)`;
        }
        ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [grid, width, height, colorScheme]);
  
  return (
    <div style={{ textAlign: 'center' }}>
      <canvas ref={canvasRef} width={width} height={height} 
        style={{ borderRadius: '4px', border: '1px solid #334155' }} />
      {label && <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px', fontFamily: 'monospace' }}>{label}</div>}
    </div>
  );
}

// ============================================================
// SLIDER COMPONENT
// ============================================================
function ParamSlider({ label, value, min, max, step, onChange, unit = '', color = '#3b82f6' }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'monospace', color: '#94a3b8', marginBottom: '3px' }}>
        <span>{label}</span>
        <span style={{ color }}>{typeof value === 'number' ? (step < 0.01 ? value.toExponential(1) : value.toFixed(step < 1 ? 2 : 0)) : value}{unit && ` ${unit}`}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', height: '4px', accentColor: color, cursor: 'pointer' }} />
    </div>
  );
}

// ============================================================
// METRIC CARD
// ============================================================
function MetricCard({ label, value, unit, status, description }) {
  const colors = { good: '#10b981', warn: '#f59e0b', bad: '#ef4444', neutral: '#3b82f6' };
  return (
    <div style={{
      background: '#0f172a', border: `1px solid ${colors[status] || colors.neutral}33`,
      borderRadius: '8px', padding: '10px 12px', flex: '1', minWidth: '130px'
    }}>
      <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
      <div style={{ fontSize: '20px', fontWeight: '700', color: colors[status] || colors.neutral, fontFamily: "'JetBrains Mono', monospace", marginTop: '2px' }}>
        {value}
      </div>
      <div style={{ fontSize: '10px', color: '#475569', fontFamily: 'monospace' }}>{unit}</div>
      {description && <div style={{ fontSize: '9px', color: '#334155', marginTop: '3px' }}>{description}</div>}
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function EUVSimulator() {
  // Process parameters
  const [temperature, setTemperature] = useState(295);
  const [relativeHumidity, setRelativeHumidity] = useState(0.40);
  const [pO2, setPO2] = useState(0.21);
  const [agingTime, setAgingTime] = useState(2);
  
  // Material parameters
  const [nSn, setNSn] = useState(12);
  const [moleculeSize, setMoleculeSize] = useState(1.2);
  const [fOH0, setFOH0] = useState(0.15);
  const [absCoeff, setAbsCoeff] = useState(0.02);
  
  // Lithography parameters
  const [halfPitch, setHalfPitch] = useState(14);
  const [NA, setNA] = useState(0.33);
  const [dose, setDose] = useState(30);
  const [blur, setBlur] = useState(2.5);
  const [seEnergy, setSeEnergy] = useState(15);
  
  // Threshold parameters
  const [thRE, setThRE] = useState(2);
  const [thMSS, setThMSS] = useState(14);
  const [thSCF, setThSCF] = useState(14);
  const [overDispersion, setOverDispersion] = useState(0.15);
  
  // Pattern type
  const [patternType, setPatternType] = useState('line_space');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Aging kinetics parameters (with defaults from the uploaded document)
  const agingParams = useMemo(() => ({
    T: temperature, RH: relativeHumidity, pO2, Kads: 15, fOH0, rMol0: moleculeSize * 0.5, nSn,
    A_A: 2.5e8, Ea_A: 0.55, A_B: 5.0e6, Ea_B: 0.40, A_C: 1.0e12, Ea_C: 0.85,
    A_D: 1.0e9, Ea_D: 0.30, A_D2: 1.0e8, Ea_D2: 0.95,
    kE_prime: 1e-3, Ea_Diff: 0.35, alpha: 0.1
  }), [temperature, relativeHumidity, pO2, fOH0, moleculeSize, nSn]);

  // Solve aging ODE
  const agingHistory = useMemo(() => {
    const maxTime = Math.max(agingTime, 1);
    const dt = Math.max(0.01, maxTime / 500);
    return solveAgingODE(agingParams, dt, maxTime);
  }, [agingParams, agingTime]);

  // Get current aging state
  const currentAgingState = useMemo(() => {
    if (!agingHistory || agingHistory.length === 0) return null;
    return agingHistory[agingHistory.length - 1];
  }, [agingHistory]);

  // Compute pattern profile
  const filmThickness = moleculeSize * 10;
  const patternResult = useMemo(() => {
    return computePatternProfile({
      halfPitch, NA, lambda: 13.5, moleculeSize, filmThickness,
      dose, blur, thRE, thMSS, thSCF, absCoeff, seEnergy,
      overDispersion, agingState: currentAgingState
    });
  }, [halfPitch, NA, moleculeSize, filmThickness, dose, blur, thRE, thMSS, thSCF, absCoeff, seEnergy, overDispersion, currentAgingState]);

  // Compute metrics
  const metrics = useMemo(() => {
    return computeMetrics(patternResult.profileData, halfPitch, moleculeSize, dose, filmThickness);
  }, [patternResult, halfPitch, moleculeSize, dose, filmThickness]);

  // Generate 2D pattern
  const patternGrid = useMemo(() => {
    return generatePattern2D(patternType, 120, halfPitch, metrics, currentAgingState, moleculeSize);
  }, [patternType, halfPitch, metrics, currentAgingState, moleculeSize]);

  // Generate cross-section
  const crossSection = useMemo(() => {
    return generateCrossSection(halfPitch, metrics, currentAgingState, filmThickness, moleculeSize);
  }, [halfPitch, metrics, currentAgingState, filmThickness, moleculeSize]);

  const k1Factor = halfPitch / (13.5 / NA);

  const getStatus = (metric, thresholds) => {
    if (metric <= thresholds[0]) return 'good';
    if (metric <= thresholds[1]) return 'warn';
    return 'bad';
  };

  return (
    <div style={{
      minHeight: '100vh', background: '#020617', color: '#e2e8f0',
      fontFamily: "'Inter', 'SF Pro Display', -apple-system, sans-serif",
      padding: '0', margin: '0'
    }}>
      {/* HEADER */}
      <div style={{
        background: 'linear-gradient(135deg, #0f172a 0%, #020617 50%, #0c1222 100%)',
        borderBottom: '1px solid #1e293b', padding: '16px 24px',
        display: 'flex', alignItems: 'center', gap: '16px'
      }}>
        <div style={{
          width: '40px', height: '40px', borderRadius: '10px',
          background: 'linear-gradient(135deg, #10b981, #3b82f6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '18px', fontWeight: '900', color: '#020617', fontFamily: 'monospace'
        }}>Sn</div>
        <div>
          <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', letterSpacing: '-0.5px',
            background: 'linear-gradient(90deg, #10b981, #3b82f6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            EUV MOR-PR Stochastic Simulator
          </h1>
          <div style={{ fontSize: '11px', color: '#475569', fontFamily: 'monospace' }}>
            Sn-Oxo Cluster Aging · Coupled ODE · Hierarchical PMF Network · Pattern Visualization
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '4px' }}>
          {['dashboard', 'aging', 'patterns', 'analysis'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              style={{
                padding: '6px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                fontSize: '11px', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.5px',
                background: activeTab === tab ? '#10b981' : '#1e293b',
                color: activeTab === tab ? '#020617' : '#64748b',
                fontWeight: activeTab === tab ? '700' : '500',
                transition: 'all 0.2s'
              }}>{tab}</button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', minHeight: 'calc(100vh - 73px)' }}>
        {/* LEFT PANEL: Controls */}
        <div style={{
          width: '300px', minWidth: '300px', background: '#0f172a',
          borderRight: '1px solid #1e293b', padding: '16px', overflowY: 'auto',
          maxHeight: 'calc(100vh - 73px)'
        }}>
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#10b981', fontFamily: 'monospace', 
            textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '12px',
            paddingBottom: '8px', borderBottom: '1px solid #1e293b' }}>
            ⚙ Process Conditions
          </div>
          
          <ParamSlider label="Temperature" value={temperature} min={273} max={373} step={1} onChange={setTemperature} unit="K" color="#ef4444" />
          <ParamSlider label="Relative Humidity" value={relativeHumidity} min={0} max={1} step={0.01} onChange={setRelativeHumidity} unit="" color="#3b82f6" />
          <ParamSlider label="O₂ Partial Pressure" value={pO2} min={0} max={1} step={0.01} onChange={setPO2} unit="atm" color="#f59e0b" />
          <ParamSlider label="Queue/Aging Time" value={agingTime} min={0} max={168} step={0.5} onChange={setAgingTime} unit="hr" color="#a855f7" />
          
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#3b82f6', fontFamily: 'monospace',
            textTransform: 'uppercase', letterSpacing: '1px', margin: '16px 0 12px',
            paddingBottom: '8px', borderBottom: '1px solid #1e293b' }}>
            🧬 Material (Sn-Oxo Cluster)
          </div>
          
          <ParamSlider label="Sn atoms/cluster" value={nSn} min={3} max={16} step={1} onChange={setNSn} unit="" color="#10b981" />
          <ParamSlider label="Molecule Size" value={moleculeSize} min={0.5} max={3} step={0.1} onChange={setMoleculeSize} unit="nm" color="#10b981" />
          <ParamSlider label="Initial Sn-OH fraction" value={fOH0} min={0.05} max={0.30} step={0.01} onChange={setFOH0} unit="" color="#10b981" />
          <ParamSlider label="Absorption Coeff (μ)" value={absCoeff} min={0.004} max={0.04} step={0.001} onChange={setAbsCoeff} unit="nm⁻¹" color="#10b981" />
          
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#f59e0b', fontFamily: 'monospace',
            textTransform: 'uppercase', letterSpacing: '1px', margin: '16px 0 12px',
            paddingBottom: '8px', borderBottom: '1px solid #1e293b' }}>
            🔬 Lithography
          </div>
          
          <ParamSlider label="Half-Pitch" value={halfPitch} min={5} max={30} step={0.5} onChange={setHalfPitch} unit="nm" color="#f59e0b" />
          <ParamSlider label="NA" value={NA} min={0.25} max={0.75} step={0.01} onChange={setNA} unit="" color="#f59e0b" />
          <ParamSlider label="Dose" value={dose} min={5} max={100} step={1} onChange={setDose} unit="mJ/cm²" color="#f59e0b" />
          <ParamSlider label="SE + Chem Blur" value={blur} min={0.5} max={8} step={0.1} onChange={setBlur} unit="nm" color="#f59e0b" />
          
          <div style={{ marginTop: '8px' }}>
            <button onClick={() => setShowAdvanced(!showAdvanced)}
              style={{ background: 'none', border: '1px solid #334155', borderRadius: '4px', color: '#64748b',
                fontSize: '10px', fontFamily: 'monospace', padding: '4px 8px', cursor: 'pointer', width: '100%' }}>
              {showAdvanced ? '▼' : '▶'} Advanced Thresholds
            </button>
          </div>
          
          {showAdvanced && (
            <div style={{ marginTop: '8px', padding: '8px', background: '#020617', borderRadius: '6px' }}>
              <ParamSlider label="Reaction threshold (thRE)" value={thRE} min={1} max={5} step={1} onChange={setThRE} unit="" color="#8b5cf6" />
              <ParamSlider label="MSS threshold (thMSS)" value={thMSS} min={5} max={22} step={1} onChange={setThMSS} unit="/27" color="#8b5cf6" />
              <ParamSlider label="SCF threshold (thSCF)" value={thSCF} min={5} max={22} step={1} onChange={setThSCF} unit="/27" color="#8b5cf6" />
              <ParamSlider label="Over-dispersion (w)" value={overDispersion} min={0.01} max={0.5} step={0.01} onChange={setOverDispersion} unit="" color="#8b5cf6" />
              <ParamSlider label="SE Energy" value={seEnergy} min={5} max={30} step={1} onChange={setSeEnergy} unit="eV" color="#8b5cf6" />
            </div>
          )}

          <div style={{ fontSize: '11px', fontWeight: '700', color: '#a855f7', fontFamily: 'monospace',
            textTransform: 'uppercase', letterSpacing: '1px', margin: '16px 0 12px',
            paddingBottom: '8px', borderBottom: '1px solid #1e293b' }}>
            📐 Pattern Type
          </div>
          
          <div style={{ display: 'flex', gap: '4px' }}>
            {[['line_space', 'L/S'], ['pillar', 'Pillar'], ['square', 'Square']].map(([val, lbl]) => (
              <button key={val} onClick={() => setPatternType(val)}
                style={{
                  flex: 1, padding: '6px', borderRadius: '4px', border: 'none', cursor: 'pointer',
                  fontSize: '10px', fontFamily: 'monospace', fontWeight: '600',
                  background: patternType === val ? '#a855f7' : '#1e293b',
                  color: patternType === val ? '#fff' : '#64748b'
                }}>{lbl}</button>
            ))}
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div style={{ flex: 1, padding: '16px', overflowY: 'auto', maxHeight: 'calc(100vh - 73px)' }}>
          
          {/* METRICS BAR */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', flexWrap: 'wrap' }}>
            <MetricCard label="LER (3σ)" value={metrics.LER_3sigma.toFixed(2)} unit="nm" 
              status={getStatus(metrics.LER_3sigma, [1.5, 2.5])} />
            <MetricCard label="LWR (3σ)" value={metrics.LWR_3sigma.toFixed(2)} unit="nm"
              status={getStatus(metrics.LWR_3sigma, [2.0, 3.5])} />
            <MetricCard label="CD Error" value={metrics.cdError.toFixed(2)} unit="nm"
              status={getStatus(Math.abs(metrics.cdError), [0.5, 1.5])} />
            <MetricCard label="k₁ factor" value={k1Factor.toFixed(3)} unit={`${halfPitch}nm / (λ/NA)`}
              status={getStatus(k1Factor, [0.35, 0.28].reverse())} />
            <MetricCard label="DtS" value={dose.toFixed(0)} unit="mJ/cm²"
              status={getStatus(dose, [30, 60])} />
            <MetricCard label="DR Log-Slope" value={metrics.drLogSlope.toFixed(1)} unit=""
              status={metrics.drLogSlope > 10 ? 'good' : metrics.drLogSlope > 5 ? 'warn' : 'bad'} />
            <MetricCard label="P(defect)" value={metrics.defectProb < 1e-6 ? '<1e-6' : metrics.defectProb.toExponential(1)} unit=""
              status={metrics.defectProb < 1e-6 ? 'good' : metrics.defectProb < 1e-3 ? 'warn' : 'bad'} />
            <MetricCard label="Process Window" value={(metrics.processWindow * 100).toFixed(0)} unit="%"
              status={metrics.processWindow > 0.7 ? 'good' : metrics.processWindow > 0.4 ? 'warn' : 'bad'} />
          </div>

          {activeTab === 'dashboard' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              {/* Pattern Visualization */}
              <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#10b981', fontFamily: 'monospace', marginBottom: '10px' }}>
                  TOP VIEW — {patternType === 'line_space' ? 'Line/Space' : patternType === 'pillar' ? 'Pillar Array' : 'Square Array'} ({halfPitch}nm hp)
                </div>
                <PatternCanvas grid={patternGrid} width={320} height={320} colorScheme="resist" />
                <div style={{ fontSize: '9px', color: '#475569', fontFamily: 'monospace', marginTop: '4px' }}>
                  Green = resist remaining (negative tone) · Dark = dissolved · Field: {(halfPitch * 4).toFixed(0)} × {(halfPitch * 4).toFixed(0)} nm
                </div>
              </div>

              {/* Cross Section */}
              <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#3b82f6', fontFamily: 'monospace', marginBottom: '10px' }}>
                  CROSS-SECTION VIEW — Film thickness: {filmThickness.toFixed(1)} nm
                </div>
                <PatternCanvas grid={crossSection} width={320} height={160} colorScheme="cross" />
                <div style={{ fontSize: '9px', color: '#475569', fontFamily: 'monospace', marginTop: '4px' }}>
                  Cross-section perpendicular to line edge · Sidewall profile with stochastic roughness
                </div>
                
                <div style={{ marginTop: '12px' }}>
                  <PMFHeatmap profileData={patternResult.profileData} width={320} height={160} nMolThickness={patternResult.nMolThickness} />
                </div>
              </div>

              {/* Profile Plot */}
              <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#f59e0b', fontFamily: 'monospace', marginBottom: '10px' }}>
                  EVENT HIERARCHY — Probability Profiles
                </div>
                <svg width={320} height={200} style={{ display: 'block' }}>
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1].map(v => (
                    <line key={v} x1={40} y1={20 + (1-v)*160} x2={310} y2={20 + (1-v)*160} stroke="#1e293b" strokeWidth="0.5" />
                  ))}
                  {/* Image intensity */}
                  <polyline
                    points={patternResult.profileData.map((d, i) => 
                      `${40 + (i / (patternResult.profileData.length-1)) * 270},${20 + (1 - d.imageIntensity) * 160}`
                    ).join(' ')}
                    fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="4,2"
                  />
                  {/* MSS probability */}
                  <polyline
                    points={patternResult.profileData.map((d, i) => 
                      `${40 + (i / (patternResult.profileData.length-1)) * 270},${20 + (1 - d.pMSS) * 160}`
                    ).join(' ')}
                    fill="none" stroke="#f59e0b" strokeWidth="1.5"
                  />
                  {/* SC1 probability */}
                  <polyline
                    points={patternResult.profileData.map((d, i) => 
                      `${40 + (i / (patternResult.profileData.length-1)) * 270},${20 + (1 - d.pSC1) * 160}`
                    ).join(' ')}
                    fill="none" stroke="#3b82f6" strokeWidth="1.5"
                  />
                  {/* Dissolution rate */}
                  <polyline
                    points={patternResult.profileData.map((d, i) => 
                      `${40 + (i / (patternResult.profileData.length-1)) * 270},${20 + (1 - d.dissolutionRate) * 160}`
                    ).join(' ')}
                    fill="none" stroke="#10b981" strokeWidth="2"
                  />
                  {/* 50% threshold line */}
                  <line x1={40} y1={100} x2={310} y2={100} stroke="#ef4444" strokeWidth="0.5" strokeDasharray="3,3" />
                  <text x={312} y={103} fontSize="8" fill="#ef4444" fontFamily="monospace">th</text>
                  {/* Legend */}
                  {[['Image', '#475569', 8], ['p_MSS', '#f59e0b', 18], ['p_SC', '#3b82f6', 28], ['DR', '#10b981', 38]].map(([l,c,y]) => (
                    <g key={l}><line x1={42} y1={185+0} x2={55} y2={185+0} stroke={c} strokeWidth="2" transform={`translate(0,${y-8})`}/>
                    <text x={58} y={185+0} fontSize="8" fill={c} fontFamily="monospace" transform={`translate(0,${y-5})`}>{l}</text></g>
                  ))}
                  <text x={5} y={100} fontSize="8" fill="#64748b" fontFamily="monospace" transform="rotate(-90,10,100)">Probability</text>
                </svg>
              </div>

              {/* Aging State */}
              <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#a855f7', fontFamily: 'monospace', marginBottom: '10px' }}>
                  AGING STATE @ t = {agingTime.toFixed(1)} hr
                </div>
                {currentAgingState && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '11px', fontFamily: 'monospace' }}>
                    {[
                      ['f(Sn-O-Sn)', currentAgingState.fSnOSn.toFixed(4), currentAgingState.fSnOSn > 0.95 ? '#10b981' : '#ef4444'],
                      ['f(Sn-C)', currentAgingState.fSnC.toFixed(4), currentAgingState.fSnC > 0.9 ? '#10b981' : '#ef4444'],
                      ['f(Sn-OH)', currentAgingState.fSnOH.toFixed(4), '#f59e0b'],
                      ['f(Sn(II))', currentAgingState.fSnII.toFixed(4), '#3b82f6'],
                      ['r_agg', `${currentAgingState.rAgg.toFixed(3)} nm`, currentAgingState.rAgg < moleculeSize ? '#10b981' : '#ef4444'],
                      ['μ_abs/μ₀', currentAgingState.muAbs.toFixed(3), '#a855f7'],
                      ['φ_eff/φ₀', currentAgingState.phiEff.toFixed(3), '#f59e0b'],
                      ['DR contrast', currentAgingState.DRcontrast.toFixed(2), '#10b981'],
                    ].map(([name, val, col]) => (
                      <div key={name} style={{ padding: '4px 6px', background: '#020617', borderRadius: '4px', display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: '#64748b' }}>{name}</span>
                        <span style={{ color: col, fontWeight: '600' }}>{val}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                  <MiniChart data={agingHistory} xKey="time" yKey="fSnOSn" color="#10b981" label="Sn-O-Sn backbone" width={148} height={100} />
                  <MiniChart data={agingHistory} xKey="time" yKey="fSnC" color="#3b82f6" label="Sn-C ligand" width={148} height={100} />
                  <MiniChart data={agingHistory} xKey="time" yKey="LERratio" color="#ef4444" label="LER ratio (aging)" width={148} height={100} />
                  <MiniChart data={agingHistory} xKey="time" yKey="DRcontrast" color="#f59e0b" label="DR contrast" width={148} height={100} />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'aging' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
              {[
                ['fSnOSn', 'Sn-O-Sn Backbone', '#10b981', false],
                ['fSnC', 'Sn-C Ligand Bonds', '#3b82f6', false],
                ['fSnOH', 'Surface Sn-OH', '#f59e0b', false],
                ['fSnII', 'Sn(II) Defects', '#ef4444', false],
                ['rAgg', 'Aggregate Radius (nm)', '#a855f7', false],
                ['muAbs', 'Absorption μ/μ₀', '#ec4899', false],
                ['phiEff', 'Quantum Yield φ/φ₀', '#06b6d4', false],
                ['DRcontrast', 'DR Contrast', '#84cc16', false],
                ['LERratio', 'LER Degradation Factor', '#f97316', false],
                ['NmssRatio', 'N_mss Effective', '#8b5cf6', false],
                ['sigmaDR', 'σ_DR (stochasticity)', '#e11d48', false],
                ['DRslopeRatio', 'DR Slope Ratio', '#14b8a6', false],
              ].map(([key, label, color, log]) => (
                <div key={key} style={{ background: '#0f172a', borderRadius: '8px', padding: '10px', border: '1px solid #1e293b' }}>
                  <MiniChart data={agingHistory} xKey="time" yKey={key} color={color} label={label} width={280} height={140} logScale={log} />
                </div>
              ))}
            </div>
          )}

          {activeTab === 'patterns' && (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                {['line_space', 'pillar', 'square'].map(type => {
                  const g = generatePattern2D(type, 120, halfPitch, metrics, currentAgingState, moleculeSize);
                  return (
                    <div key={type} style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b', textAlign: 'center' }}>
                      <div style={{ fontSize: '12px', fontWeight: '700', color: '#10b981', fontFamily: 'monospace', marginBottom: '8px' }}>
                        {type === 'line_space' ? 'Line/Space' : type === 'pillar' ? 'Cylindrical Pillar' : 'Square Array'}
                      </div>
                      <PatternCanvas grid={g} width={260} height={260} colorScheme="resist" />
                      <div style={{ fontSize: '9px', color: '#475569', fontFamily: 'monospace', marginTop: '4px' }}>
                        hp={halfPitch}nm · LER={metrics.LER_3sigma.toFixed(1)}nm · aging={agingTime}hr
                      </div>
                    </div>
                  );
                })}
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: '#3b82f6', fontFamily: 'monospace', marginBottom: '8px' }}>
                    CROSS-SECTION (L/S)
                  </div>
                  <PatternCanvas grid={crossSection} width={400} height={120} colorScheme="cross" />
                </div>
                <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                  <PMFHeatmap profileData={patternResult.profileData} width={400} height={200} nMolThickness={patternResult.nMolThickness} />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'analysis' && (
            <div>
              {/* RLS Trade-off analysis */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: '#f59e0b', fontFamily: 'monospace', marginBottom: '10px' }}>
                    RLS TRADE-OFF ANALYSIS
                  </div>
                  <div style={{ fontSize: '11px', fontFamily: 'monospace', color: '#94a3b8', lineHeight: '1.8' }}>
                    <div>Resolution: k₁ = <span style={{color:'#10b981', fontWeight:'700'}}>{k1Factor.toFixed(3)}</span> (target {'<'} 0.35)</div>
                    <div>LER (3σ): <span style={{color:'#3b82f6', fontWeight:'700'}}>{metrics.LER_3sigma.toFixed(2)} nm</span> ({(metrics.LER_3sigma/halfPitch*100).toFixed(1)}% of hp)</div>
                    <div>Sensitivity: <span style={{color:'#f59e0b', fontWeight:'700'}}>{dose} mJ/cm²</span></div>
                    <div style={{marginTop:'8px', paddingTop:'8px', borderTop:'1px solid #1e293b'}}>
                      <div>Aging impact on LER: <span style={{color: currentAgingState?.LERratio > 1.2 ? '#ef4444' : '#10b981', fontWeight:'700'}}>×{currentAgingState?.LERratio.toFixed(2)}</span></div>
                      <div>Aging impact on DR slope: <span style={{color: currentAgingState?.DRslopeRatio < 0.8 ? '#ef4444' : '#10b981', fontWeight:'700'}}>{((currentAgingState?.DRslopeRatio || 1) * 100).toFixed(0)}%</span></div>
                      <div>Aggregate growth: <span style={{color:'#a855f7', fontWeight:'700'}}>{currentAgingState?.rAgg.toFixed(3)} nm</span> (init: {(moleculeSize*0.5).toFixed(2)} nm)</div>
                    </div>
                    <div style={{marginTop:'8px', paddingTop:'8px', borderTop:'1px solid #1e293b', fontSize:'10px', color:'#475569'}}>
                      <div>Based on Fukuda (2025) directional network model:</div>
                      <div>• Binomial PMF for molecular solubility switching</div>
                      <div>• Beta-binomial for overdispersion (EUV radiochemistry)</div>
                      <div>• Bernoulli convolution for molecular interaction</div>
                      <div>• Coupled Arrhenius kinetics for 5 degradation channels</div>
                    </div>
                  </div>
                </div>
                
                <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: '#ef4444', fontFamily: 'monospace', marginBottom: '10px' }}>
                    DEGRADATION CHANNELS
                  </div>
                  <div style={{ fontSize: '10px', fontFamily: 'monospace', color: '#94a3b8', lineHeight: '2' }}>
                    {[
                      { name: 'Hydrolysis (Sn-O-Sn)', rate: arrheniusRate(2.5e8, 0.55, temperature) * langmuirCoverage(15, relativeHumidity), color: '#3b82f6', impact: 'CD gain, bridge/break' },
                      { name: 'Condensation (Sn-OH)', rate: arrheniusRate(5e6, 0.40, temperature), color: '#10b981', impact: 'CD loss, LER↑' },
                      { name: 'Ligand loss (Sn-C)', rate: arrheniusRate(1e12, 0.85, temperature) * langmuirCoverage(15, relativeHumidity), color: '#f59e0b', impact: 'Dark loss, fogging' },
                      { name: 'Oxidation (Sn(II))', rate: arrheniusRate(1e9, 0.30, temperature) * pO2, color: '#ef4444', impact: 'Dose drift' },
                      { name: 'Aggregation (LSW)', rate: arrheniusRate(1, 0.35, temperature) * 1e-3, color: '#a855f7', impact: 'LER/LWR↑, defects↑' },
                    ].map(ch => (
                      <div key={ch.name} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: ch.color, flexShrink: 0 }} />
                        <div style={{ flex: 1 }}>
                          <span style={{ color: ch.color }}>{ch.name}</span>
                          <span style={{ color: '#475569' }}> — k={ch.rate.toExponential(1)} s⁻¹</span>
                        </div>
                      </div>
                    ))}
                    <div style={{ marginTop: '8px', padding: '6px', background: '#020617', borderRadius: '4px', fontSize: '9px', color: '#475569' }}>
                      Dominant channel: <span style={{ color: '#f59e0b', fontWeight: '700' }}>
                        {relativeHumidity > 0.5 ? 'Hydrolysis (high RH)' : temperature > 323 ? 'Condensation (high T)' : pO2 > 0.3 ? 'Oxidation' : 'Ligand loss'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Cluster architecture info */}
              <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '12px', fontWeight: '700', color: '#10b981', fontFamily: 'monospace', marginBottom: '10px' }}>
                  MATERIAL: Sn{nSn} OXO NANO-CAGE
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '10px', fontSize: '10px', fontFamily: 'monospace' }}>
                  <div style={{ padding: '8px', background: '#020617', borderRadius: '6px' }}>
                    <div style={{ color: '#64748b' }}>Formula</div>
                    <div style={{ color: '#e2e8f0', fontSize: '11px', fontWeight: '600' }}>[(R-Sn)_{nSn}O₁₄(OH)₆]²⁺</div>
                    <div style={{ color: '#475569', marginTop: '4px' }}>Cage diameter: {(moleculeSize * 1.5).toFixed(1)} nm</div>
                  </div>
                  <div style={{ padding: '8px', background: '#020617', borderRadius: '6px' }}>
                    <div style={{ color: '#64748b' }}>Sn-C BDE</div>
                    <div style={{ color: '#f59e0b', fontSize: '14px', fontWeight: '700' }}>~2.1 eV</div>
                    <div style={{ color: '#475569', marginTop: '4px' }}>Primary photoreactive site</div>
                  </div>
                  <div style={{ padding: '8px', background: '#020617', borderRadius: '6px' }}>
                    <div style={{ color: '#64748b' }}>EUV σ_abs</div>
                    <div style={{ color: '#3b82f6', fontSize: '14px', fontWeight: '700' }}>{(absCoeff * 1e4).toFixed(0)} cm⁻¹</div>
                    <div style={{ color: '#475569', marginTop: '4px' }}>~10× organic CAR</div>
                  </div>
                  <div style={{ padding: '8px', background: '#020617', borderRadius: '6px' }}>
                    <div style={{ color: '#64748b' }}>SE mean free path</div>
                    <div style={{ color: '#a855f7', fontSize: '14px', fontWeight: '700' }}>1-3 nm</div>
                    <div style={{ color: '#475569', marginTop: '4px' }}>Patterning blur origin</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* FOOTER */}
          <div style={{ marginTop: '16px', padding: '12px', background: '#0f172a', borderRadius: '8px', border: '1px solid #1e293b',
            fontSize: '9px', color: '#334155', fontFamily: 'monospace', lineHeight: '1.6' }}>
            <strong style={{ color: '#475569' }}>References & Methods:</strong> Stochastic patterning based on Fukuda, J. Appl. Phys. 137, 204902 (2025) — 
            hierarchical binomial/beta-binomial/Bernoulli convolution PMF network. Aging kinetics: coupled Arrhenius ODE 
            with 5 degradation channels (hydrolysis, condensation, ligand loss, oxidation, LSW aggregation) for Sn-O-Sn MOR PR. 
            권석준 (SKKU) pathfinding research model. All parameters experimentally calibratable. DFT-free, real-time computation.
          </div>
        </div>
      </div>
    </div>
  );
}
