/* =====================================================================
 *  vepmap_hierprior.stan
 * ---------------------------------------------------------------------
 *  Softmax prior + fixed sigma_hier.
 *
 *    1. OBSERVATION : sensors explained by CORTEX ONLY
 *    2. SUBCORTEX x0 : SC-weighted softmax hierarchical prior
 *       Prior center emphasizes the most epileptogenic connected cortex.
 *       sigma_hier is FIXED (data), preventing optimizer collapse.
 *    3. DYNAMICS : FULL circuit (cortex + subcortex) coupled via SC
 *       Dynamics provide indirect likelihood signal for subcortex.
 * ===================================================================== */

functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D;
  }

  row_vector x_step(row_vector x, row_vector z, real I1, real time_step) {
    int nn = num_elements(x);
    row_vector[nn] x_next;
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    x_next = x + (time_step * dx);
    return x_next;
  }

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC,
		    real time_step, real tau0) {
    int nn = num_elements(z);
    row_vector[nn] z_next;
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D));
    row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
    z_next = z + (time_step * dz);
    return z_next;
  }

  real sc_weighted_softmax(row_vector sc_weights, row_vector x0_cortex,
                           real beta, int nc) {
    real max_bx = negative_infinity();
    for (k in 1:nc) {
      if (sc_weights[k] > 1e-8) {
        real bx = beta * x0_cortex[k] + log(sc_weights[k]);
        if (bx > max_bx) max_bx = bx;
      }
    }
    real sum_exp = 0;
    real sum_w = 0;
    for (k in 1:nc) {
      if (sc_weights[k] > 1e-8) {
        sum_exp += exp(beta * x0_cortex[k] + log(sc_weights[k]) - max_bx);
        sum_w += sc_weights[k];
      }
    }
    return (max_bx + log(sum_exp) - log(sum_w)) / beta;
  }
}

data {
  int nn;
  int nc;
  int nsub;
  int ns;
  int nt;

  array[nc] int cortex_idx;
  array[nsub] int subcortex_idx;

  matrix[ns, nc] gain;
  matrix<lower=0.0>[nn, nn] SC;

  matrix<lower=0.0>[nsub, nc] SC_sub_cortex;
  row_vector<lower=0.0>[nsub] SC_sub_cortex_sum;

  array[nt] row_vector[ns] slp;
  row_vector[ns] snsr_pwr;

  row_vector[nn] x0_mu;

  real<lower=0> beta_softmax;
  real<lower=0> sigma_hier;
}

transformed data {
  real I1 = 3.1;
  real time_step = 0.1;
}

parameters {
  row_vector<lower=-4.5, upper=-1.0>[nn] x0;
  real amplitude;
  real offset_param;
  real<lower=0> K;
  real tau0;
  row_vector<lower=-3.2, upper=-0.8>[nn] x_init;
  row_vector<lower=2.5, upper=4.5>[nn] z_init;
  real<lower=0> eps_slp;
  real<lower=0> eps_snsr_pwr;
}

transformed parameters {
  array[nt] row_vector[nn] x;
  array[nt] row_vector[nn] z;
  array[nt] row_vector[ns] mu_slp;
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);

  for (t in 1:nt) {
    if (t == 1) {
      x[t] = x_step(x_init, z_init, I1, time_step);
      z[t] = z_step(x_init, z_init, x0, K * SC, time_step, tau0);
    } else {
      x[t] = x_step(x[t - 1], z[t - 1], I1, time_step);
      z[t] = z_step(x[t - 1], z[t - 1], x0, K * SC, time_step, tau0);
    }

    {
      row_vector[nc] xc = x[t][cortex_idx];
      mu_slp[t] = amplitude * (log(gain * exp(xc)')' + offset_param);
    }
    mu_snsr_pwr += mu_slp[t] .* mu_slp[t];
  }

  mu_snsr_pwr = mu_snsr_pwr / nt;
}

model {
  x0[cortex_idx] ~ normal(x0_mu[cortex_idx], 1.0);

  for (j in 1:nsub) {
    if (SC_sub_cortex_sum[j] > 1e-8) {
      row_vector[nc] x0c = x0[cortex_idx];
      real x0_pred = sc_weighted_softmax(SC_sub_cortex[j], x0c,
                                         beta_softmax, nc);
      x0[subcortex_idx[j]] ~ normal(x0_pred, sigma_hier);
    } else {
      x0[subcortex_idx[j]] ~ normal(x0_mu[subcortex_idx[j]], 1.0);
    }
  }

  amplitude ~ normal(1.0, 10.0);
  offset_param ~ normal(0, 10.0);
  tau0 ~ normal(20, 10.0);
  K ~ normal(1.0, 10.0);
  for (i in 1:nn) {
    x_init[i] ~ normal(-2.0, 10.0);
    z_init[i] ~ normal(3.5, 10.0);
  }
  eps_slp ~ normal(1, 10);
  eps_snsr_pwr ~ normal(1, 10);

  for (t in 1:nt) {
    target += normal_lpdf(slp[t] | mu_slp[t], eps_slp);
  }
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, eps_snsr_pwr);
}

generated quantities {
}
