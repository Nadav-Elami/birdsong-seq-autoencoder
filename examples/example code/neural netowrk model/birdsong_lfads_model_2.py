import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    Compute KL(N(q|mu_q, var_q) || N(p|mu_p, var_p)) for diagonal Gaussians.
    KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1 )
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p)**2) / var_p
        - 1.0
    )
    return kl.sum(dim=-1)  # sum over latent dimension

# def rowwise_softmax(flat_factors, alphabet_size):
#     """
#     Takes a (batch, time, alphabet_size^2) tensor,
#     reshapes to (batch, time, alphabet_size, alphabet_size),
#     applies row-wise softmax over the last dimension,
#     then flattens back to (batch, time, alphabet_size^2).
#
#     Returns a row-wise normalized distribution over each row.
#     """
#     B, T, _ = flat_factors.shape
#     # Reshape to (B, T, alphabet_size, alphabet_size)
#     mat = flat_factors.view(B, T, alphabet_size, alphabet_size)
#     # For each row in [alphabet_size], do softmax over columns
#     # dimension=-1 means softmax over the last dimension
#     mat_softmax = F.softmax(mat, dim=-1)
#     # Flatten back to (B, T, alphabet_size^2)
#     return mat_softmax.view(B, T, alphabet_size**2)


def rowwise_softmax(flat_factors, alphabet_size, order=1):
    """
    Standard rowwise softmax:
    Reshape a (B, T, alphabet_size^(order+1)) tensor to
      (B, T, alphabet_size^order, alphabet_size),
    apply softmax over the last dimension, then flatten back.
    """
    B, T, _ = flat_factors.shape
    new_shape = (B, T, alphabet_size ** order, alphabet_size)
    mat = flat_factors.view(*new_shape)
    mat_softmax = F.softmax(mat, dim=-1)
    return mat_softmax.view(B, T, alphabet_size ** (order + 1))


# def rowwise_masked_softmax(flat_factors, alphabet_size, order=1, mask_value=-1e8, eps=1e-12):
#     """
#     Masked rowwise softmax:
#     Reshape a (B, T, alphabet_size^(order+1)) tensor to
#       (B, T, alphabet_size^order, alphabet_size),
#     then apply softmax over the last dimension. For any "row" (a vector of length alphabet_size)
#     that is entirely equal to mask_value, return a zero vector (instead of a uniform distribution).
#     Finally, flatten back to (B, T, alphabet_size^(order+1)).
#     """
#     B, T, _ = flat_factors.shape
#     new_shape = (B, T, alphabet_size ** order, alphabet_size)
#     mat = flat_factors.view(*new_shape)
#
#     # Flatten the first three dimensions so each "row" is a vector of length alphabet_size.
#     mat_flat = mat.view(-1, alphabet_size)  # shape: (B*T*alphabet_size**order, alphabet_size)
#
#     # Identify rows that are fully masked.
#     fully_masked = (mat_flat == mask_value).all(dim=1)  # shape: (B*T*alphabet_size**order,)
#
#     # Compute standard softmax (this will yield uniform distributions for fully masked rows).
#     soft = F.softmax(mat_flat, dim=1)
#
#     # Instead of modifying 'soft' in-place, use torch.where to replace fully masked rows with zeros.
#     soft = torch.where(fully_masked.unsqueeze(1), torch.zeros_like(soft), soft)
#
#     # Reshape back to (B, T, alphabet_size**order, alphabet_size)
#     soft = soft.view(B, T, alphabet_size ** order, alphabet_size)
#     # Finally, flatten the last two dimensions.
#     return soft.view(B, T, alphabet_size ** (order + 1))



def rowwise_masked_softmax(flat_factors, alphabet_size, order=1, target_matrix=None, mask_value=-1e8, eps=1e-12,
                           tol=1e-3):
    """
    Applies a masked rowwise softmax to the input tensor.

    The input tensor flat_factors has shape (B, T, alphabet_size^(order+1)).
    It is reshaped to (B, T, alphabet_size^order, alphabet_size) so that each row (of length alphabet_size)
    corresponds to the candidate next symbols given a context.

    If target_matrix is provided (of shape (alphabet_size^order, alphabet_size)), we derive a Boolean mask by
    checking which rows sum to (approximately) 1. For each row i:
      - If target_matrix[i].sum() > 1 - tol, then we mark that row as allowed (all True)
      - Otherwise, that entire row is disallowed (False)

    Any disallowed position in the logits is then replaced with mask_value before softmax is applied.
    Also, if an entire row is masked, the function returns a zero vector for that row (instead of a uniform distribution).

    Finally, the result is flattened back to shape (B, T, alphabet_size^(order+1)).

    Args:
        flat_factors (torch.Tensor): Input tensor of shape (B, T, alphabet_size^(order+1)).
        alphabet_size (int): The size of the alphabet.
        order (int): Markov order (e.g. 1 for bigrams, 2 for trigrams).
        target_matrix (torch.Tensor, optional): A tensor of shape (alphabet_size^order, alphabet_size)
                                                  that represents the true transition matrix.
                                                  Rows that do not sum to ~1 are considered masked.
        mask_value (float): The value to assign to disallowed positions (default: -1e8).
        eps (float): A small value to avoid log(0) (default: 1e-12).
        tol (float): Tolerance for checking row-sum (default: 1e-3).

    Returns:
        torch.Tensor: A tensor of shape (B, T, alphabet_size^(order+1)) representing the rowwise
                      normalized (masked) probabilities.
    """
    B, T, _ = flat_factors.shape
    new_shape = (B, T, alphabet_size ** order, alphabet_size)
    mat = flat_factors.view(*new_shape)

    # If a target_matrix is provided, derive the mask from it.
    if target_matrix is not None:
        # target_matrix should be of shape (alphabet_size**order, alphabet_size)
        # Compute the row sums.
        row_sums = target_matrix.sum(dim=1)
        # For each row, allow it only if its sum is close to 1.
        allowed_rows = row_sums > (1 - tol)
        # Create a Boolean mask where for each row i:
        # if allowed_rows[i] is True, then allow all columns; otherwise, mark all as False.
        derived_mask = allowed_rows.unsqueeze(1).expand_as(target_matrix)
        # We'll use this mask in place of any external mask.
        mask = derived_mask
    else:
        # If no target_matrix is provided, you could pass in an external mask.
        mask = None

    # If a mask is provided (either derived or external), apply it.
    if mask is not None:
        # We need the mask to have shape (alphabet_size**order, alphabet_size).
        # Expand it to (1, 1, alphabet_size**order, alphabet_size) to broadcast over B and T.
        mask = mask.unsqueeze(0).unsqueeze(0)
        mat = mat.masked_fill(~mask, mask_value)

    # Flatten the first three dimensions so that each row is of length alphabet_size.
    mat_flat = mat.view(-1, alphabet_size)

    # Identify rows that are entirely masked.
    fully_masked = (mat_flat == mask_value).all(dim=1)

    # Compute standard softmax.
    soft = F.softmax(mat_flat, dim=1)

    # Replace fully masked rows with zeros.
    soft = torch.where(fully_masked.unsqueeze(1), torch.zeros_like(soft), soft)

    # Reshape back to (B, T, alphabet_size**order, alphabet_size)
    soft = soft.view(B, T, alphabet_size ** order, alphabet_size)
    # Flatten the last two dimensions.
    return soft.view(B, T, alphabet_size ** (order + 1))

def reconstruction_loss_rowwise(pred_dist, target, alphabet_size, eps=1e-12):
    """
    Example reconstruction loss where:
      - pred_dist is row-wise normalized (batch, time, alpha^2).
      - target is (batch, time, alpha^2) as well,
        presumably also row-wise normalized *or* a one-hot per row.

    Here we do a cross-entropy–like loss row by row:
       CE = - sum_{row} [ target_row * log(pred_dist_row) ]
    Then sum across rows, average across batch/time.
    """
    # Both pred_dist, target -> (B, T, alpha^2)
    # We'll reshape to (B*T*alpha, alpha) if we want rowwise, but
    # it's easier to do row by row in a single expression:

    # Clip pred_dist to avoid log(0)
    pred_dist = torch.clamp(pred_dist, eps, 1.0)
    # - target * log(pred_dist)
    ce = -target * torch.log(pred_dist)
    # Sum over the alpha^2 dimension, then average batch/time
    return ce.sum(dim=-1).mean()

class BirdsongLFADSModel2(nn.Module):
    """
    LFADS-style model for birdsong data, but with a rowwise softmax output
    instead of Poisson. The final "rates" become distributions over each row
    of an alphabet_size x alphabet_size matrix.
    """
    def __init__(
        self,
        alphabet_size,       # e.g. 7
        order =1,
        encoder_dim=64,      # per-direction for the g0 encoder
        controller_dim=64,   # per-direction for the controller encoder
        generator_dim=64,    # generator RNN hidden size
        factor_dim=32,       # dimension of factors f_t
        latent_dim=16,       # dimension of g0
        inferred_input_dim=8,# dimension of u_t
        kappa=1.0,           # prior variance scaling for g0
        ar_step_size=0.99,   # alpha in AR(1)
        ar_process_var=0.1   # sigma^2_e in AR(1)
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        # self.bigram_dim = alphabet_size ** 2
        self.order = order  # new parameter
        # The output dimension is alphabet_size^(order+1):
        self.out_dim = alphabet_size ** (order + 1)
        # For order==1, this is bigram_dim; for order==2, trigram_dim
        self.bigram_dim = self.out_dim

        self.latent_dim = latent_dim
        self.inferred_input_dim = inferred_input_dim
        self.factor_dim = factor_dim
        self.generator_dim = generator_dim
        self.kappa = kappa

        #----------------------------------------------------------------------
        # 1) g0 Encoder (bidirectional)
        #----------------------------------------------------------------------
        self.g0_encoder = nn.GRU(
            input_size=self.bigram_dim,
            hidden_size=encoder_dim,
            batch_first=True,
            bidirectional=True
        )
        self.g0_linear = nn.Linear(2 * encoder_dim, 2 * latent_dim)

        #----------------------------------------------------------------------
        # 2) Controller RNN stack to infer u_t
        #----------------------------------------------------------------------
        self.controller_encoder = nn.GRU(
            input_size=self.bigram_dim,
            hidden_size=controller_dim,
            batch_first=True,
            bidirectional=True
        )
        self.controller_rnn = nn.GRU(
            input_size=2 * controller_dim + factor_dim,
            hidden_size=controller_dim,
            batch_first=True
        )
        self.u_linear = nn.Linear(controller_dim, 2 * inferred_input_dim)

        #----------------------------------------------------------------------
        # 3) Generator RNN
        #----------------------------------------------------------------------
        # Unidirectional
        # self.latent_to_gen = nn.Linear(latent_dim, generator_dim)
        # self.generator_rnn = nn.GRU(
        #     input_size=inferred_input_dim,
        #     hidden_size=generator_dim,
        #     batch_first=True
        # )
        # self.factors_linear = nn.Linear(generator_dim, factor_dim)

        # Bidirectional
        # We'll define TWO separate linear layers from g0_fwd, g0_bwd
        # to the forward and backward hidden states, respectively.
        half_latent = latent_dim // 2  # must be integer!
        self.latent_to_gen_fwd = nn.Linear(half_latent, generator_dim)
        self.latent_to_gen_bwd = nn.Linear(half_latent, generator_dim)

        self.generator_rnn = nn.GRU(
            input_size=inferred_input_dim,
            hidden_size=generator_dim,
            batch_first=True,
            bidirectional=True
        )
        # factors need to map from 2*generator_dim => factor_dim
        self.factors_linear = nn.Linear(2 * generator_dim, factor_dim)
        self.rate_linear = nn.Linear(factor_dim, self.bigram_dim)  # Now "logits"

        # AR(1) prior
        self.register_buffer("alpha_ar", torch.tensor(ar_step_size, dtype=torch.float))
        self.register_buffer("sigma2_e", torch.tensor(ar_process_var, dtype=torch.float))

    def encode_g0(self, x):
        """
        Bidirectional pass over entire sequence to produce mu_g0, logvar_g0.
        x: (batch, time, bigram_dim)
        """
        _, hidden = self.g0_encoder(x)
        h_cat = torch.cat([hidden[0], hidden[1]], dim=-1)  # (batch, 2*encoder_dim)
        g0_params = self.g0_linear(h_cat)  # (batch, 2*latent_dim)
        mu_g0, logvar_g0 = torch.chunk(g0_params, 2, dim=-1)
        return mu_g0, logvar_g0

    def sample_g0(self, mu_g0, logvar_g0):
        std = torch.exp(0.5 * logvar_g0)
        eps = torch.randn_like(std)
        return mu_g0 + eps * std

    def encode_u(self, x, factors):
        """
        Produce mu_u, logvar_u for each time step via:
          1) Bidirectional encoder -> e^con_t
          2) Forward controller RNN over t=1..T with input [e^con_t, f_{t-1}].
        """
        enc_out, _ = self.controller_encoder(x)  # (batch, time, 2*ctrl_dim)
        B, T, _ = enc_out.shape

        # Shift factors by one step so that f_{t-1} lines up with x_t
        f_shift = torch.zeros_like(factors)
        f_shift[:, 1:] = factors[:, :-1]

        # inputs to controller = cat(enc_out_t, f_shift_t)
        ctrl_in = torch.cat([enc_out, f_shift], dim=-1)  # (B, T, 2*ctrl_dim + factor_dim)
        ctrl_out, _ = self.controller_rnn(ctrl_in)       # (B, T, controller_dim)
        u_params = self.u_linear(ctrl_out)               # (B, T, 2*inferred_input_dim)
        mu_u, logvar_u = torch.chunk(u_params, 2, dim=-1)
        return mu_u, logvar_u

    def sample_u(self, mu_u, logvar_u):
        std = torch.exp(0.5 * logvar_u)
        eps = torch.randn_like(std)
        return mu_u + eps * std

    # Unidirectional
    # def forward_generator(self, g0, u):
    #     """
    #     Run generator RNN forward given g0 and sequence u.
    #     g_t = RNN^gen(g_{t-1}, u_t)
    #     f_t = W^fac(g_t)
    #     'logits' = W^rate(f_t) => rowwise softmax for final predicted distribution
    #     """
    #     g0_init = self.latent_to_gen(g0).unsqueeze(0)  # (1, B, generator_dim)
    #     gen_out, _ = self.generator_rnn(u, g0_init)    # (B, T, generator_dim)
    #     f_t = self.factors_linear(gen_out)             # (B, T, factor_dim)
    #
    #     # We'll call this "logits" but we'll do a rowwise softmax next
    #     logits = self.rate_linear(f_t)                 # (B, T, bigram_dim)
    #     return f_t, logits

    def forward_generator(self, g0, u):
        """
        Bidirectional generator RNN:
          1) Split g0 => (forward half, backward half)
          2) Transform each half => forward init, backward init
          3) Concatenate => shape (2, B, generator_dim)
          4) Run bidirectional generator => (B, T, 2*generator_dim)
          5) factors => W_factors(gen_out), shape (B, T, factor_dim)
          6) logits => W_rate(factors)
        """
        B, LD = g0.shape
        LD_half = LD // 2
        # e.g. if latent_dim=16 => LD_half=8 => g0_fwd=..., g0_bwd=...

        g0_fwd = g0[:, :LD_half]  # (B, LD_half)
        g0_bwd = g0[:, LD_half:]  # (B, LD_half)

        # Transform each half
        init_fwd = self.latent_to_gen_fwd(g0_fwd).unsqueeze(0)  # (1, B, generator_dim)
        init_bwd = self.latent_to_gen_bwd(g0_bwd).unsqueeze(0)  # (1, B, generator_dim)

        # shape => (2, B, generator_dim)
        g0_init = torch.cat([init_fwd, init_bwd], dim=0)

        # generator_rnn => bidirectional => output shape => (B, T, 2*generator_dim)
        gen_out, _ = self.generator_rnn(u, g0_init)

        # map 2*gdim => factor_dim
        f_t = self.factors_linear(gen_out)  # (B, T, factor_dim)

        # Some non linearity
        #f_t = torch.relu(f_t)


        logits = self.rate_linear(f_t)  # (B, T, bigram_dim)



        return f_t, logits

    def forward(self, x):
        """
        Full forward pass: returns a dict with 'logits' as the un-normalized
        rowwise outputs, which you'll then softmax rowwise.
        """
        # 1) Encode g0
        mu_g0, logvar_g0 = self.encode_g0(x)
        g0_post = self.sample_g0(mu_g0, logvar_g0)

        # 2) Quick first pass with u=0 -> approximate factors
        B, T, _ = x.shape
        u_init = torch.zeros(B, T, self.inferred_input_dim, device=x.device)
        f_init, _ = self.forward_generator(g0_post, u_init)

        # 3) Encode u
        mu_u, logvar_u = self.encode_u(x, f_init)
        u_post = self.sample_u(mu_u, logvar_u)

        # 4) Final generator pass
        f_t, logits = self.forward_generator(g0_post, u_post)

        return {
            "logits": logits,  # (B, T, alpha^2) un-normalized
            "factors": f_t,
            "mu_g0": mu_g0,
            "logvar_g0": logvar_g0,
            "mu_u": mu_u,
            "logvar_u": logvar_u
        }

    # def compute_loss(self, x, outputs):
    #     """
    #     - Let pred_dist = rowwise_softmax(outputs['logits']).
    #     - Compare pred_dist to x with some reconstruction loss.
    #     - Then add KL for g0 and u vs. their priors (AR(1) for u).
    #     """
    #     # 1) rowwise-softmax the logits
    #     logits = outputs["logits"]
    #     # pred_dist = rowwise_softmax(logits, self.alphabet_size)  # (B, T, alpha^2)
    #     if self.order == 1:
    #         pred_dist = rowwise_softmax(logits, self.alphabet_size, order=self.order)
    #     # elif self.order == 2:
    #     #     pred_dist = rowwise_masked_softmax(logits, self.alphabet_size, order=self.order, mask_value=-1e8)
    #     elif self.order == 2:
    #         pred_dist = rowwise_masked_softmax(
    #             logits,
    #             self.alphabet_size,
    #             order=self.order,
    #             target_matrix=self.true_transition,  # derived mask from true_transition
    #             mask_value=-1e8
    #         )
    #     else:
    #         raise ValueError("Unsupported order. Use order==1 or order==2.")
    #
    #     # 2) Reconstruction loss
    #     #    If x is also a rowwise distribution (batch, time, alpha^2), we can do cross-entropy:
    #     rec_loss = reconstruction_loss_rowwise(pred_dist, x, self.alphabet_size)
    #
    #     # 3) KL for g0
    #     mu_g0 = outputs["mu_g0"]
    #     logvar_g0 = outputs["logvar_g0"]
    #     mu_p_g0 = torch.zeros_like(mu_g0)
    #     logvar_p_g0 = torch.log(torch.full_like(mu_g0, self.kappa))
    #     kl_g0 = kl_div_gaussian(mu_g0, logvar_g0, mu_p_g0, logvar_p_g0).mean()
    #
    #     # 4) KL for u (AR(1) prior)
    #     mu_u = outputs["mu_u"]       # (B, T, inferred_dim)
    #     logvar_u = outputs["logvar_u"]
    #     B, T, dU = mu_u.shape
    #
    #     alpha = self.alpha_ar
    #     sigma2_e = self.sigma2_e
    #     sigma2_p = sigma2_e / (1.0 - alpha**2)  # stationary var
    #
    #     # t=0 => u1 ~ N(0, sigma2_p)
    #     mu_u1_p = torch.zeros_like(mu_u[:, 0])
    #     logvar_u1_p = torch.log(torch.full_like(mu_u[:, 0], sigma2_p))
    #     kl_u1 = kl_div_gaussian(mu_u[:, 0], logvar_u[:, 0],
    #                             mu_u1_p, logvar_u1_p).mean()
    #
    #     # t=1..T-1 => N(alpha*u_{t-1}, sigma2_e)
    #     kl_ut = 0.0
    #     for t in range(1, T):
    #         prior_mean = alpha * mu_u[:, t - 1]
    #         prior_logvar = torch.log(torch.full_like(prior_mean, sigma2_e))
    #         kl_t = kl_div_gaussian(mu_u[:, t], logvar_u[:, t],
    #                                prior_mean, prior_logvar)
    #         kl_ut += kl_t.mean()
    #
    #     kl_u = kl_u1 + kl_ut
    #
    #     # 5) Total loss
    #     loss = rec_loss + kl_g0 + kl_u
    #     return loss, {
    #         # return them as TENSORS
    #         "rec_loss": rec_loss,
    #         "kl_g0": kl_g0,
    #         "kl_u": kl_u
    #     }
    ## def compute_loss(self, x, outputs):
    #     """
    #     Compute the reconstruction loss and KL terms.
    #     x: target distribution tensor of shape (B, T, alphabet_size^(order+1)).
    #        It should be row-normalized when reshaped to (B, T, alphabet_size**order, alphabet_size).
    #     outputs: dict from the model containing "logits" among other keys.
    #
    #     This function derives a mask from x by reshaping x to
    #        (B, T, alphabet_size**order, alphabet_size)
    #     and then for each row checking if it sums to 1 (within tolerance).
    #     Rows that do not sum to 1 are masked out.
    #
    #     The masked softmax is then applied to the model’s logits.
    #     """
    #     # x: (B, T, alphabet_size^(order+1))
    #     B, T, flat_dim = x.shape
    #     # Reshape x to (B, T, alphabet_size**order, alphabet_size)
    #     new_shape = (B, T, self.alphabet_size ** self.order, self.alphabet_size)
    #     target_matrix = x.view(*new_shape)
    #
    #     # Derive a mask from target_matrix.
    #     # For each row (i.e. each vector of length alphabet_size), check if the row-sum is ~1.
    #     tol = 1e-3
    #     row_sums = target_matrix.sum(dim=-1)
    #     # Allowed rows have sums greater than (1 - tol); rows with lower sum are considered masked.
    #     mask = row_sums > (1 - tol)
    #     # mask now has shape (B, T, alphabet_size**order). For softmax we need a mask of shape (alphabet_size**order, alphabet_size)
    #     # per instance. We assume the same mask applies for all examples in the batch and time steps.
    #     # For simplicity, here we will take the mask from the first batch and first time step.
    #     derived_mask = mask[0, 0]  # shape: (alphabet_size**order,)
    #     # Expand this mask to (alphabet_size**order, alphabet_size): allow entire rows if allowed, else mask them.
    #     # Here, if a row is not allowed, we set all positions in that row to False.
    #     # For each row i in derived_mask, if derived_mask[i] is True, then the row is allowed.
    #     # Otherwise, it is masked entirely.
    #     final_mask = derived_mask.unsqueeze(1).expand(self.alphabet_size ** self.order, self.alphabet_size)
    #
    #     # Now compute predicted distribution.
    #     if self.order == 1:
    #         pred_dist = rowwise_softmax(outputs["logits"], self.alphabet_size, order=self.order)
    #     elif self.order == 2:
    #         pred_dist = rowwise_masked_softmax(
    #             outputs["logits"],
    #             self.alphabet_size,
    #             order=self.order,
    #             target_matrix=final_mask,  # pass in the derived mask
    #             mask_value=-1e8
    #         )
    #     else:
    #         raise ValueError("Unsupported order. Use order==1 or order==2.")
    #
    #     # Now compute reconstruction loss (e.g., cross-entropy rowwise).
    #     pred_dist = torch.clamp(pred_dist, 1e-12, 1.0)
    #     ce = -x * torch.log(pred_dist)
    #     rec_loss = ce.sum(dim=-1).mean()
    #
    #     # KL for g0
    #     mu_g0 = outputs["mu_g0"]
    #     logvar_g0 = outputs["logvar_g0"]
    #     mu_p_g0 = torch.zeros_like(mu_g0)
    #     logvar_p_g0 = torch.log(torch.full_like(mu_g0, self.kappa))
    #     kl_g0 = kl_div_gaussian(mu_g0, logvar_g0, mu_p_g0, logvar_p_g0).mean()
    #
    #     # KL for u (AR(1) prior)
    #     mu_u = outputs["mu_u"]
    #     logvar_u = outputs["logvar_u"]
    #     B, T, dU = mu_u.shape
    #     alpha = self.alpha_ar
    #     sigma2_e = self.sigma2_e
    #     sigma2_p = sigma2_e / (1.0 - alpha ** 2)
    #     mu_u1_p = torch.zeros_like(mu_u[:, 0])
    #     logvar_u1_p = torch.log(torch.full_like(mu_u[:, 0], sigma2_p))
    #     kl_u1 = kl_div_gaussian(mu_u[:, 0], logvar_u[:, 0],
    #                             mu_u1_p, logvar_u1_p).mean()
    #     kl_ut = 0.0
    #     for t in range(1, T):
    #         prior_mean = alpha * mu_u[:, t - 1]
    #         prior_logvar = torch.log(torch.full_like(prior_mean, sigma2_e))
    #         kl_t = kl_div_gaussian(mu_u[:, t], logvar_u[:, t],
    #                                prior_mean, prior_logvar)
    #         kl_ut += kl_t.mean()
    #     kl_u = kl_u1 + kl_ut
    #
    #     loss = rec_loss + kl_g0 + kl_u
    #     return loss, {"rec_loss": rec_loss, "kl_g0": kl_g0, "kl_u": kl_u}

    # def compute_loss(self, x, outputs=None, num_samples=1):
    #     """
    #     Compute the total loss (reconstruction loss + KL divergences) for target x,
    #     efficiently approximating the expectation over latent variables by drawing multiple
    #     latent samples using the reparameterization trick and running the generator in a
    #     single, vectorized pass.
    #
    #     x: target tensor of shape (B, T, alphabet_size^(order+1)); assumed to be a rowwise
    #        probability distribution.
    #     outputs: (optional) dictionary from self.forward(x) containing "mu_g0", "logvar_g0",
    #              "mu_u", "logvar_u", etc. If provided, these will be used to sample latent
    #              variables; otherwise, the encoder is run.
    #     num_samples: number of latent samples to draw.
    #     """
    #     B, T, _ = x.shape
    #     device = next(self.parameters()).device
    #
    #     # --- 1. Get latent posterior parameters (use provided outputs if available) ---
    #     if outputs is None:
    #         # Run the encoder once on x to get the posterior parameters.
    #         mu_g0, logvar_g0 = self.encode_g0(x)
    #         # For u, we need an approximate factor state; use a quick generator pass with u=0.
    #         g0_sample = self.sample_g0(mu_g0, logvar_g0)
    #         u_init = torch.zeros(B, T, self.inferred_input_dim, device=device)
    #         f_init, _ = self.forward_generator(g0_sample, u_init)
    #         mu_u, logvar_u = self.encode_u(x, f_init)
    #     else:
    #         mu_g0, logvar_g0 = outputs["mu_g0"], outputs["logvar_g0"]
    #         mu_u, logvar_u = outputs["mu_u"], outputs["logvar_u"]
    #
    #     # --- 2. Compute KL divergences (analytically) ---
    #     # KL for g₀:
    #     mu_p_g0 = torch.zeros_like(mu_g0)
    #     logvar_p_g0 = torch.log(torch.full_like(mu_g0, self.kappa))
    #     kl_g0 = kl_div_gaussian(mu_g0, logvar_g0, mu_p_g0, logvar_p_g0).mean()
    #
    #     # KL for u (using AR(1) prior):
    #     alpha = self.alpha_ar
    #     sigma2_e = self.sigma2_e
    #     sigma2_p = sigma2_e / (1.0 - alpha ** 2)
    #     mu_u1_p = torch.zeros_like(mu_u[:, 0])
    #     logvar_u1_p = torch.log(torch.full_like(mu_u[:, 0], sigma2_p))
    #     kl_u1 = kl_div_gaussian(mu_u[:, 0], logvar_u[:, 0], mu_u1_p, logvar_u1_p).mean()
    #     kl_ut = 0.0
    #     for t in range(1, T):
    #         prior_mean = alpha * mu_u[:, t - 1]
    #         prior_logvar = torch.log(torch.full_like(prior_mean, sigma2_e))
    #         kl_t = kl_div_gaussian(mu_u[:, t], logvar_u[:, t], prior_mean, prior_logvar)
    #         kl_ut += kl_t.mean()
    #     kl_u = kl_u1 + kl_ut
    #
    #     # --- 3. Draw multiple latent samples (vectorized) using the reparameterization trick ---
    #     std_g0 = torch.exp(0.5 * logvar_g0)
    #     eps_g0 = torch.randn(num_samples, *std_g0.shape, device=device)
    #     # g0_samples: shape (num_samples, B, latent_dim)
    #     g0_samples = mu_g0.unsqueeze(0) + eps_g0 * std_g0.unsqueeze(0)
    #
    #     std_u = torch.exp(0.5 * logvar_u)
    #     eps_u = torch.randn(num_samples, *std_u.shape, device=device)
    #     # u_samples: shape (num_samples, B, T, inferred_input_dim)
    #     u_samples = mu_u.unsqueeze(0) + eps_u * std_u.unsqueeze(0)
    #
    #     # --- 4. Run the generator on all latent samples in one batched pass ---
    #     # Merge the num_samples and batch dimensions.
    #     g0_samples_flat = g0_samples.reshape(num_samples * B, -1)  # (num_samples*B, latent_dim)
    #     u_samples_flat = u_samples.reshape(num_samples * B, T, -1)  # (num_samples*B, T, inferred_input_dim)
    #     f_samples, logits_samples = self.forward_generator(g0_samples_flat, u_samples_flat)
    #     # Reshape logits to (num_samples, B, T, bigram_dim)
    #     logits_samples = logits_samples.view(num_samples, B, T, -1)
    #
    #     # --- 5. Compute reconstruction loss (cross entropy) ---
    #     # Derive a mask from x (assumed rowwise normalized distribution)
    #     new_shape = (B, T, self.alphabet_size ** self.order, self.alphabet_size)
    #     target_matrix = x.view(*new_shape)
    #     tol = 1e-3
    #     row_sums = target_matrix.sum(dim=-1)
    #     mask = row_sums > (1 - tol)
    #     derived_mask = mask[0, 0]  # use first sample as representative
    #     final_mask = derived_mask.unsqueeze(1).expand(self.alphabet_size ** self.order, self.alphabet_size)
    #
    #     # Compute predicted distributions using the appropriate rowwise softmax:
    #     if self.order == 1:
    #         pred_dist = rowwise_softmax(logits_samples.view(num_samples * B, T, -1),
    #                                     self.alphabet_size, order=self.order)
    #         pred_dist = pred_dist.view(num_samples, B, T, -1)
    #     elif self.order == 2:
    #         pred_dist = rowwise_masked_softmax(
    #             logits_samples.view(num_samples * B, T, -1),
    #             self.alphabet_size,
    #             order=self.order,
    #             target_matrix=final_mask,
    #             mask_value=-1e8
    #         )
    #         pred_dist = pred_dist.view(num_samples, B, T, -1)
    #     else:
    #         raise ValueError("Unsupported order. Use order==1 or order==2.")
    #
    #     pred_dist = torch.clamp(pred_dist, 1e-12, 1.0)
    #     # Expand x to include sample dimension: (num_samples, B, T, bigram_dim)
    #     x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1, -1)
    #     ce = -x_expanded * torch.log(pred_dist)
    #     rec_loss = ce.sum(dim=-1).mean()  # average over samples, batch, time
    #
    #     # --- 6. Total loss ---
    #     total_loss = rec_loss + kl_g0 + kl_u
    #     return total_loss, {"rec_loss": rec_loss, "kl_g0": kl_g0, "kl_u": kl_u}

    def compute_recurrent_l2_reg(self):
        """
        Compute the L2 regularization term for recurrent weights.
        Only weights that multiply the previous hidden state (i.e. 'weight_hh')
        in the controller and generator GRUs are included.
        """
        l2_reg = 0.0
        # Regularize controller RNN recurrent weights
        for name, param in self.controller_rnn.named_parameters():
            if "weight_hh" in name:
                l2_reg += torch.sum(param ** 2)
        # Regularize generator RNN recurrent weights
        for name, param in self.generator_rnn.named_parameters():
            if "weight_hh" in name:
                l2_reg += torch.sum(param ** 2)
        return l2_reg

    def compute_loss(self, x, outputs=None, num_samples=1):
        """
        Compute the total loss and return individual loss components.
        Returns a tuple: (total_loss, loss_dict)

        The loss_dict includes:
          - rec_loss: reconstruction loss (cross-entropy)
          - kl_g0: KL divergence for the initial condition (g0)
          - kl_u: KL divergence for the inferred inputs (u)
          - l2_loss: L2 regularization term for recurrent weights (hidden-to-hidden)
        """
        B, T, _ = x.shape
        device = next(self.parameters()).device

        # --- 1. Get latent posterior parameters (or use provided outputs) ---
        if outputs is None:
            mu_g0, logvar_g0 = self.encode_g0(x)
            g0_sample = self.sample_g0(mu_g0, logvar_g0)
            u_init = torch.zeros(B, T, self.inferred_input_dim, device=device)
            f_init, _ = self.forward_generator(g0_sample, u_init)
            mu_u, logvar_u = self.encode_u(x, f_init)
        else:
            mu_g0, logvar_g0 = outputs["mu_g0"], outputs["logvar_g0"]
            mu_u, logvar_u = outputs["mu_u"], outputs["logvar_u"]

        # --- 2. Compute KL divergences ---
        # KL for g0:
        mu_p_g0 = torch.zeros_like(mu_g0)
        logvar_p_g0 = torch.log(torch.full_like(mu_g0, self.kappa))
        kl_g0 = kl_div_gaussian(mu_g0, logvar_g0, mu_p_g0, logvar_p_g0).mean()

        # KL for u (AR(1) prior):
        alpha = self.alpha_ar
        sigma2_e = self.sigma2_e
        sigma2_p = sigma2_e / (1.0 - alpha ** 2)
        mu_u1_p = torch.zeros_like(mu_u[:, 0])
        logvar_u1_p = torch.log(torch.full_like(mu_u[:, 0], sigma2_p))
        kl_u1 = kl_div_gaussian(mu_u[:, 0], logvar_u[:, 0], mu_u1_p, logvar_u1_p).mean()
        kl_ut = 0.0
        for t in range(1, T):
            prior_mean = alpha * mu_u[:, t - 1]
            prior_logvar = torch.log(torch.full_like(prior_mean, sigma2_e))
            kl_t = kl_div_gaussian(mu_u[:, t], logvar_u[:, t], prior_mean, prior_logvar)
            kl_ut += kl_t.mean()
        kl_u = kl_u1 + kl_ut

        # --- 3. Draw multiple latent samples using the reparameterization trick ---
        std_g0 = torch.exp(0.5 * logvar_g0)
        eps_g0 = torch.randn(num_samples, *std_g0.shape, device=device)
        g0_samples = mu_g0.unsqueeze(0) + eps_g0 * std_g0.unsqueeze(0)

        std_u = torch.exp(0.5 * logvar_u)
        eps_u = torch.randn(num_samples, *std_u.shape, device=device)
        u_samples = mu_u.unsqueeze(0) + eps_u * std_u.unsqueeze(0)

        # --- 4. Run the generator on all latent samples in one batched pass ---
        g0_samples_flat = g0_samples.reshape(num_samples * B, -1)
        u_samples_flat = u_samples.reshape(num_samples * B, T, -1)
        f_samples, logits_samples = self.forward_generator(g0_samples_flat, u_samples_flat)
        logits_samples = logits_samples.view(num_samples, B, T, -1)

        # --- 5. Compute reconstruction loss ---
        new_shape = (B, T, self.alphabet_size ** self.order, self.alphabet_size)
        target_matrix = x.view(*new_shape)
        tol = 1e-3
        row_sums = target_matrix.sum(dim=-1)
        mask = row_sums > (1 - tol)
        derived_mask = mask[0, 0]  # assuming same mask across batch/time
        final_mask = derived_mask.unsqueeze(1).expand(self.alphabet_size ** self.order, self.alphabet_size)

        if self.order == 1:
            pred_dist = rowwise_softmax(logits_samples.view(num_samples * B, T, -1),
                                        self.alphabet_size, order=self.order)
            pred_dist = pred_dist.view(num_samples, B, T, -1)
        elif self.order == 2:
            pred_dist = rowwise_masked_softmax(
                logits_samples.view(num_samples * B, T, -1),
                self.alphabet_size,
                order=self.order,
                target_matrix=final_mask,
                mask_value=-1e8
            )
            pred_dist = pred_dist.view(num_samples, B, T, -1)
        else:
            raise ValueError("Unsupported order. Use order==1 or order==2.")

        pred_dist = torch.clamp(pred_dist, 1e-12, 1.0)
        x_expanded = x.unsqueeze(0).expand(num_samples, -1, -1, -1)
        ce = -x_expanded * torch.log(pred_dist)
        rec_loss = ce.sum(dim=-1).mean()

        # --- 6. Compute L2 regularization loss separately ---
        l2_loss = self.compute_recurrent_l2_reg()
        l2_coef = 1e-4  # set your coefficient here (can be scheduled externally)

        # --- 7. Total loss ---
        total_loss = rec_loss + kl_g0 + kl_u + (l2_coef * l2_loss)

        loss_dict = {
            "rec_loss": rec_loss,
            "kl_g0": kl_g0,
            "kl_u": kl_u,
            "l2_loss": l2_loss
        }
        return total_loss, loss_dict

# Example usage:
#
# if __name__ == "__main__":
#     alphabet_size = 7
#     model = BirdsongLFADSModel2(
#         alphabet_size=alphabet_size,
#         encoder_dim=32,
#         controller_dim=32,
#         generator_dim=64,
#         factor_dim=16,
#         latent_dim=8,
#         inferred_input_dim=4,
#         kappa=1.0,
#         ar_step_size=0.95,
#         ar_process_var=0.1
#     )
#
#     # Suppose we have targets that are row-wise normalized bigram distributions.
#     # For demonstration, let's create random one-hot rows (so each row has exactly 1 "1" entry).
#     # We'll do that for each position i=0..alphabet_size-1
#     # so each row is a one-hot over the columns.
#     batch_size = 2
#     time_steps = 5
#
#     # We'll build random row-wise distributions:
#     # shape (batch_size, time_steps, alphabet_size, alphabet_size)
#     x_mat = torch.zeros(batch_size, time_steps, alphabet_size, alphabet_size)
#     for b in range(batch_size):
#         for t in range(time_steps):
#             for row in range(alphabet_size):
#                 col_idx = torch.randint(0, alphabet_size, (1,))
#                 x_mat[b, t, row, col_idx] = 1.0
#
#     # Flatten to (batch, time, alpha^2)
#     x_flat = x_mat.view(batch_size, time_steps, alphabet_size**2)
#
#     outputs = model(x_flat)
#     loss, loss_dict = model.compute_loss(x_flat, outputs)
#
#     print("Loss:", loss.item())
#     print("Loss breakdown:", loss_dict)
#     print("Logits shape:", outputs["logits"].shape)
#     print("Factors shape:", outputs["factors"].shape)
