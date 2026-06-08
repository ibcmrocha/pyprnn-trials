#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Physically Recurrent Neural Network (PRNN)

The PRNN is an encoder-decoder architecture that learns tasks akin to
localization and homogenization operations in micromechanics. Meaningful
representations are encouraged by including between encoder and decoder
a material layer with a set of real (physics-based) constitutive models.

    - Encoder (localization): encodes global strains to a set of local
      strains for an adjustable number of 'fictitious' material points.
      The current setup implements a single affine and fully-connected
      layer. Including non-linearity through hidden layers might be 
      beneficial depending on the application.

    - Material layer: converts local strains to local stresses according
      to fixed (non-trainable) and purely physics-based constitutive laws.
      If targets come from microscale simulations, including in the material 
      layer the exact same constitutive models as in the microscale model
      tends to yield the best results. Material properties are in principle
      fixed to their microscale values, although the code can be trivially
      extended to make them trainable. State variable evolution (e.g plastic
      strains, damage) is handled exclusively by the material models and
      are therefore not learned from data

    - Decoder (homogenization): decodes the complete set of local stresses
      to a global homogenized stress. To more closely mimic the usual
      micromodel volume averaging, decoder weights are constrained to be
      positive by activating them with softplus.

The code here also includes a small custom layer class implementing the 
softplus weight activation.
"""

import math

import torch

from J2Tensor_vect import J2Material


class PRNN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_matpts, **kwargs):
        super(PRNN,self).__init__()

        self.device = kwargs.get('device',torch.device('cpu'))

        self.n_features   = n_features
        self.mat_pts      = n_matpts
        self.n_latents    = self.mat_pts*self.n_features
        self.n_outputs    = n_outputs
                
        print('------ PRNN model summary ------')
        print('Input (strain) size', self.n_features)
        print('Material layer size (points)', self.mat_pts)
        print('Material layer size (units)', self.n_latents)
        print('Output (stress) size', self.n_outputs)
        print('--------------------------------')
        
        self.fc1 = torch.nn.Linear(in_features=self.n_features,
                                   out_features=self.n_latents,
                                   device = self.device,
                                   bias = False)
        self.fc2 = SoftLayer(in_features=self.n_latents,
                             out_features=self.n_outputs,
                             device = self.device,
                             bias = False)
 
    def forward(self,x):
        batch_size, seq_len, _ = x.size()

        output =  x.clone()
        out = torch.zeros(
                [batch_size,seq_len, self.n_outputs]).to(self.device)
        
        # Create material model and fictitious integration points
        
        material_model = J2Material(self.device) 
        
        ip_pointsb = batch_size*self.mat_pts
        material_model.configure(ip_pointsb)        
        
        # Process (batched) strain paths one time step at a time

        for t in range(seq_len):
          # Encoder (localization)
               
          outputt = self.fc1(output[:, t,:])
               
          # Run material model (strain, oldstate -> stress, newstate) 
          
          outputt = material_model.update(
                  outputt.view(ip_pointsb,self.n_features))

          # Store updated material history

          material_model.commit() 
    
          # Decoder (homogenization)
                   
          outputt = self.fc2(outputt.view(batch_size, self.n_latents))
          out[:, t, :] = outputt.view(-1,self.n_outputs)

        output = out.to(self.device)
        return output


class PRNNClassifier(torch.nn.Module):
    """Stochastic PRNN classifier for small image classification tasks.

    The classifier can encode images as stochastic Gaussian-process strain
    paths or as deterministic rasterized image time signals.  In GP mode, a
    simple image encoder parameterizes three independent zero-mean Gaussian
    processes (one each for ``eps_xx``, ``eps_yy`` and ``gam_xy``), and the
    GP amplitudes can either be constant trainable parameters or
    image-dependent encoder outputs.  Every call to :meth:`forward` sends a
    strain path through the same material layer used by :class:`PRNN`, and
    decodes the selected final material state features of the material
    points to class logits or probabilities.
    """

    def __init__(self, image_shape, n_matpts, seq_len, **kwargs):
        super(PRNNClassifier, self).__init__()

        self.device = kwargs.get('device', torch.device('cpu'))
        self.n_features = 3
        self.n_outputs = kwargs.get('n_classes', kwargs.get('n_outputs', 2))
        self.mat_pts = n_matpts
        self.n_latents = self.mat_pts * self.n_features
        self.seq_len = seq_len
        self.jitter = kwargs.get('jitter', 1e-6)
        self.length_scale_min = kwargs.get('length_scale_min', 1e-2)
        self.strain_scale = kwargs.get('strain_scale', 1.0)
        hidden_size = kwargs.get('hidden_size', 64)
        self.image_parametrization = kwargs.get('image_parametrization', 'gp')
        valid_image_parametrizations = ('gp', 'timesignal')
        if self.image_parametrization not in valid_image_parametrizations:
            raise ValueError(
                'image_parametrization must be one of '
                f'{valid_image_parametrizations}, '
                f'got {self.image_parametrization!r}.'
            )
        self.sigma_f_mode = kwargs.get('sigma_f_mode', 'constant')
        valid_sigma_f_modes = ('constant', 'image')
        if self.sigma_f_mode not in valid_sigma_f_modes:
            raise ValueError(
                'sigma_f_mode must be one of '
                f'{valid_sigma_f_modes}, got {self.sigma_f_mode!r}.'
            )
        self.decoder_features = kwargs.get('decoder_features', 'epspeq')
        valid_decoder_features = ('epspeq', 'epsp', 'both')
        if self.decoder_features not in valid_decoder_features:
            raise ValueError(
                'decoder_features must be one of '
                f'{valid_decoder_features}, got {self.decoder_features!r}.'
            )

        if self.n_outputs < 2:
            raise ValueError('PRNNClassifier requires at least two classes.')

        if isinstance(image_shape, int):
            image_shape = (image_shape,)
        self.image_shape = tuple(image_shape)
        image_size = math.prod(self.image_shape)
        self.timesignal_seq_len = image_size
        if len(self.image_shape) >= 2:
            self.timesignal_seq_len = self.image_shape[-2] * self.image_shape[-1]
        if self.image_parametrization == 'timesignal':
            self.seq_len = self.timesignal_seq_len

        print('--- PRNNClassifier model summary ---')
        print('Input (image) size', self.image_shape)
        print('Image parametrization', self.image_parametrization)
        print('Strain path length', self.seq_len)
        print('Material layer size (points)', self.mat_pts)
        print('Material layer size (units)', self.n_latents)
        print('Decoder features', self.decoder_features)
        print('GP sigma_f mode', self.sigma_f_mode)
        print('Output classes', self.n_outputs)
        print('------------------------------------')

        if self.image_parametrization == 'gp':
            encoder_outputs = self.n_features
            if self.sigma_f_mode == 'image':
                encoder_outputs = 2 * self.n_features
            self.image_encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(image_size, hidden_size, device=self.device),
                torch.nn.Linear(hidden_size, encoder_outputs, device=self.device),
            )
        else:
            self.image_encoder = None
        self.fc1 = torch.nn.Linear(
            in_features=self.n_features,
            out_features=self.n_latents,
            device=self.device,
            bias=False,
        )
        decoder_input_sizes = {
            'epspeq': self.mat_pts,
            'epsp': self.n_latents,
            'both': self.mat_pts + self.n_latents,
        }
        decoder_input_size = decoder_input_sizes[self.decoder_features]
        self.decoder = torch.nn.Linear(
            in_features=decoder_input_size,
            out_features=self.n_outputs,
            device=self.device,
        )
        sigma_f_init = torch.full(
            (self.n_features,),
            kwargs.get('sigma_f_init', 2.0e-2),
            device=self.device,
        )
        if (
            self.image_parametrization == 'gp'
            and self.sigma_f_mode == 'constant'
        ):
            self.raw_sigma_f = torch.nn.Parameter(_inverse_softplus(sigma_f_init))
        else:
            self.register_parameter('raw_sigma_f', None)
        time_grid = torch.linspace(0.0, 1.0, self.seq_len, device=self.device)
        self.register_buffer('time_grid', time_grid)

    @property
    def sigma_f(self):
        if self.raw_sigma_f is None:
            return None
        return torch.nn.functional.softplus(self.raw_sigma_f)

    def encode_length_scales(self, images):
        length_scales, _ = self.encode_gp_parameters(images)
        return length_scales

    def encode_gp_parameters(self, images):
        if self.image_parametrization != 'gp':
            raise ValueError(
                'GP parameters are unavailable when image_parametrization '
                f'is {self.image_parametrization!r}.'
            )
        images = images.to(device=self.device, dtype=self.time_grid.dtype)
        raw_gp_parameters = self.image_encoder(images)
        if self.sigma_f_mode == 'constant':
            raw_length_scales = raw_gp_parameters
            sigma_f = self.sigma_f
        else:
            raw_length_scales, raw_sigma_f = raw_gp_parameters.chunk(2, dim=-1)
            sigma_f = torch.nn.functional.softplus(raw_sigma_f)
        length_scales = (
            torch.nn.functional.softplus(raw_length_scales)
            + self.length_scale_min
        )
        return length_scales, sigma_f

    def encode_strain_paths(self, images):
        if self.image_parametrization == 'timesignal':
            return self.encode_timesignal_paths(images), None, None

        length_scales, sigma_f = self.encode_gp_parameters(images)
        strain_paths = self.sample_strain_paths(length_scales, sigma_f)
        return strain_paths, length_scales, sigma_f

    def encode_timesignal_paths(self, images):
        images = images.to(device=self.device, dtype=self.time_grid.dtype)
        if images.dim() == 4:
            if images.size(1) != 1:
                raise ValueError(
                    'timesignal parametrization expects grayscale images.'
                )
            image_matrix = images[:, 0, :, :]
        elif images.dim() == 3:
            image_matrix = images
        else:
            raise ValueError(
                'timesignal parametrization expects images shaped as '
                '[batch, channels, height, width] or [batch, height, width].'
            )

        row_signal = image_matrix.reshape(image_matrix.size(0), -1)
        column_signal = image_matrix.transpose(1, 2).reshape(
            image_matrix.size(0),
            -1,
        )
        diagonal_signal = self._diagonal_vectorize(image_matrix)
        strain_paths = 0.1 * torch.stack((
            row_signal,
            column_signal,
            diagonal_signal,
        ), dim=-1)
        return strain_paths.contiguous()

    def _diagonal_vectorize(self, image_matrix):
        height = image_matrix.size(1)
        width = image_matrix.size(2)
        diagonals = []
        for offset in range(width):
            diagonals.append(
                image_matrix.diagonal(offset=offset, dim1=1, dim2=2)
            )
        for offset in range(-1, -height, -1):
            diagonals.append(
                image_matrix.diagonal(offset=offset, dim1=1, dim2=2)
            )
        return torch.cat(diagonals, dim=1)

    def sample_strain_paths(self, length_scales, sigma_f=None):
        batch_size = length_scales.size(0)
        dtype = length_scales.dtype
        time_grid = self.time_grid.to(dtype=dtype)
        dt2 = (time_grid[:, None] - time_grid[None, :]).pow(2)
        if sigma_f is None:
            sigma_f = self.sigma_f
        if sigma_f is None:
            raise ValueError(
                'sigma_f must be provided when sigma_f_mode is image.'
            )
        sigma_f = sigma_f.to(device=self.device, dtype=dtype)
        if sigma_f.dim() == 1:
            sigma_f = sigma_f.view(1, self.n_features, 1, 1)
        else:
            sigma_f = sigma_f.view(batch_size, self.n_features, 1, 1)

        cov = sigma_f.pow(2) * torch.exp(
            -0.5 * dt2.view(1, 1, self.seq_len, self.seq_len)
            / length_scales.view(batch_size, self.n_features, 1, 1).pow(2)
        )
        eye = torch.eye(self.seq_len, device=self.device, dtype=dtype)
        cov = cov + self.jitter * eye.view(1, 1, self.seq_len, self.seq_len)
        chol = torch.linalg.cholesky(cov)
        noise = torch.randn(
            batch_size,
            self.n_features,
            self.seq_len,
            device=self.device,
            dtype=dtype,
        )
        paths = torch.matmul(chol, noise.unsqueeze(-1)).squeeze(-1)
        return self.strain_scale * paths.transpose(1, 2).contiguous()

    def forward(self, images, return_paths=False, return_logits=False):
        strain_paths, length_scales, _ = self.encode_strain_paths(images)
        batch_size = strain_paths.size(0)

        material_model = J2Material(self.device)
        ip_pointsb = batch_size * self.mat_pts
        material_model.configure(ip_pointsb)

        for t in range(self.seq_len):
            local_strain = self.fc1(strain_paths[:, t, :])
            material_model.update(local_strain.view(ip_pointsb, self.n_features))
            material_model.commit()

        decoder_features = self._get_decoder_features(material_model, batch_size)
        logits = self.decoder(decoder_features)
        output = logits
        if not return_logits:
            output = torch.softmax(logits, dim=-1)

        if return_paths:
            return output, strain_paths, length_scales, decoder_features
        return output

    def _get_decoder_features(self, material_model, batch_size):
        epspeq = material_model.getHistory().view(batch_size, self.mat_pts)
        epsp = material_model.epsp_hist.view(batch_size, self.n_latents)
        if self.decoder_features == 'epspeq':
            return epspeq
        if self.decoder_features == 'epsp':
            return epsp
        return torch.cat((epspeq, epsp), dim=1)


class SoftLayer(torch.nn.Module): 
    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SoftLayer,self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.sp = torch.nn.Softplus()
        self.weight = torch.nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(
                    torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
      return torch.nn.functional.linear(
              input, self.sp(self.weight), self.bias) 


def _inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))
