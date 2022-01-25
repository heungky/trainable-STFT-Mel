from __future__ import division
import torch
import numpy as np
import math
import logging
from packaging import version
from nnAudio.Spectrogram import *
from torch import nn
from torch.nn import functional as F


class Filterbank(torch.nn.Module):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Top dB valu used for log-mels.
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).

    Example
    -------
    >>> import torch
    >>> compute_fbanks = Filterbank()
    >>> inputs = torch.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
        sort=False
    ):
        super().__init__()
        self.sort=sort
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor
        zero = torch.zeros(1)
        self.register_buffer('zero', zero)         

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Filter definition
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        band = band[:-1]
        f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
#             self.f_central = torch.nn.Parameter(
#                 self.f_central / (self.sample_rate * self.param_change_factor)
#             )
#             self.band = torch.nn.Parameter(
#                 self.band / (self.sample_rate * self.param_change_factor)
#             )
            f_central = torch.nn.Parameter(
                f_central / (self.sample_rate * self.param_change_factor)
            )
            band = torch.nn.Parameter(
                band / (self.sample_rate * self.param_change_factor)
            )
            self.register_parameter('f_central', f_central)
            self.register_parameter('band', band)
        else:
            self.register_buffer('f_central', f_central/self.sample_rate)
            self.register_buffer('band', band/self.sample_rate)            

#         print(f"init: {self.f_central=}")

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # Replicating for all the filters
        all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)
        self.register_buffer('all_freqs_mat', all_freqs_mat)         

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        x : tensor
            A batch of spectrogram tensors.
        """
        f_central=torch.clamp(self.f_central, 0, 0.5)		
        band=torch.clamp(self.band, 3.1/16000, 603.7/16000)		
        if self.sort:		
            f_central, _ = torch.sort(f_central)		
            band, _ = torch.sort(band)
        # Computing central frequency and bandwidth of each filter
#         print(f'{f_central=}')   
        f_central_mat = f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
#         print(f'1st {f_central_mat=}')     
        
        
        band_mat = band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )
        
        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
        else:
            f_central_mat = f_central_mat * (self.sample_rate)
            band_mat = band_mat * (self.sample_rate)            
                
        # Regularization with random changes of filter central frequency and band
        if self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]
#        print(f'{f_central_mat=}')
#        print(f'{band_mat=}')
        
        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(spectrogram.device)
#fbank_matrix trangular filter banks        
#        print(f'{fbank_matrix.max()=}')
#        print(f'{fbank_matrix.min()=}')        
        
        sp_shape = spectrogram.shape

                                  
        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.reshape(
                sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]
            )

        # FBANK computation
#        print(f'{spectrogram.max()}')
#fbanks is spectrogram
        fbanks = torch.matmul(spectrogram, fbank_matrix)
#        print(f"L255")              
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)
#            print(f"Pass")                
        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(
                sp_shape[0], fb_shape[1], fb_shape[2], sp_shape[3]
            )

        return fbanks
    
    def get_fbanks(self):
        f_central=torch.clamp(self.f_central, 0, 0.5)		
        band=torch.clamp(self.band, 3.1/16000, 603.7/16000)
#f_central and band are the 80 parameters to control filter banks
        if self.sort:		
            f_central, _ = torch.sort(f_central)		
            band, _ = torch.sort(band)
        # Computing central frequency and bandwidth of each filter
#         print(f'{f_central=}')   
        f_central_mat = f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
#         print(f'1st {f_central_mat=}')     
        
        
        band_mat = band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )
#f_central and band do some transformation become f_central_mat  and band_mat     
        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
        else:
            f_central_mat = f_central_mat * (self.sample_rate)
            band_mat = band_mat * (self.sample_rate)            
                
        # Regularization with random changes of filter central frequency and band
        if self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]
#        print(f'{f_central_mat=}')
#        print(f'{band_mat=}')
        
        return self._create_fbank_matrix(f_central_mat, band_mat)
#pass f_central_mat, band_mat to create_fbank_matrix to make filter banks 
#created  get_fbanks  to plot filter banks
        
        
    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        x : float
            The frequency point in Hz.
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        x : float
            The frequency point in the mel-scale.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # Computing the slops of the filters
        all_freqs=all_freqs                       
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0
        
                       
        # Adding zeros for negative values
        #zero = torch.zeros(1)#.()   coz this is constant, moved to __init__
        #zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(
            self.zero, torch.min(left_side, right_side)
        ).transpose(0, 1)

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        all_freqs=all_freqs#.cuda()
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : Tensor
            A batch of linear FBANK tensors.

        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max
        new_x_db_max = x_db.max() - self.top_db
        # Clipping to dB max
        x_db = torch.max(x_db, new_x_db_max)

        return x_db

    #############################
    
class FastAudio(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool (default: False)
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool (default: False)
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool (default: False)
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int (default: 160000)
        Sampling rate for the input waveforms.
    f_min : int (default: 0)
        Lowest frequency for the Mel filters.
    f_max : int (default: None)
        Highest frequency for the Mel filters. Note that if f_max is not
        specified it will be set to sample_rate // 2.
    win_length : float (default: 25)
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float (default: 10)
        Length (in ms) of the hop of the sliding window used to compute
        the STFT.
    n_fft : int (default: 400)
        Number of samples to use in each stft.
    n_mels : int (default: 40)
        Number of Mel filters.
    filter_shape : str (default: triangular)
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor : float (default: 1.0)
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training.
    param_rand_factor : float (default: 0.0)
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int (default: 5)
        Number of frames of left context to add.
    right_frames : int (default: 5)
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        f_min=0,
        f_max=None,
        n_fft=400,
        n_mels=40,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
        left_frames=5,
        right_frames=5,
        win_length=25,
        hop_length=10,
        sort=False
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad
        self.sort=sort

        if f_max is None:
            f_max = sample_rate / 2

        self.compute_STFT = STFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.compute_fbanks = Filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            freeze=not requires_grad,
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
            sort=sort
        )
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        # with torch.no_grad():

        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag)

        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            delta2 = self.compute_deltas(delta1)
            fbanks = torch.cat([fbanks, delta1, delta2], dim=2)

        if self.context:
            fbanks = self.context_window(fbanks)

        return fbanks