import torch
import torch.nn as nn
from DCUnet10_TSTM.Dual_Transformer import Dual_Transformer

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
            
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

class DCUnet10(nn.Module):
    """
    Deep Complex U-Net.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    # downsampling/encoding 
    def forward(self, x, n_fft, hop_length,is_istft=True):

        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        u0 = self.upsample0(d4)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        if is_istft:
          output = torch.squeeze(output, 1)    
          output = torch.istft(output, n_fft=n_fft, hop_length=hop_length, normalized=True)  
        
        return output

class DCUnet10_rTSTM(nn.Module):
    """
    Deep Complex U-Net with real TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, n_fft, hop_length,is_istft=True):
        # encoder
        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        # real TSTM
        d4_1 = d4[:, :, :, :, 0]
        d4_2 = d4[:, :, :, :, 1]
        d4_1 = self.dual_transformer(d4_1)
        d4_2 = self.dual_transformer(d4_2)

        out = torch.rand(d4.shape)
        out[:, :, :, :, 0] = d4_1
        out[:, :, :, :, 1] = d4_2
        out= out.to('cuda')

        # decoder
        u0 = self.upsample0(out)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        if is_istft:
          output = torch.squeeze(output, 1) 
          output = torch.istft(output, n_fft=n_fft, hop_length=hop_length, normalized=True) 
        
        return output

class DCUnet10_cTSTM(nn.Module):
    """
    Deep Complex U-Net with complex TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer_real = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]
        self.dual_transformer_imag = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    # downsampling/encoding 
    def forward(self, x, n_fft, hop_length,is_istft=True):
        # encoder
        d0 = self.downsample0(x)
        # print("d0:",d0.shape)
        d1 = self.downsample1(d0) 
        # print("d1:",d1.shape)
        d2 = self.downsample2(d1)
        # print("d2:",d2.shape)        
        d3 = self.downsample3(d2)    
        # print("d3:",d3.shape)    
        d4 = self.downsample4(d3)
        # print("d4:",d4.shape)
        
        # complex TSTM
        d4_real = d4[:, :, :, :, 0]
        d4_imag = d4[:, :, :, :, 1]

        out_real = self.dual_transformer_real(d4_real)- self.dual_transformer_imag(d4_imag)
        out_imag = self.dual_transformer_imag(d4_real) + self.dual_transformer_real(d4_imag)   
        
        out = torch.rand(d4.shape)
        out[:, :, :, :, 0] = out_real
        out[:, :, :, :, 1] = out_imag
        out= out.to('cuda')
        #print("out:",out.shape)  

        # decoder
        u0 = self.upsample0(out)    # upsampling/decoding 
        c0 = torch.cat((u0, d3), dim=1)   # skip-connection
        # print("c0:",c0.shape)
        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)
        # print("c1:",c1.shape)
        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)
        # print("c2:",c2.shape)
        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)
        # print("c3:",c3.shape)
        u4 = self.upsample4(c3)
        # print("u4:",u4.shape)

        output = u4 * x    # u4 - the mask

        if is_istft:
          output = torch.squeeze(output, 1)
          output = torch.istft(output, n_fft=n_fft, hop_length=hop_length, normalized=True)
        
        return output