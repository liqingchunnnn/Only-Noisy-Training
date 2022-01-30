import os
import gc
import torch
import torchaudio
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pesq import pesq
from scipy import interpolate
from torch.utils.data import DataLoader

from dataset_utils import SpeechDataset,subsample2,subsample4
from DCUnet10_TSTM.DCUnet import DCUnet10,DCUnet10_rTSTM,DCUnet10_cTSTM
from metrics import AudioMetrics2
from loss import RegularizedLoss
    
# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()
if (train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.random.seed(999)
torch.manual_seed(999)

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set Audio backend as Soundfile for windows and Sox for Linux
torchaudio.set_audio_backend("sox_io")
print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

###################################### Parameters of Speech processing ##################################
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

######################################## Datasets setting #########################################
# Choose white noise or different noise types in urbansound8K
noise_class = "white"

# Load white noise
if noise_class == "white":
    TRAIN_INPUT_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_Input')
    TRAIN_TARGET_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Train_Output')

    TEST_NOISY_DIR = Path('/home/abc/n2n/Datasets/WhiteNoise_Test_Input')
    TEST_CLEAN_DIR = Path('/home/abc/n2n/Datasets/clean_testset_wav')

# LOad urbansound8K noise
else:
    TRAIN_INPUT_DIR = Path('/home/abc/n2n/Datasets/US_Class' + str(noise_class) + '_Train_Input')
    TRAIN_TARGET_DIR = Path('/home/abc/n2n/Datasets/US_Class' + str(noise_class) + '_Train_Output')

    TEST_NOISY_DIR = Path('/home/abc/n2n/Datasets/US_Class' + str(noise_class) + '_Test_Input')
    TEST_CLEAN_DIR = Path('/home/abc/n2n/Datasets/clean_testset_wav')

train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

print("No. of Training files:",len(train_input_files))
print("No. of Testing files:",len(test_noisy_files))

basepath = str(noise_class)
fixedpath = '/home/abc/n2n/SNA-DF/DCUnet10_complex_TSTM_subsample2/'

os.makedirs(fixedpath + basepath,exist_ok=True)
os.makedirs(fixedpath + basepath+"/Weights",exist_ok=True)
respath = fixedpath + basepath + '/results.txt'
#os.makedirs(basepath+"/Samples",exist_ok=True)

######################################## Metrics for evaluation #########################################
def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB", "PESQ-NB", "SNR", "SSNR", "STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i + 1) % 10 == 0:
            end_str = "\n"
        else:
            end_str = ","
        # print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            x_noisy_stft = data[0]
            g1_stft = data[1]
            g1_wav = data[2]
            g2_wav = data[3]
            x_clean_stft = data[4]

            if use_net:  # Forward of net returns the istft version
                x_est = net(x_noisy_stft.to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, is_istft=True)  #返回波形图
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                x_est_np = x_noisy_stft.view(-1).detach().cpu().numpy()
            
            x_clean_np = torch.istft(torch.squeeze(x_clean_stft, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)

            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')

            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            # print(new_scores)
            # print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] = np.mean(overall_metrics[i])
        temp["STD"] = np.std(overall_metrics[i])
        temp["Min"] = min(overall_metrics[i])
        temp["Max"] = max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data", addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results
######################################## TRAIN #########################################

def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0

    for x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft in train_loader:
        # zero gradients
        net.zero_grad()

        # for base training (input---g1_stft, target---fg1_wav)
        g1_stft = g1_stft.to(DEVICE)
        fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
      
        # for regularization loss (input---x_noisy_stft, target---fx_wav)
        with torch.no_grad():
            x_noisy_stft = x_noisy_stft.to(DEVICE)  
            fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
            g1fx, g2fx = subsample2(fx_wav)
            g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)

        # calculate loss
        g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.to(DEVICE), fg1_wav.to(DEVICE), g2_wav.to(DEVICE), g1fx.to(DEVICE), g2fx.to(DEVICE)
        loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.

    with torch.no_grad():
        for x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft in test_loader:

            # for base training (input---g1_stft, target---fg1_wav)
            g1_stft = g1_stft.to(DEVICE)
            fg1_wav = net(g1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)

            # for regularization loss (input---x_noisy_stft, target---fx_wav)
            x_noisy_stft= x_noisy_stft.to(DEVICE)
            fx_wav = net(x_noisy_stft, n_fft=N_FFT, hop_length=HOP_LENGTH)
            g1fx, g2fx = subsample2(fx_wav)
            g1fx, g2fx = g1fx.type(torch.FloatTensor), g2fx.type(torch.FloatTensor)

            # calculate loss
            g1_wav, fg1_wav, g2_wav, g1fx, g2fx = g1_wav.to(DEVICE), fg1_wav.to(DEVICE), g2_wav.to(DEVICE), g1fx.to(DEVICE), g2fx.to(DEVICE)
            loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()

            test_ep_loss += loss.item()
            counter += 1

        test_ep_loss /= counter

        print("Actual compute done...testing now")

        testmet = getMetricsonLoader(test_loader, net, use_net)

        # clear cache
        gc.collect()
        torch.cuda.empty_cache()

        return test_ep_loss, testmet

def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")

        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        with open(fixedpath + basepath + '/results.txt', "a") as f:
            f.write("Epoch :" + str(e + 1) + "\n" + str(testmet))
            f.write("\n")

        print("OPed to txt")

        torch.save(net.state_dict(), fixedpath + basepath + '/Weights/dc10_model_' + str(e + 1) + '.pth')
        torch.save(optimizer.state_dict(), fixedpath + basepath + '/Weights/dc10_opt_' + str(e + 1) + '.pth')

        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("Epoch: {}/{}...".format(e+1, epochs),
                     "Loss: {:.6f}...".format(train_loss),
                     "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss

######################################## Train CONFI #########################################
test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

# clear cache
gc.collect()
torch.cuda.empty_cache()

dcunet = DCUnet10(N_FFT, HOP_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(dcunet.parameters())
loss_fn = RegularizedLoss()
loss_fn = loss_fn.to(DEVICE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# specify paths and uncomment to resume training from a given point
# model_checkpoint = torch.load(path_to_model)
# opt_checkpoint = torch.load(path_to_opt)
# dcunet20.load_state_dict(model_checkpoint)
# optimizer.load_state_dict(opt_checkpoint)

train_losses, test_losses = train(dcunet, train_loader, test_loader, loss_fn, optimizer, scheduler, 20)

