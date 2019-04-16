# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR
import sys, os, time
sys.path.append('utils')
sys.path.append('models')
from utils.data import CelebA, RandomNoiseGenerator
from models.model import Generator, Discriminator
import argparse
import numpy as np
from scipy.misc import imsave
from utils.logger import Logger

class PGGAN():
    def __init__(self, G, D, data, noise, opts):
        self.G = G
        self.D = D
        self.data = data
        self.noise = noise
        self.opts = opts
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.logger = Logger('./logs/' + self.current_time + "/")
        gpu = self.opts['gpu']
        self.use_cuda = len(gpu) > 0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)} # batch size map keyed by resolution_level
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}

        self.restore_model()

        # save opts
        with open(os.path.join(self.opts['exp_dir'], self.time, 'options_%s.txt'%self.current_time), 'w') as f:
            for k, v in self.opts.items():
                print('%s: %s' % (k, v), file=f)
            print('batch_size_map: %s' % self.bs_map, file=f)

    def restore_model(self):
        exp_dir = self.opts['restore_dir']
        which_file = self.opts['which_file']  # 128x128-fade_in-105000
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        if exp_dir == '' or which_file == '':
            self.time = self.current_time
            self._from_resol = self.opts['first_resol']
            self._phase = 'stabilize'
            self._epoch = 0
            self.is_restored = False
            self.opts['sample_dir'] = os.path.join(self.opts['exp_dir'], self.current_time, 'samples')
            self.opts['ckpt_dir'] = os.path.join(self.opts['exp_dir'], self.current_time, 'ckpts')
            os.makedirs(self.opts['sample_dir'])
            os.makedirs(self.opts['ckpt_dir'])
            return 
        else:
            pattern = which_file.split('-')
            self._from_resol = int(pattern[0].split('x')[0])
            self._phase = pattern[1]
            self._epoch = int(pattern[2])
            tmp = exp_dir.split('/')
            self.opts['exp_dir'] = '/'.join(tmp[:-1])
            self.time = tmp[-1]
            self.opts['sample_dir'] = os.path.join(exp_dir, 'samples')
            self.opts['ckpt_dir'] = os.path.join(exp_dir, 'ckpts')
            assert os.path.exists(self.opts['sample_dir']) and os.path.exists(self.opts['ckpt_dir'])

            G_model = os.path.join(self.opts['ckpt_dir'], which_file+'-G.pth')
            D_model = os.path.join(self.opts['ckpt_dir'], which_file+'-D.pth')
            assert os.path.exists(G_model) and os.path.exists(D_model)
            self.G.load_state_dict(torch.load(G_model))
            self.D.load_state_dict(torch.load(D_model))
            self.is_restored = True
            print('Restored from dir: %s, pattern: %s' % (exp_dir, which_file))

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)

    def register_on_gpu(self):
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.opts['g_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.opts['d_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))
        
    def create_criterion(self):
        # w is for gan
        if self.opts['gan'] == 'lsgan':
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  # sigmoid is applied here
        elif self.opts['gan'] == 'wgan_gp':
            self.adv_criterion = lambda p,t,w: (-2*t+1) * torch.mean(p)
        elif self.opts['gan'] == 'gan':
            self.adv_criterion = lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.opts['gan'])

    def compute_adv_loss(self, prediction, target, w):
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self):
        return 0.0

    def compute_additional_d_loss(self):  # drifting loss and gradient penalty, weighting inside this function
        return 0.0

    def _get_data(self, d):
        return d.data[0] if isinstance(d, Variable) else d

    def compute_G_loss(self):
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_add_loss = self.compute_additional_g_loss()
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_add_loss = self._get_data(g_add_loss)
        return g_adv_loss + g_add_loss

    def compute_D_loss(self):
        self.d_adv_loss_real = self.compute_adv_loss(self.d_real, True, 0.5)
        self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5) * self.opts['fake_weight']
        d_adv_loss = self.d_adv_loss_real + self.d_adv_loss_fake
        d_add_loss = self.compute_additional_d_loss()
        self.d_adv_loss = self._get_data(d_adv_loss)
        self.d_add_loss = self._get_data(d_add_loss)

        return d_adv_loss + d_add_loss

    def _rampup(self, epoch, rampup_length):
        if epoch < rampup_length:
            p = max(0.0, float(epoch)) / float(rampup_length)
            p = 1.0 - p
            return np.exp(-p*p*5.0)
        else:
            return 1.0

    def _rampdown_linear(self, epoch, num_epochs, rampdown_length):
        if epoch >= num_epochs - rampdown_length:
            return float(num_epochs - epoch) / rampdown_length
        else:
            return 1.0

    '''Update Learning rate
    '''
    def update_lr(self, cur_nimg):
        for param_group in self.optim_G.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.opts['total_kimg'], self.opts['rampdown_kimg'])
            param_group['lr'] = lrate_coef * self.opts['g_lr_max']
        for param_group in self.optim_D.param_groups:
            lrate_coef = self._rampup(cur_nimg / 1000.0, self.opts['rampup_kimg'])
            lrate_coef *= self._rampdown_linear(cur_nimg / 1000.0, self.opts['total_kimg'], self.opts['rampdown_kimg'])
            param_group['lr'] = lrate_coef * self.opts['d_lr_max']

    def postprocess(self):
        # TODO: weight cliping or others
        pass

    def _numpy2var(self, x):
        var = Variable(torch.from_numpy(x))
        if self.use_cuda:
            var = var.cuda()
        return var

    def _var2numpy(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()

    # def add_noise(self, x):
    #     # TODO: support more method of adding noise.
    #     if self.opts.get('no_noise', False):
    #         return x

    #     if hasattr(self, '_d_'):
    #         self._d_ = self._d_ * 0.9 + torch.mean(self.d_real).data[0] * 0.1
    #     else:
    #         self._d_ = 0.0
    #     strength = 0.2 * max(0, self._d_ - 0.5)**2
    #     noise = self._numpy2var(np.random.randn(*x.size()).astype(np.float32) * strength)
    #     return x + noise

    def compute_noise_strength(self):
        if self.opts.get('no_noise', False):
            return 0

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + np.clip(torch.mean(self.d_real).data[0], 0.0, 1.0) * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        return strength

    def preprocess(self, z, real):
        self.z = self._numpy2var(z)
        self.real = self._numpy2var(real)

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake, cur_level=cur_level)
    
    def forward_D(self, cur_level, detach=True):
        self.fake = self.G(self.z, cur_level=cur_level)
        strength = self.compute_noise_strength()
        self.d_real = self.D(self.real, cur_level=cur_level, gdrop_strength=strength)
        self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level)
        # print('d_real', self.d_real.view(-1))
        # print('d_fake', self.d_fake.view(-1))
        # print(self.fake[0].view(-1))

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f, D_add: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss, self.g_add_loss, self.d_adv_loss, self.d_add_loss)
        print(formation % values)

    def tensorboard(self, it, num_it, phase, resol, samples):
        # (1) Log the scalar values
        prefix = str(resol)+'/'+phase+'/'
        info = {prefix + 'G_loss': self.g_loss,
                prefix + 'G_adv_loss': self.g_adv_loss,
                prefix + 'G_add_loss': self.g_add_loss,
                prefix + 'D_loss': self.d_loss,
                prefix + 'D_adv_loss': self.d_adv_loss,
                prefix + 'D_add_loss': self.d_add_loss,
                prefix + 'D_adv_loss_fake': self._get_data(self.d_adv_loss_fake),
                prefix + 'D_adv_loss_real': self._get_data(self.d_adv_loss_real)}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix +tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix +tag + '/grad', self._var2numpy(value.grad), it)

        for tag, value in self.D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          self._var2numpy(value.grad), it)

        # (3) Log the images
        # info = {'images': samples[:10]}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, it)

    def train_phase(self, R, phase, batch_size, cur_nimg, from_it, total_it):
        assert total_it >= from_it
        resol = 2 ** (R+1)
 
        for it in range(from_it, total_it):
            if phase == 'stabilize':
                cur_level = R
            else:
                cur_level = R + total_it/float(from_it)
            cur_resol = 2 ** int(np.ceil(cur_level+1))

            # get a batch noise and real images
            z = self.noise(batch_size)
            x = self.data(batch_size, cur_resol, cur_level)

            # ===preprocess===
            self.preprocess(z, x)
            self.update_lr(cur_nimg)

            # ===update D===
            self.optim_D.zero_grad()
            self.forward_D(cur_level, detach=True)
            self.backward_D()

            # ===update G===
            self.optim_G.zero_grad()
            self.forward_G(cur_level)
            self.backward_G()

            # ===report ===
            self.report(it, total_it, phase, cur_resol)

            cur_nimg += batch_size

            # ===generate sample images===
            samples = []
            if (it % self.opts['sample_freq'] == 0) or it == total_it-1:
                samples = self.sample()
                imsave(os.path.join(self.opts['sample_dir'],
                                    '%dx%d-%s-%s.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))), samples)

            # ===tensorboard visualization===
            if (it % self.opts['sample_freq'] == 0) or it == total_it - 1:
                self.tensorboard(it, total_it, phase, cur_resol, samples)

            # ===save model===
            if (it % self.opts['save_freq'] == 0 and it > 0) or it == total_it-1:
                self.save(os.path.join(self.opts['ckpt_dir'], '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))
        
    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        self.register_on_gpu()

        to_level = int(np.log2(self.opts['target_resol']))
        from_level = int(np.log2(self._from_resol))
        assert 2**to_level == self.opts['target_resol'] and 2**from_level == self._from_resol and to_level >= from_level >= 2

        train_kimg = int(self.opts['train_kimg'] * 1000)
        transition_kimg = int(self.opts['transition_kimg'] * 1000)

        for R in range(from_level-1, to_level):
            batch_size = self.bs_map[2 ** (R+1)]

            phases = {'stabilize':[0, train_kimg//batch_size], 'fade_in':[train_kimg//batch_size+1, (transition_kimg+train_kimg)//batch_size]}
            if self.is_restored and R == from_level-1:
                phases[self._phase][0] = self._epoch + 1
                if self._phase == 'fade_in':
                    del phases['stabilize']

            for phase in ['stabilize', 'fade_in']:
                if phase in phases:
                    _range = phases[phase]
                    self.train_phase(R, phase, batch_size, _range[0]*batch_size, _range[0], _range[1])

    def sample(self):
        batch_size = self.z.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                one_row.append(self.fake[i].cpu().data.numpy())
                i += 1
            # real
            for col in range(n_col):
                one_row.append(self.real[j].cpu().data.numpy())
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])

        half = samples.shape[1] // 2
        samples[:, :half, :] = samples[:, :half, :] - np.min(samples[:, :half, :])
        samples[:, :half, :] = samples[:, :half, :] / np.max(samples[:, :half, :])
        samples[:, half:, :] = samples[:, half:, :] - np.min(samples[:, half:, :])
        samples[:, half:, :] = samples[:, half:, :] / np.max(samples[:, half:, :])
        return samples

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str, help='gpu(s) to use.')
    parser.add_argument('--train_kimg', default=600, type=float, help='# * 1000 real samples for each stabilizing training phase.')
    parser.add_argument('--transition_kimg', default=600, type=float, help='# * 1000 real samples for each fading in phase.')
    parser.add_argument('--total_kimg', default=10000, type=float, help='total_kimg: a param to compute lr.')
    parser.add_argument('--rampup_kimg', default=10000, type=float, help='rampup_kimg.')
    parser.add_argument('--rampdown_kimg', default=10000, type=float, help='rampdown_kimg.')
    parser.add_argument('--g_lr_max', default=1e-3, type=float, help='Generator learning rate')
    parser.add_argument('--d_lr_max', default=1e-3, type=float, help='Discriminator learning rate')
    parser.add_argument('--fake_weight', default=0.1, type=float, help="weight of fake images' loss of D")
    parser.add_argument('--beta1', default=0, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
    parser.add_argument('--gan', default='lsgan', type=str, help='model: lsgan/wgan_gp/gan, currently only support lsgan or gan with no_noise option.')
    parser.add_argument('--first_resol', default=4, type=int, help='first resolution')
    parser.add_argument('--target_resol', default=256, type=int, help='target resolution')
    parser.add_argument('--drift', default=1e-3, type=float, help='drift, only available for wgan_gp.')
    parser.add_argument('--mbstat_avg', default='all', type=str, help='MinibatchStatConcatLayer averaging strategy (Which dimensions to average the statistic over?)')
    parser.add_argument('--sample_freq', default=500, type=int, help='sampling frequency.')
    parser.add_argument('--save_freq', default=5000, type=int, help='save model frequency.')
    parser.add_argument('--exp_dir', default='./exp', type=str, help='experiment dir.')
    parser.add_argument('--no_noise', action='store_true', help='do not add noise to real data.')
    parser.add_argument('--no_tanh', action='store_true', help='do not use tanh in the last layer of the generator.')
    parser.add_argument('--restore_dir', default='', type=str, help='restore from which exp dir.')
    parser.add_argument('--which_file', default='', type=str, help='restore from which file, e.g. 128x128-fade_in-105000.')

    # TODO: support conditional inputs

    args = parser.parse_args()
    opts = {k:v for k,v in args._get_kwargs()}

    # Dimensionality of the latent vector.
    latent_size = 512
    # Use sigmoid activation for the last layer?
    sigmoid_at_end = args.gan in ['lsgan', 'gan']
    if hasattr(args, 'no_tanh'):
        tanh_at_end = False
    else:
        tanh_at_end = True

    G = Generator(num_channels=3, latent_size=latent_size, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, tanh_at_end=tanh_at_end)
    D = Discriminator(num_channels=3, mbstat_avg=args.mbstat_avg, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end)
    print(G)
    print(D)
    data = CelebA()
    noise = RandomNoiseGenerator(latent_size, 'gaussian')
    pggan = PGGAN(G, D, data, noise, opts)
    pggan.train()
