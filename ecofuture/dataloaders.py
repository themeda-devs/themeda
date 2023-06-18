from pathlib import Path
import random
from fastai.data.core import TfmdDL
from fastai.data.load import _loaders, to_device
from fastai.callback.core import Callback

from .transforms import Chiplet

class TPlus1Dataloader(TfmdDL):
    def __iter__(self):
        return super().__iter__()
        # breakpoint()
        # self.randomize()
        # self.before_iter()
        # self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
        # for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
        #     # pin_memory causes tuples to be converted to lists, so convert them back to tuples
        #     if self.pin_memory and type(b) == list: b = tuple(b)
        #     if self.device is not None: b = to_device(b, self.device)
        #     yield self.after_batch(b)
        # self.after_iter()
        # if hasattr(self, 'it'): del(self.it)
        
    # def after_batch(self, batch):
    #     assert False
    #     breakpoint()
    #     return batch[:,:-1],batch[:,1:]
    


def t_plus_one(batch:tuple):
    breakpoint()
    if isinstance(batch, tuple):
        x = batch[0]
    else:
        x = batch
    return x[:,:-1],x[:,1:]



class TPlus1Callback(Callback):
    def before_batch(self):
        x = self.xb[0]

        self.learn.xb = (x[:,:-1],)
        self.learn.yb = (x[:,1:],)


# class ConditionalDDPMCallback(Callback):
#     """
#     Derived from https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1#using-fastai-to-train-your-diffusion-model
#     """
#     def __init__(self, n_steps, beta_min, beta_max, tensor_type=TensorImage, cosine_scheduler:bool = False):
#         store_attr()
#         self.cosine_scheduler = cosine_scheduler

#     def before_fit(self):
#         if self.cosine_scheduler:
#             self.alpha_bar = torch.cumprod(self.alpha, dim=0)
#             self.alpha = alpha_bar/torch.cat([torch.ones(1), alpha_bar[:-1]])
#             self.beta = 1.0 - self.alpha
#         else:    
#             self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps).to(self.dls.device) # variance schedule, linearly increased with timestep
#             self.alpha = 1. - self.beta 
#             self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
#         self.sigma = torch.sqrt(self.beta)

#     def before_batch_training(self):
#         lr = self.xb[0] # input low resolution images
#         hr = self.xb[1] # original high resolution images, x_0 
#         # delta = hr - lr
#         label = self.yb[0]
#         eps = self.tensor_type(torch.randn(hr.shape, device=hr.device)) # noise, x_T
#         batch_size = hr.shape[0]
#         t = torch.randint(0, self.n_steps, (batch_size,), device=hr.device, dtype=torch.long) # select random timesteps
#         alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1, 1) # HACK
#         # xt =  torch.sqrt(alpha_bar_t)*delta + torch.sqrt(1-alpha_bar_t)*eps #noisify the image
#         xt =  torch.sqrt(alpha_bar_t)*hr + torch.sqrt(1-alpha_bar_t)*eps #noisify the image
#         self.learn.xb = (xt, lr, t) # input to our model is noisy image and timestep
#         self.learn.yb = (eps,label) # ground truth is the noise 

#     def before_batch_sampling(self):
#         lr = self.xb[0] # input low resolution images
#         batch_size = lr.shape[0]
        
#         # Generate a batch of random noise to start with
#         # We can ignore the self.xb[0] data and just generate random noise here.
#         xt = self.tensor_type(torch.randn_like(lr))
#         # print(xt.mean().cpu().numpy(), xt.std().cpu().numpy())

#         for t in track(reversed(range(self.n_steps)), total=self.n_steps, description="Performing diffusion steps for batch:"):
#             t_batch = torch.full((batch_size,), t, device=xt.device, dtype=torch.long)
#             z = torch.randn(xt.shape, device=xt.device) if t > 0 else torch.zeros(xt.shape, device=xt.device)
#             alpha_t = self.alpha[t] # get noise level at current timestep
#             alpha_bar_t = self.alpha_bar[t]
#             sigma_t = self.sigma[t]
#             # result, predictions = self.model(xt, lr, t_batch)
#             result,  = self.model(xt, lr, t_batch)
            
#             xt = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * result)  + sigma_t*z # predict x_(t-1) in accordance to Algorithm 2 in paper
#             # corrected = lr + xt

#             # print(result.mean().cpu().numpy(), result.std().cpu().numpy(), xt.mean().cpu().numpy(), xt.std().cpu().numpy(), lr.mean().cpu().numpy(), lr.std().cpu().numpy(), sigma_t)
#             # print(result.mean().cpu().numpy(), result.std().cpu().numpy(), xt.mean().cpu().numpy(), xt.std().cpu().numpy(), lr.mean().cpu().numpy(), lr.std().cpu().numpy(), corrected.mean().cpu().numpy(), corrected.std().cpu().numpy(), sigma_t)

#         xt = torch.clamp(xt, min=-1.0, max=1.0)
#         # corrected = torch.clamp(lr + xt, min=-1.0, max=1.0)
#         self.learn.pred = (xt,) # just give the last category prediction
#         # self.learn.pred = (xt,predictions) # just give the last category prediction

#         raise CancelBatchException

#     def before_batch(self):
#         if not hasattr(self, 'gather_preds'): 
#             self.before_batch_training()
#         else: 
#             self.before_batch_sampling()


def get_chiplets_list(chiplet_dir:Path, max_chiplets:int=0):
    chiplets = set()
    for path in Path(chiplet_dir).glob("*.npz"):
        # path is something like: ecofuture_chiplet_level4_1988_subset_1_00004207.npz
        chiplet_components = path.name.split("_")
        subset = int(chiplet_components[5])
        chiplet_id = chiplet_components[6]
        chiplets.add( Chiplet(subset=subset, id=chiplet_id) )
    chiplets = list(chiplets)

    if max_chiplets and len(chiplets) > max_chiplets:
        random.seed(42)
        chiplets = random.sample(chiplets, max_chiplets)

    return chiplets
