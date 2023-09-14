''' This code is taken and modified from https://github.com/drgHannah/Explorable_CT_Reconstruction. '''


from radon_transformation import radon
from DL.GD_MODEL.Sparsity import Sparse_net
from DL.GD_MODEL.Dictionary import Dictupdate
from DL.KSVD_MODEL.Denoising_helper import Dictionary_Learning as KSVD_dictionary
from network import BasicResnet
import itertools
import warnings
import numpy as np
import torchvision
import math
import dataset as ds
import utils
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")


class Optim:

    def __init__(self, angles, name="model.pt", device="cuda") -> None:
        self.device = device
        self.angles = angles
        self.name = name
        self.nosz = 16

        # load model
        self.model = self.load_network()
        self.model.to(device)

        # get data
        self.dataset = ds.IDLC_Dataset(angles=angles, mode="train")

        # Learning the dictionary
        self.DL_KSVD = KSVD_dictionary()

        # classification loss and model predition
        mse_loss = torch.nn.HuberLoss(delta=0.01)
        self.en_class = lambda x, model, pos, malig: mse_loss(torch.nn.functional.softmax(
            model(ds.crop_center(x, pos, self.nosz)), dim=1)[:, 1], malig)
        self.en_class_nocrop = lambda x, model, malig: mse_loss(
            torch.nn.functional.softmax(model(x), dim=1)[:, 1], malig)
        self.model_pred = lambda x, model, pos: torch.nn.functional.softmax(
            model(ds.crop_center(x, pos, self.nosz)), dim=1)[:, 1]
        self.psnr = lambda np_ground, np_compressed: np.log10(
            255**2/(np.mean((np_ground - np_compressed)**2))) * 10

        # total variation
        tova = utils.TV()
        self.c_tova = lambda xt2, loc, nosz: tova(
            ds.crop_center(xt2, loc, nosz))
        self.frame = []
        self.list_psnr = list()

    def load_network(self):
        ''' Load and return the pretrained classification network.
        '''

        # get path
        net_path = self.name
        # load network
        model = BasicResnet()
        model.load_state_dict(torch.load(net_path))
        model.eval()
        print("Loaded", net_path)
        return model

    def prepared_data(self, idx):
        ''' Load and prepare an data item. Define the radon loss E_1.

        args:
            idx: index of dataitem

        return:
            mean and std of cropped low dose, the true reconstruction, the sinogram, 
            the low dose, the location of the nodule of interest, 
            the malignancy of the nodule and the number of angles
        '''

        # get data
        slice, _, _, loc, malig, angles = self.dataset[idx]
        malig = malig[None].to(self.device)
        loc = loc[None]
        slice = slice[None]

        # load and apply radon transform
        self.radon_t, self.iradon_t = radon.get_operators(
            n_angles=angles, image_size=slice.shape[-1], circle=1, device='cpu')
        sino = self.radon_t(slice).float()
        low_dose = self.iradon_t(sino).float()

        # energy E_1
        radon_t, _ = radon.get_operators(
            n_angles=angles, image_size=slice.shape[-1], circle=1, device='cpu')
        radon_t.rotated = radon_t.rotated.to(self.device)
        self.en_radon = lambda x, sino: torch.mean((radon_t(x) - sino)**2)

        # mean and std of low dose data
        tmean = ds.crop_center(low_dose, loc, self.nosz).mean(
            [1, 2, 3])[:, None, None, None]
        tstd = ds.crop_center(low_dose, loc, self.nosz).std(
            [1, 2, 3])[:, None, None, None]
        return tmean.to(self.device), tstd.to(self.device), slice.to(self.device), \
            sino.to(self.device), low_dose.to(self.device), loc, malig, angles

    def create_gradient_mask(self, loc, bzs, shape_val=512+200, steps=4):
        ''' Creates and returns the gaussian mask G.

        args:
            loc: the location of the nodule
            bzs: the batchsize of the data
            shape_val: size of one axis of the reconstruction
            steps: std of the gaussian mask G

        return:
            gaussian mask
        '''

        steps = steps * (self.nosz // 16)
        gaussian_mask = []
        for i in range(bzs):
            gaussian_mask.append(torch.tensor(utils.makeGaussian(
                shape_val, fwhm=steps, center=np.array([loc[i][1], loc[i][0]])))[None])
        gaussian_mask = torch.stack(gaussian_mask, dim=0)
        return (gaussian_mask).to(self.device)

    def measure_error(self, loc, xt, sino):
        ''' Measures interior and exterior error (e_i and e_o).

        args:
            loc: location of the nodule
            xt: reconstruction
            sino: corresponding sinogram

        return:
            interior error, exterior error
        '''

        xt = xt.detach().cpu()
        sino = sino.detach().cpu()

        def get_mask(loc, shape_val=512):
            mask = torch.zeros([loc.shape[0], 1, shape_val, shape_val])
            for i in range(loc.shape[0]):
                mask[i, :, int(loc[i, 0]-self.nosz):int(loc[i, 0]+self.nosz),
                     int(loc[i, 1]-self.nosz):int(loc[i, 1]+self.nosz)] = 1
                mask_sino = self.radon_t(mask) > 0
            return mask, mask_sino

        _, mask_sino = get_mask(loc, xt.shape[-1])
        xt = xt.to(torch.float32)
        sinogram_calc = self.radon_t(xt)

        # apply mask on nodule
        masked_nodule_xt = sinogram_calc * mask_sino
        masked_nodule = sino * mask_sino
        norm_nodule = torch.sum(
            (masked_nodule_xt-masked_nodule)**2, dim=[2, 3])/torch.sum(mask_sino, dim=[2, 3])

        # apply mask on surrounding
        masked_sur_xt = sinogram_calc * ~mask_sino
        masked_sur = sino * ~mask_sino
        norm_sur = torch.sum((masked_sur_xt-masked_sur)**2,
                             dim=[2, 3])/torch.sum(~mask_sino, dim=[2, 3])
        return norm_nodule, norm_sur

    def add_perms(self, x, pos, tmean, tvar, perms=None):
        ''' Add permutations to the input.

        args:
            x: input reconstruction
            pos: position of the nodule
            tmean: mean value of the nodule (ideally after filtered backprojection)
            tvar: mean value of the nodule (ideally after filtered backprojection)
            perms: permutations to apply

        return:
            permutated input, tmean and tvar 
        '''

        if perms is not None:
            s1 = perms['shift']
            reszs = perms['zoom'](self.nosz * 2)
            rot_as = perms['rotate']
        else:
            s1 = [0]
            red = self.nosz * 2
            reszs = [red]
            rot_as = list(range(0, 360, 360//7))

        shifts = list(itertools.product(s1, s1))
        shifts = list(itertools.product(rot_as, shifts))
        perms = []
        tmeans = []
        tvars = []
        for rot_a, shift in shifts:
            xc = ds.crop_center(x, pos+torch.tensor(shift), self.nosz*2)
            xc = torchvision.transforms.functional.rotate(xc, angle=rot_a)
            xc = ds.crop_center(xc, torch.tensor(
                [[(xc.shape[2]//2), (xc.shape[3]//2)]]).repeat([xc.shape[0], 1]).float(), self.nosz)
            if shift == (0, 0) and reszs != []:
                for resz in reszs:
                    padv = (self.nosz*2 - resz)//2
                    xci = torch.nn.functional.interpolate(
                        xc, size=resz, mode='bilinear', align_corners=False)
                    xci = torch.nn.functional.pad(
                        xci, (padv, padv, padv, padv), mode='replicate')
                    perms.append(xci)
                    tmeans.append(tmean)
                    tvars.append(tvar)
            else:
                perms.append(xc)
                tmeans.append(tmean)
                tvars.append(tvar)
        return torch.cat(perms, 0), torch.cat(tmeans, 0), torch.cat(tvars, 0)

    def pretrain_lowdose(self, low_dose, sino):
        ''' Pre-calculation of the reconstruction.

        args:
            low_dose: initial reconstruction, ideally the low dose
            sino: corresponding sinogram

        return:
            optimized reconstruction
        '''
        print("Pre-calulate the input...")
        low_dose[low_dose > 1] = 1
        low_dose[low_dose < 0] = 0
        xt = low_dose.detach().clone().to(self.device).float()
        xt.requires_grad = True
        pbar = tqdm(range(600), disable=False)
        optim = torch.optim.SGD([xt], lr=0.0005, momentum=0.9)
        start_en = self.en_radon(xt, sino)
        for iter in pbar:
            optim.zero_grad()
            loss = self.en_radon(xt, sino)
            loss.backward()
            optim.step()
            with torch.no_grad():
                xt[xt > 1] = 1
                xt[xt < 0] = 0
        return xt.detach().clone().to(self.device).float()

    def optimize_GD(self, dataid, patchsize, epochs=2000, mask_w=11, w_r=1.0, w_c=1.0, w_tv=0.01, lr=1.0, lr_Dict=1.0,
                 epochs_Dict=10, epochs_SP=10, opt_to=None, perms=None, l1_lambda=0.1, number_of_atoms=64, size_of_crop = 32):
        ''' Data consistant optimization of the data towards a pre-defined malignancy.

        args:
            dataid: data index
            epochs: optimization iteration
            mask_w: width of the gaussian mask
            w_r: weighting of the radon loss E_1
            w_c: weighting of the classification loss (E_2 (part 1))
            w_tv: weighting of the tv loss (E_2 (part 2))
            lr: learning rate
            opt_to: optimization towrds this malignancy. If none, the malignancy is chosen to be opposite the true malignancy value.
            perms: permutation

        return:
            the resulting reconstruction, a list with the interior and exterior errors, 
            the loss, the radon loss, the classificatio loss E2, the predictions of the network, the stopping iteration,
            a tupel containing: (mean value of the low dose, std of the low dose, original rec., sinogram, low_dose, 
                                location of the nodule, malignancy, number of angles, final sinogram, low dose sinogram)

        '''
        # saves
        tmeans = []
        tstds = []
        mean_loss = []
        mean_loss_prev = 1e9
        stopiter = epochs
        cond = False
        Dict_err = 1e9
        
        def cropping(xt, loc, size_of_crop):
            return xt[0, 0, math.floor(float(loc[0, 0]))-size_of_crop:math.floor(float(loc[0, 0]))+size_of_crop, math.floor(
                float(loc[0, 1])) - size_of_crop:math.floor(float(loc[0, 1]))+size_of_crop].unsqueeze(dim=0).unsqueeze(dim=0)
        
        def patchify_the_image(image, patchsize=12):
            # Extract patches from the image
            patches = torch.nn.functional.unfold(
                image, kernel_size=patchsize, stride=patchsize)
            patches = patches.to(torch.float32)
            return patches

        # Image reconstruction
        def image_reconstruction(Dictionary, sparse_codes, image, patchsize=12, eps=1e-6):
            reconstructed_patches = torch.mm(Dictionary.cuda().to(torch.float32).T,
                                            sparse_codes.cuda().to(torch.float32))
            reconstructed_patches = torch.clamp(reconstructed_patches, 0, 1)

            reconstructed_patches = reconstructed_patches.view(
                1, patchsize*patchsize, -1)
            output_size = (image.shape[-2], image.shape[-1])

            reconstructed_image = torch.nn.functional.fold(
                reconstructed_patches, output_size=output_size, kernel_size=patchsize, stride=patchsize)
            reconstructed_image = reconstructed_image.to(torch.float32)
            return reconstructed_image

        with torch.no_grad():
            tmean, tstd, slice, sino, low_dose, loc, malig, angles = self.prepared_data(
                dataid)
            ld_sino = np.abs(self.radon_t(low_dose.cpu()) - sino.cpu())

        # pretrain on E_1
        pretrained_lowdose = self.pretrain_lowdose(low_dose, sino)
        low_dose = pretrained_lowdose
        low_dose[low_dose > 1] = 1
        low_dose[low_dose < 0] = 0
        xt = low_dose.detach().clone().to(self.device).float()

        # Networks for Dictionary learning
        # patch_for_size = patchify_the_image(image=cropping(xt, loc, size_of_crop), patchsize=patchsize)
        patch_for_size = patchify_the_image(image=xt, patchsize=patchsize)
        Net = Dictupdate(patchsize=patchsize, number_of_atoms=number_of_atoms)
        Net.error = 1e9
        sp_Net = Sparse_net(number_of_atoms=number_of_atoms,
                            num_patches=patch_for_size.squeeze().shape[1])
        xt.requires_grad = True
        sp_Net.sv.requires_grad = True

        # optimizers
        optim = torch.optim.SGD(filter(lambda p: p.requires_grad, [xt, sp_Net.sv]), lr=lr)  # + [sp_Net.sv]
        optimizer_dict = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, Net.parameters()), lr=lr_Dict)

        # optimize to
        with torch.no_grad():
            if opt_to is None:
                opt_to = 1.0-malig
            else:
                opt_to = torch.tensor(opt_to)[None].to(self.device)

        # gradient mask
        gradient_mask = self.create_gradient_mask(
            loc, low_dose.shape[0], shape_val=low_dose.shape[-1], steps=mask_w)
        # saves
        losses = []
        losses_r = []
        dict_loss = []
        spars_network_loss = []
        losses_c = []
        preds = []
        error_list = []
        self.frame = []
        # save iteration 0/eliasn97
        with torch.no_grad():
            self.frame.append(
                (ds.crop_center(low_dose, loc, size=4)).cpu().detach().numpy())
            preds.append(self.model_pred((low_dose-tmean)/tstd,
                         self.model, loc).cpu().detach().numpy())
        pbar = tqdm(range(epochs), disable=False)
        self.list_psnr = list()     

        for iter in pbar:
            # add gradient mask
            xt2 = gradient_mask * xt + (1 - gradient_mask) * xt.detach()
            # cropp the image to work on critical area related to the nodule
            # cropped = cropping(xt2, loc, size_of_crop)
            
            # patchify the image
            patch_ = patchify_the_image(xt2, patchsize=patchsize)
            # optimie the dictianry
            Net.fit(patch=[patch_], sv=sp_Net.sv, optimizer=optimizer_dict, epoch_num_D=epochs_Dict, cond=cond)
            iter_next = True
            for i in range(epochs_SP):
                optim.zero_grad()
                # learning the sparse representation assigned to the founded dictionary
                xt2 = gradient_mask * xt + (1 - gradient_mask) * xt.detach()
                patch_ = patchify_the_image(image=xt2, patchsize=patchsize)
                # finding the psnr
                with torch.no_grad():
                    np_ground = xt2.squeeze().cpu().numpy().astype(float)     
                # reconstruction of the sparse representation based on optimized sparsed matrix
                
                # cropped_res = image_reconstruction(
                #     sparse_codes=sp_Net.sv, Dictionary=Net.dictionary, patchsize=patchsize, image=xt2)
                
                # replace the sparsed image in x2
                xt2 = image_reconstruction(sparse_codes=sp_Net.sv, Dictionary=Net.dictionary, patchsize=patchsize, image=xt2)
                # xt2[0,0,math.floor(float(loc[0,0]))-size_of_crop:math.floor(float(loc[0,0]))+size_of_crop,math.floor(float(loc[0,1]))- size_of_crop:math.floor(float(loc[0,1]))+size_of_crop] = image_reconstruction(sparse_codes=sp_Net.sv, Dictionary=Net.dictionary, patchsize=patchsize, image=cropped_res)
               
                # finding the psnr
                with torch.no_grad():
                    np_compressed = xt2.squeeze().cpu().numpy().astype(float)
                    if iter_next is True:
                        self.list_psnr.append(self.psnr(np_ground, np_compressed))
                        iter_next = False
                # total variation
                tv = self.c_tova(xt2, loc, self.nosz)

                predict = sp_Net.forward(Net.dictionary)
                # Compute the the lasso
                l1_loss = (torch.norm(predict - patch_.cuda(), p='fro')) + \
                    l1_lambda * sum(p.abs().sum() for p in sp_Net.sv)
                # add permutation
                xt2, tmeans, tstds = self.add_perms(
                    xt2, loc, tmean, tstd, perms)
                # calculate loss
                en_c = self.en_class_nocrop(
                    (xt2-tmeans)/tstds, self.model, opt_to)
                en_r = self.en_radon(xt.to(torch.float32), sino)
                loss = w_r * en_r + w_c * en_c + w_tv * tv + l1_loss
                loss.backward(retain_graph=True)
                optim.step()

                with torch.no_grad():
                    mean_loss.append(loss.item())
                    if len(mean_loss) > 200:
                        cond = True

                    if Net.error > Dict_err:
                        optimizer_dict.param_groups[0]['lr'] = optimizer_dict.param_groups[0]['lr'] * 0.9
                    Dict_err = Net.error
                    # check every 500 iteration for updating the learning rate or early stopping
                    if optim.param_groups[0]['lr'] < 1e-4:
                        optim.param_groups[0]['lr'] = 1e-2
                    if optimizer_dict.param_groups[0]['lr'] < 1e-6:
                        optimizer_dict.param_groups[0]['lr'] = 1e-2
                    if len(mean_loss) > 10:
                        meanlosscalc = sum(mean_loss) / len(mean_loss)
                        condition = (meanlosscalc >= mean_loss_prev)
                        clr = optim.param_groups[0]['lr']

                        if condition:
                            optim.param_groups[0]['lr'] = clr * 0.9
                            mean_loss = []
                        else:
                            mean_loss_prev = meanlosscalc
                            mean_loss = []
                    # save loss and prediction
                    self.frame.append(
                        (ds.crop_center(xt, loc, size=2*self.nosz)).cpu().detach().numpy())
                    preds.append(self.model_pred((xt-tmean)/tstd,
                                 self.model, loc).cpu().detach().numpy())
                    losses.append(loss.item())
                    losses_r.append(en_r.item())  # (radon loss E_1)
                    losses_c.append(en_c.item())  # (class loss E_2 (w/o tv))
                    pbar.set_description(" Learning rate: {:.08f}, dict_lr: {:.08f}, loss: {:.04f}, radon: {:.04f}, class: {:.04f}, pred: {:.04f}, Dictionary loss: {:.06f}, psnr:{:.04f}".format(
                        optim.param_groups[0]['lr'], optimizer_dict.param_groups[0]['lr'], losses[-1], losses_r[-1], losses_c[-1], preds[-1][0], Net.error, self.list_psnr[-1]))
                    if (iter) % 100 == 0:
                        errors = self.measure_error(loc, xt, sino)
                        error_list.append((iter, errors))

                    if i == 2:
                        dict_loss.append(Net.error.item())
                        spars_network_loss.append(l1_loss.item())

        # save error and resulting sinogram
        with torch.no_grad():
            error_list.append((iter, self.measure_error(loc, xt, sino)))
            end_sino = np.abs(self.radon_t(
                xt.detach().cpu().to(torch.float32)) - sino.cpu())
        del self.en_radon
        torch.cuda.empty_cache()
        return dict_loss, spars_network_loss, self.list_psnr, xt.detach().cpu(), error_list, losses, losses_r, losses_c, preds, stopiter, \
            (tmean, tstd, slice.cpu(), sino.cpu(), low_dose.cpu(),
             loc.cpu(), malig, angles.item(), end_sino, ld_sino)

    def optimize_ksvd(self, dataid, non_zero_coefs, number_of_atoms=60, epochs=2000, mask_w=11, w_r=1.0, w_c=1.0, w_tv=0.01, lr=1.0,
                      opt_to=None, perms=None, patchsize=8):
        ''' Data consistant optimization of the data towards a pre-defined malignancy.

        args:
            dataid: data index
            epochs: optimization iteration
            mask_w: width of the gaussian mask
            w_r: weighting of the radon loss E_1
            w_c: weighting of the classification loss (E_2 (part 1))
            w_tv: weighting of the tv loss (E_2 (part 2))
            lr: learning rate
            opt_to: optimization towrds this malignancy. If none, the malignancy is chosen to be opposite the true malignancy value.
            perms: permutation

        return:
            the resulting reconstruction, a list with the interior and exterior errors, 
            the loss, the radon loss, the classification loss E2, the predictions of the network, the stopping iteration,
            a tupel containing: (mean value of the low dose, std of the low dose, original rec., sinogram, low_dose, 
                                location of the nodule, malignancy, number of angles, final sinogram, low dose sinogram)

        '''
        # saves
        tmeans = []
        tstds = []
        mean_loss = []
        mean_loss_prev = 1e9
        # Help function to create the dictionary of dictionary learning
        def learning_Dic(dataset, non_zero_coefs, number_of_atoms, patchsize):
            dictionary = self.DL_KSVD.sparse_coding_with_ksvd(n_nonzero_coefs=non_zero_coefs,
                                                        images=dataset,
                                                        number_of_atoms=number_of_atoms,
                                                        patchsize=patchsize,
                                                        device=self.device)
            return dictionary

        with torch.no_grad():
            tmean, tstd, slice, sino, low_dose, loc, malig, angles = self.prepared_data(
                dataid)
            ld_sino = np.abs(self.radon_t(low_dose.cpu()) - sino.cpu())

        # pretrain on E_1
        pretrained_lowdose = self.pretrain_lowdose(low_dose, sino)
        low_dose = pretrained_lowdose
        low_dose[low_dose > 1] = 1
        low_dose[low_dose < 0] = 0
        xt = low_dose.detach().clone().to(self.device).float()
        xt.requires_grad = True
        # optimizer
        optim = torch.optim.SGD([xt], lr=lr)
        # optimize to
        with torch.no_grad():
            if opt_to is None:
                opt_to = 1.0-malig
            else:
                opt_to = torch.tensor(opt_to)[None].to(self.device)
        # gradient mask
        gradient_mask = self.create_gradient_mask(
            loc, low_dose.shape[0], shape_val=low_dose.shape[-1], steps=mask_w)
        # saves
        losses = []
        losses_r = []
        losses_c = []
        preds = []
        error_list = []
        self.frame = []
        # save iteration 0
        with torch.no_grad():
            self.frame.append(
                (ds.crop_center(low_dose, loc, size=2*self.nosz)).cpu().detach().numpy())
            preds.append(self.model_pred((low_dose-tmean)/tstd,
                         self.model, loc).cpu().detach().numpy())
        pbar = tqdm(range(epochs), disable=False)
        stopiter = epochs
        for iter in pbar:
            # add gradient mask
            xt2 = gradient_mask * xt + (1 - gradient_mask) * xt.detach()
            # help function for psnr
            with torch.no_grad():
                np_ground = xt2.squeeze().detach().cpu().numpy().astype(float)
            # make the dictionary using the help function that is defined in the line 493
            Dictionary = learning_Dic(dataset=[xt], non_zero_coefs=non_zero_coefs,
                                           number_of_atoms=number_of_atoms, patchsize=patchsize)
         
            xt2 = self.DL_KSVD.predict(Dictionary, xt2, patchsize=patchsize)
            # help function for psnr
            with torch.no_grad():
                np_compressed = xt2.squeeze().detach().cpu().numpy().astype(float)
                self.list_psnr.append(self.psnr(np_ground, np_compressed))
            # total variation
            tv = self.c_tova(xt2, loc, self.nosz)
            # add permutation
            xt2, tmeans, tstds = self.add_perms(xt2, loc, tmean, tstd, perms)
            # calculate loss
            en_c = self.en_class_nocrop((xt2-tmeans)/tstds, self.model, opt_to)
            en_r = self.en_radon(xt.to(torch.float32), sino)
            loss = w_r * en_r + w_c * en_c + w_tv * tv
            loss = loss.to(torch.float32)
            loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()

            with torch.no_grad():
                xt[xt > 1] = 1
                xt[xt < 0] = 0

            with torch.no_grad():
                mean_loss.append(loss.item())
                # check every 500 iteration for updating the learning rate or early stopping
                if len(mean_loss) > 10:
                    meanlosscalc = sum(mean_loss) / len(mean_loss)
                    condition = (meanlosscalc >= mean_loss_prev)
                    clr = optim.param_groups[0]['lr']
                    if condition:
                        optim.param_groups[0]['lr'] = clr * 0.9
                        mean_loss = []
                    else:
                        mean_loss_prev = meanlosscalc
                        mean_loss = []
                # save loss and prediction
                self.frame.append(
                    (ds.crop_center(xt, loc, size=2*self.nosz)).cpu().detach().numpy())
                preds.append(self.model_pred((xt-tmean)/tstd,
                             self.model, loc).cpu().detach().numpy())
                losses.append(loss.item())
                losses_r.append(en_r.item())  # (radon loss E_1)
                losses_c.append(en_c.item())  # (class loss E_2 (w/o tv))
                pbar.set_description("Learning Rate: {:.14f}, loss: {:.04f}, radon: {:.04f}, class: {:.04f}, pred: {:.04f}, psnr: {:.04f}".format(optim.param_groups[0]['lr'],
                                                                                                                                                  losses[-1], losses_r[-1], losses_c[-1], preds[-1][0], self.list_psnr[-1]))

                if (iter+1) % 1000 == 0:
                    errors = self.measure_error(loc, xt, sino)
                    error_list.append((iter, errors))
                    
        # save error and resulting sinogram
        with torch.no_grad():
            error_list.append((iter, self.measure_error(loc, xt, sino)))
            end_sino = np.abs(self.radon_t(
                xt.detach().cpu().to(torch.float32)) - sino.cpu())
        del self.en_radon
        torch.cuda.empty_cache()
        return self.list_psnr, xt.detach().cpu(), error_list, losses, losses_r, losses_c, preds, stopiter, \
            (tmean, tstd, slice.cpu(), sino.cpu(), low_dose.cpu(),
             loc.cpu(), malig, angles.item(), end_sino, ld_sino)

