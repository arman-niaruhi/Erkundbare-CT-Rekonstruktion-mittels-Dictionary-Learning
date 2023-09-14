import torch
from DL.KSVD_MODEL.KSVD_model import ApproximateKSVD
import time


class Dictionary_Learning():
    '''
    Here are more detailed steps:

        Load the noisy image and convert it to grayscale if it is in color.
        Choose the size of the patches you want to use for training the KSVD algorithm. A common patch size is 8 by 8 pixels but here 2 by 2.
        Extract small patches from the training images and stack them into a matrix X.
        Normalize each patch to have zero mean and unit variance.
        Use KSVD to learn a dictionary D of filters that can be used to efficiently represent the patches in a sparse manner.
        Extract patches from the noisy image and stack them into a matrix Y.
        Normalize each patch in input image to have zero mean and unit variance.
        Use the learned dictionary D to obtain a sparse representation of each patch in input image by solving an optimization problem that seeks to minimize 
        Reconstruct the denoised image by applying the learned dictionary D and Sparse representation.

    '''
    def __init__(self, device = "cuda") -> None:
        self.device = device

    # Patchify the image to the patches
    def patchify_the_image(self, image, patchsize = 12):
        """
        This help function helps to patchify the image and extract the standard deviation and mean of the normalization process
        ---------------
        Parameters:
            image:
                the input image
            patchsize:
                patch size for patchifying and convert the patchsizes and stack them horizontally 
        return:
            The stacked form of patches and mean and standard deviation
        """

        # Extract patches from the image
        patches = torch.nn.functional.unfold(image, kernel_size = patchsize,stride = patchsize)
        # Normalize the patches
        eps = 1e-6
        mean = torch.mean(patches, dim=1, keepdim=True)
        std = torch.std(patches, dim=1, keepdim=True)
        patches = (patches - mean) / (std + eps)
        patches = patches.to(torch.float32)
        std = std.to(torch.float32)
        mean = mean.to(torch.float32)
        return patches, mean, std


    # Image reconstruction
    def image_reconstruction(self, reconstructed_patches, std, mean, image, patchsize = 12, eps= 1e-6):
        
        """
        This help function helps to reconstruct the patches which are sparse now in the original form of image
        ---------------
        Parameters:
            reconstructed_patches:
                Sparse patches stacked together
            patchsize:
                Patch size for patchifying and convert the patchsizes and stack them horizontally 
            std:
                List of standard deviations
            mean:
                List of means
        return:
            the sparse image
        """
        
        # Unnormalize the image
        try: reconstructed_patches = reconstructed_patches.T * (std.cuda().to(torch.float32)+eps) + mean.cuda().to(torch.float32)
        except: pass
        try: reconstructed_patches = reconstructed_patches * (std.cuda().to(torch.float32)+eps) + mean.cuda().to(torch.float32)
        except: pass
        
        reconstructed_patches = torch.clamp(reconstructed_patches,0,1)
        reconstructed_patches = reconstructed_patches.view(1, patchsize*patchsize,-1)
        output_size = (image.shape[-2], image.shape[-1])

        reconstructed_image = torch.nn.functional.fold(reconstructed_patches, output_size = output_size 
                                                       ,kernel_size=patchsize, stride = patchsize)
        reconstructed_image = reconstructed_image.to(torch.float32)
        reconstructed_image = torch.nn.functional.interpolate(reconstructed_image,mode='bicubic',size=output_size)
        return reconstructed_image


    # Entire process in just on function
    def sparse_coding_with_ksvd(self, images, n_nonzero_coefs, patchsize, number_of_atoms = 65, device = "cuda"):
        """
        Finding Dictionary of the dictionary learning
        ---------------
        Parameters:
            Image:
                Original Image as input
            patchsize:
                Patch size for patchifying and convert the patchsizes and stack them horizontally 
            number_of_atoms:
                Select the number of atoms
            n_nonzero_coefs:
                Select the number of Nonezeros
        return:
            Dictionary structure
        """
        dictionary = ApproximateKSVD(number_atoms=number_of_atoms, 
                                transform_n_nonzero_coefs=n_nonzero_coefs, 
                                device=self.device, 
                                patchsize = patchsize
                              )
        dictionary.fit(images)
        return dictionary
     

    def predict(self, dic, x, patchsize = 12):
        """
        Given the dictionary and sparse representation finding the sparse reconstruction
        ---------------
        Parameters:
            dic:
                Input dictionary
            patchsize:
                Patch size for patchifying and convert the patchsizes and stack them horizontally 
            x:
                Input sparse representation
        return:
            Image reconstructed
        """
        patches, mean, std = self.patchify_the_image(image=x, patchsize=patchsize)
        patches = patches.squeeze()
        x2 = dic.transform(patches.T.squeeze())
        x2 = x2.T.to(torch.float32)
        reconstructed_patches = torch.mm(x2.T, dic.atoms)
        reconstructed_image = self.image_reconstruction(reconstructed_patches=reconstructed_patches,
                                                        std=std,mean=mean, patchsize=patchsize, 
                                                        image=x)
        reconstructed_image = reconstructed_image.to(torch.float32)

        return reconstructed_image




   


        




    



