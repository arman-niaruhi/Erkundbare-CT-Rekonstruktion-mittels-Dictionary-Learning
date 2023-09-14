import scipy as sp
import numpy as np
from sklearn.linear_model import orthogonal_mp
import torch



class ApproximateKSVD(object):
    def __init__(self, number_atoms, patchsize, tol=1e-3,
                 transform_n_nonzero_coefs=None, device="cuda"):
        """
        Parameters
        ----------
        number_atoms:
            Number of dictionary atoms
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.total_D = []
        self.atoms = None
        self.tol = tol
        self.number_atoms = number_atoms
        self.transform_non_zero_coefs = transform_n_nonzero_coefs
        self.device = device
        self.patchsize = patchsize

    def dictionary_initialize(self, X):
        """
        Initialize the Dictionary using svd
        ----------
        Parameters
        ----------
        X:
            Original Image
        ----------
        return: 
            the initialized dictionary or modified dictionary
        """
        
        # Initialize matrix randomly if the original image has bigger dimension rather than number of atoms
        if min(X.shape) < self.number_atoms:
            D = torch.rand(self.number_atoms, X.shape[1]).to(self.device)
        else:
            # If the number atoms is bigger then find the dictionary using Singular Value Decompostion
            X = X.detach().cpu().numpy()
            u, s, vt = sp.sparse.linalg.svds(X.T, k=self.number_atoms)
            D = np.dot(np.diag(s), vt)
            D = torch.tensor(D)
        D = D/torch.norm(D, dim=1, keepdim=True)
        D = D.to(self.device).float()
        return D
    
    def _update_dict(self, X, D, y):
        """
        This function is provided to update the dictionary. 
        
        Parameters
        ----------
        X:
            Original Image
        D:
            Dictionary
        y: 
            Sparse representation
        ----------
        return:
            Updated dictionary D and sparse representation
        """
        y = y.to(torch.float32)
        for j in range(self.number_atoms):
            if torch.sum(y[:, j] > 0) == 0:
                continue

            D[j, :] = torch.zeros_like(D[j, :])
            g = y[y[:, j] > 0, j].unsqueeze(
                dim=1).cuda().to(torch.float32)
            D = D.to(torch.float32).clone()
            X = X.cuda().clone()
            y = y.cuda().clone()

            r = X[:, y[:, j] > 0].T - \
                torch.mm(y[y[:, j] > 0, :], D).to(
                    torch.float32).clone()
            d = torch.mm(g.T, r)
            d = d / torch.norm(d).clone()
            g = torch.mm(r, d.T)
            D[j, :] = d.clone()
            y[y[:, j] > 0, j] = g.T.clone()

        return D, y

    def _transform(self, D, X):
        """
        In this function the Orthogonal Matching Pursuit (OMP) is used to solve the objective function X = DY 
        ----------
        Parameters
        ----------
        D: 
            Dictionary that contains all the atoms
        X: 
            Input Image tensor
        ----------
        return:
            founded Sparse representation
        """
        D = D.to(self.device).float()
        X = X.to(self.device).float()
        n_nonzero_coefs = int(self.transform_non_zero_coefs)

        # Rotate the matrix to apply on the big number of atoms and reduce the time
        if D.shape[0] > D.shape[1]:
            Y = torch.tensor(orthogonal_mp(y=X.T.cpu().detach().numpy(),
                                              X=D.T.cpu().detach().numpy(),
                                              precompute=True,
                                              tol=9e-1,
                                              n_nonzero_coefs=n_nonzero_coefs).T).to(self.device).float()
        else:
            Y = torch.tensor(orthogonal_mp(y=X.cpu().detach().numpy(),
                                              X=D.T.cpu().detach().numpy(),
                                              precompute=True,
                                              tol=9e-1,
                                              n_nonzero_coefs=n_nonzero_coefs).T).to(self.device).float()   
        return Y

    def fit(self, X):
        """
        Parameters
        ----------
        X: 
            shape = [n_samples, n_features]
        ----------
        Here the image would be reconstructed using the given X and computed Dictionary
        """
        for x in X:
            torch.cuda.empty_cache()
            patches, mean, std = self.patchify_the_image(
                image=x, patchsize=self.patchsize)
            xD = patches.squeeze().T
            D = self.dictionary_initialize(xD)
            D = D.to(torch.float32)
            for _ in range(0):
                # Give a batch of Instances to the model
                y = self._transform(D, xD)
                y = y.to(torch.float32)
                if D.shape[0] > D.shape[1]:
                    e = torch.norm(
                        xD.cuda() - torch.mm(y.cuda(), D.cuda()))
                else:
                    e = torch.norm(
                        xD.cuda() - torch.mm(y.cuda(), D.cuda()).T)
                if e < self.tol:
                    break
                if D.shape[0] > D.shape[1]:
                    D, y = self._update_dict(xD, y.T, D.T)
                else:
                    D, y = self._update_dict(xD, D, y)

            self.total_D.append(D)

            self.atoms = D.clone()

        return self

    def transform(self, X):
        return self._transform(self.atoms, X)

    # Patchify the image to the patches
    def patchify_the_image(self, image, patchsize=12):
        """
        patchifying the image using folding and unfolding
        ----------
        Parameters:
            image: original input image
        ----------
        return:
            all of the patches and assined mean and standard deviation
        """
        # Extract patches from the image
        patches = torch.nn.functional.unfold(
            image, kernel_size=patchsize, stride=patchsize)
        
        # Normalize the patches
        eps = 1e-6
        mean = torch.mean(patches, dim=1, keepdim=True)
        std = torch.std(patches, dim=1, keepdim=True)
        patches = (patches - mean) / (std + eps)
        patches = patches.to(torch.float32)
        std = std.to(torch.float32)
        mean = mean.to(torch.float32)
        return patches, mean, std

  