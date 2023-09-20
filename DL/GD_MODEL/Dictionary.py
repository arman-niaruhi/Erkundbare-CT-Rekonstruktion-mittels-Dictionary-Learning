import torch
class Dictupdate(torch.nn.Module):
    def __init__(self, number_of_atoms, patchsize = 8):
        """
        In this class you see the main training and predicting 
        the Dictionary learning by using gradient descent and also l1 norm in explore.py 
        ---------------
        Parameter:
        
        number_of_atoms : number of atoms that you need for Dictionary
        patch_size : assigned to the size of each patch in the image
        patch : each patch in to original image wich should be calculated as a sparse representation   
        """
        super(Dictupdate, self).__init__()
        self.patch_size = patchsize
        self.number_of_atoms = number_of_atoms
        self.dictionary = torch.nn.Parameter(torch.rand(self.number_of_atoms, (patchsize**2)))
        # Dedicated Loss function
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        self.dictionary.data = torch.nn.functional.normalize(self.dictionary.data, p=2.0, dim=1, eps=1e-10)
        return torch.mm(self.dictionary.cuda().T, x.cuda()).to(torch.float32)
    
    def fit(self, sv, patch, optimizer, epoch_num_D):
        """
        Optimize the objective function
         ---------------
        Parameter:
            sv : current Sparse representation
            patch : input patchified input image
            optimizer : optimizer to optimize the object fuction
            epoch_num_D : the number of iteration for optimization
        """
        for _ in range(epoch_num_D):
            for p in patch:    
                predict = self.forward(sv)
                err = self.loss(predict, p.squeeze())
                self.error = err.cpu().detach().numpy()
                # Backward pass and optimizationtor
                optimizer.zero_grad()
                err.backward(retain_graph = True)
                optimizer.step()
                
                
                
            
                
        
